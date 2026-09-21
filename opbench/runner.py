"""PyTorch benchmark CLI. GPU libraries are optional and imported only here."""
import argparse
import importlib
import json
import math
import os
import platform
import statistics
import time
import urllib.request
from pathlib import Path
from .catalog import expand_template, workload


def make_operation(torch, case, device):
    p, op, stage = case["params"], case["operator"], case["stage"]
    dtype = getattr(torch, case["dtype"])
    backward = stage == "backward"
    inputs = []

    def tensor(shape):
        x = torch.randn(shape, device=device, dtype=dtype, requires_grad=backward)
        inputs.append(x)
        return x

    reset = lambda: None
    if case["category"] == "matrix":
        m, n, k = (p[x] for x in ("m", "n", "k"))
        batch = p.get("batch", 1)
        batched = op in ("bmm", "matmul", "addbmm", "baddbmm")
        a = tensor((batch, m, k) if batched else (m, k))
        b = tensor((batch, k, n) if batched else (k, n))
        if op in ("addmm", "addbmm", "baddbmm"):
            c = tensor((batch, m, n) if op == "baddbmm" else (m, n))
            forward = lambda: getattr(torch, op)(c, a, b)
        else:
            forward = lambda: getattr(torch, op)(a, b)
    elif case["category"] in ("convolution", "transpose_convolution"):
        cls = getattr(torch.nn, "Conv2d" if op == "Conv2dPointwise" else op)
        module = cls(p["cin"], p["cout"], p["kernel"], stride=p["stride"], padding=p["padding"], groups=p["groups"], bias=False).to(device=device, dtype=dtype)
        x = tensor((p["batch"], p["cin"], *p["spatial"]))
        inputs += list(module.parameters())
        module.train(case["module_mode"] == "train")
        forward = lambda: module(x)
    elif case["category"] == "activation":
        module = getattr(torch.nn, op)().to(device=device, dtype=dtype)
        x = tensor((p["elements"],))
        forward = lambda: module(x)
    elif case["category"] == "normalization":
        shape = p["shape"]
        if op.startswith("BatchNorm"):
            module = getattr(torch.nn, op)(shape[1])
        elif op == "GroupNorm":
            module = torch.nn.GroupNorm(p.get("groups", 1), shape[1])
        elif op == "LayerNorm":
            module = torch.nn.LayerNorm(shape[-1])
        elif op == "RMSNorm":
            if not hasattr(torch.nn, "RMSNorm"):
                raise NotImplementedError("native RMSNorm requires newer PyTorch")
            module = torch.nn.RMSNorm(shape[-1])
        module = module.to(device=device, dtype=dtype)
        module.train(case["module_mode"] == "train")
        x = tensor(shape)
        inputs += list(module.parameters())
        forward = lambda: module(x)
    else:
        x = torch.nn.Parameter(torch.randn(p["elements"], device=device, dtype=dtype))
        x.grad = torch.full_like(x, 0.01)
        cls = {"sgd": "SGD", "adagrad": "Adagrad", "adam": "Adam", "adamw": "AdamW", "rmsprop": "RMSprop"}[op]
        optimizer = getattr(torch.optim, cls)([x], lr=0.001, foreach=False)
        optimizer.step()  # Initialize state outside all measured intervals.
        baseline = x.detach().clone()
        state = {k: v.clone() if torch.is_tensor(v) else v for k, v in optimizer.state.get(x, {}).items()}

        def reset():
            with torch.no_grad():
                x.copy_(baseline)
                for key, value in state.items():
                    current = optimizer.state[x][key]
                    if torch.is_tensor(current):
                        current.copy_(value)
                    else:
                        optimizer.state[x][key] = value

        def step():
            optimizer.step()
            return x
        return step, reset
    if backward:
        output = forward()  # Graph construction is explicitly outside backward timing.
        gradient = torch.ones_like(output)
        return lambda: torch.autograd.grad(output, inputs, gradient, retain_graph=True), reset

    def inference():
        with torch.no_grad():
            return forward()
    return inference, reset


def device_api(torch, device):
    kind = device.split(":")[0]
    if kind == "cpu":
        return None
    api = getattr(torch, kind, None)
    if api is None or not api.is_available():
        raise RuntimeError(f"device backend {kind} is unavailable; install its PyTorch extension")
    if hasattr(api, "set_device"):
        api.set_device(device)
    return api


def measure(torch, case, device, warmup, iterations):
    api = device_api(torch, device)
    sync = api.synchronize if api else lambda: None
    operation, reset = make_operation(torch, case, device)
    for _ in range(warmup):
        reset()
        result = operation()
        sync()
        del result
    # Numerical sanity only, explicitly NOT an independent correctness reference.
    reset()
    result = operation()
    values = result if isinstance(result, tuple) else (result,)
    finite = all(bool(torch.isfinite(x).all().item()) for x in values)
    del result, values
    if not finite:
        raise ValueError("non-finite output/gradient")
    wall, gpu, cpu = [], [], []
    events = None
    if api and hasattr(api, "Event"):
        try:
            events = (api.Event(enable_timing=True), api.Event(enable_timing=True))
            events[0].record()
            events[1].record()
            sync()
            events[0].elapsed_time(events[1])
        except (TypeError, RuntimeError, NotImplementedError):
            events = None
    memory = api and all(hasattr(api, n) for n in ("memory_allocated", "reset_peak_memory_stats", "max_memory_allocated"))
    sync()
    base = api.memory_allocated() if memory else None
    if memory:
        api.reset_peak_memory_stats()
    for _ in range(iterations):
        reset()
        sync()
        if events:
            events[0].record()
        t0 = time.perf_counter_ns()
        result = operation()
        t1 = time.perf_counter_ns()
        if events:
            events[1].record()
        sync()
        t2 = time.perf_counter_ns()
        wall.append(max((t2-t0)/1000, 0.001))
        if api:
            cpu.append((t1-t0)/1000)
        if events:
            gpu.append(max(events[0].elapsed_time(events[1])*1000, 0.001))
        del result
    peak = api.max_memory_allocated() if memory else None
    return {**case, **workload(case), "status": "pass", "correctness": "finite_only",
            "wall_us": statistics.median(wall), "gpu_us": statistics.median(gpu) if gpu else None,
            "cpu_us": statistics.median(cpu) if cpu else None,
            "p95_us": sorted(wall)[math.ceil(0.95*len(wall))-1], "stddev_us": statistics.pstdev(wall),
            "peak_allocated_bytes": peak, "peak_delta_bytes": max(0, peak-base) if peak is not None else None,
            "samples_us": wall}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", default="templates/smoke.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend-module", help="Explicit trusted extension module, e.g. torch_mlu or torch_musa")
    parser.add_argument("--backend-version", default="unknown")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--name", default="Operator benchmark")
    parser.add_argument("--output", default="results/run.json")
    parser.add_argument("--upload", help="OpBench server URL; token read from OPBENCH_API_TOKEN")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.warmup <= 10000 or not 1 <= args.iterations <= 10000 or args.threads < 1:
        parser.error("invalid warmup/iterations/threads")
    template = json.loads(Path(args.template).read_text(encoding="utf-8"))
    cases = expand_template(template)
    if args.dry_run:
        print(json.dumps({"count": len(cases), "cases": cases}, ensure_ascii=False, indent=2))
        return
    try:
        import torch
    except ImportError:
        parser.error("PyTorch is not installed. Install the vendor-supported PyTorch build first.")
    if args.backend_module:
        importlib.import_module(args.backend_module)
    api = device_api(torch, args.device)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    device_name = api.get_device_name() if api and hasattr(api, "get_device_name") else platform.processor() or args.device
    run = {"name": args.name, "device": {"name": device_name, "backend": args.device.split(":")[0], "platform": platform.platform()},
           "torch_version": torch.__version__, "backend_version": args.backend_version,
           "timing_method": "synchronized_wall_per_iteration", "precision_policy": "highest; TF32 disabled where exposed",
           "cpu_threads": args.threads, "warmup": args.warmup, "iterations": args.iterations, "seed": args.seed,
           "template": template.get("name", args.template), "synthetic": False,
           "timing_note": "Median synchronized wall includes dispatch and terminal synchronization. GPU events may include stream idle time. CPU enqueue excludes final synchronization. Backward graph prepared before timing. CPU stream/enqueue/memory metrics are null."}
    report = {"schema_version": 1, "run": run, "results": []}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    for index, case in enumerate(cases):
        try:
            record = measure(torch, case, args.device, args.warmup, args.iterations)
        except Exception as exc:
            message = str(exc)
            status = "oom" if "out of memory" in message.lower() else "unsupported" if isinstance(exc, NotImplementedError) or "not implemented" in message.lower() else "failed"
            record = {**case, "status": status, "error": message, "correctness": "not_checked"}
            if api and hasattr(api, "empty_cache"):
                api.empty_cache()
        report["results"].append(record)
        temp = target.with_suffix(".tmp")
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temp.replace(target)
        print(f"[{index+1}/{len(cases)}] {case['operator']} {case['stage']} {record['status']}", flush=True)
    if args.upload:
        headers = {"Content-Type": "application/json"}
        token = os.environ.get("OPBENCH_API_TOKEN")
        if token:
            headers["Authorization"] = "Bearer " + token
        request = urllib.request.Request(args.upload.rstrip("/")+"/api/import", data=json.dumps(report).encode(), headers=headers)
        with urllib.request.urlopen(request, timeout=60) as response:
            print(response.read().decode())
    failures = sum(x["status"] != "pass" for x in report["results"])
    print(f"Saved {target}; {failures} unsuccessful cases")
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

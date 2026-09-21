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
from .catalog import expand_template, workload, normalize_case


def addbmm_loop(torch, bias, batch1, batch2):
    """Explicit device-side composition for alpha=beta=1; not a fused kernel."""
    result = bias
    for index in range(batch1.shape[0]):
        result = torch.addmm(result, batch1[index], batch2[index])
    return result


def addbmm_composite(torch, bias, batch1, batch2):
    """GPU bmm/reduction with FP32 accumulation for low precision inputs."""
    dtype = torch.float32 if bias.dtype in (torch.float16, torch.bfloat16) else bias.dtype
    return (torch.bmm(batch1.to(dtype), batch2.to(dtype)).sum(0) + bias.to(dtype)).to(bias.dtype)


def make_operation(torch, case, device, reference_dtype=None):
    p, op, stage = case["params"], case["operator"], case["stage"]
    dtype = getattr(torch, case["dtype"])
    compute_dtype = getattr(torch, reference_dtype) if reference_dtype else dtype
    backward = stage == "backward"
    inputs = []

    def tensor(shape):
        # Initialization is outside timing; CPU generation makes inputs portable
        # across backends and does not require a vendor random-number GPU kernel.
        x = torch.randn(shape, device="cpu", dtype=dtype).to(device=device, dtype=compute_dtype).requires_grad_(backward)
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
            if case.get("implementation") == "addmm_loop_v1":
                forward = lambda: addbmm_loop(torch, c, a, b)
            elif case.get("implementation") == "bmm_fp32_sum_v1":
                forward = lambda: addbmm_composite(torch, c, a, b)
            else:
                forward = lambda: getattr(torch, op)(c, a, b)
        else:
            forward = lambda: getattr(torch, op)(a, b)
    elif case["category"] in ("convolution", "transpose_convolution"):
        cls = getattr(torch.nn, "Conv2d" if op == "Conv2dPointwise" else op)
        module = cls(p["cin"], p["cout"], p["kernel"], stride=p["stride"], padding=p["padding"], groups=p["groups"], bias=False).to(dtype=dtype).to(device=device, dtype=compute_dtype)
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
        if op == "RMSNorm":
            module.eps = torch.finfo(dtype).eps
        module = module.to(dtype=dtype).to(device=device, dtype=compute_dtype)
        module.train(case["module_mode"] == "train")
        x = tensor(shape)
        inputs += list(module.parameters())
        forward = lambda: module(x)
    else:
        x = torch.nn.Parameter(torch.randn(p["elements"], device="cpu", dtype=dtype).to(device=device, dtype=compute_dtype))
        x.grad = torch.full((p["elements"],), 0.01, dtype=dtype).to(device=device, dtype=compute_dtype)
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


def configure_precision(torch):
    """Disable exposed TF32 controls, including the vendor muDNN switch."""
    torch.set_float32_matmul_precision("highest")
    settings = {"float32_matmul_precision": torch.get_float32_matmul_precision()}
    for path in ("cuda.matmul.allow_tf32", "cudnn.allow_tf32", "cudnn.benchmark",
                 "musa.matmul.allow_tf32", "mudnn.allow_tf32"):
        parts = path.split(".")
        owner = torch.backends
        for name in parts[:-1]:
            owner = getattr(owner, name, None)
            if owner is None:
                break
        if owner is not None and hasattr(owner, parts[-1]):
            setattr(owner, parts[-1], False)
            settings[path] = bool(getattr(owner, parts[-1]))
    return settings


class ReferenceMismatch(AssertionError):
    def __init__(self, reference):
        self.reference = reference
        super().__init__(f"CPU reference tolerance exceeded: {reference}")


def verify_reference(torch, case, device):
    """Compare quantized inputs with the native CPU float64 operation.

    Restores the CPU RNG so the subsequent timed operation sees the same inputs.
    All copies and comparisons are outside timing.
    """
    tolerances = {"float64": (1e-8, 1e-8), "float32": (1e-3, 1e-4),
                  "float16": (1e-2, 1e-2), "bfloat16": (5e-2, 5e-2)}
    rtol, atol = tolerances[case["dtype"]]
    state = torch.random.get_rng_state()

    def outputs(operation):
        result = operation()
        values = result if isinstance(result, tuple) else (result,)
        return [x.detach().cpu().to(dtype=torch.float64).clone() for x in values]

    try:
        native = normalize_case({**case, "implementation": "native", "name": None})
        cpu_op, cpu_reset = make_operation(torch, native, "cpu", reference_dtype="float64")
        cpu_reset()
        expected = outputs(cpu_op)
        torch.random.set_rng_state(state)
        target_op, target_reset = make_operation(torch, case, device)
        target_reset()
        actual = outputs(target_op)
        if len(actual) != len(expected):
            raise AssertionError("reference output count mismatch")
        max_abs = max_scaled = 0.0
        for got, wanted in zip(actual, expected):
            if not bool(torch.isfinite(got).all()) or not bool(torch.isfinite(wanted).all()):
                raise ValueError("non-finite reference output/gradient")
            if got.shape != wanted.shape:
                raise AssertionError("reference output shape mismatch")
            error = (got-wanted).abs()
            max_abs = max(max_abs, error.max().item())
            max_scaled = max(max_scaled, (error/(atol+rtol*wanted.abs())).max().item())
        reference = {"device": "cpu", "implementation": "native", "rtol": rtol, "atol": atol,
                     "max_abs_error": max_abs, "max_scaled_error": max_scaled,
                     "compute_dtype": "float64", "input_dtype": case["dtype"]}
        if max_scaled > 1:
            raise ReferenceMismatch(reference)
        return reference
    finally:
        torch.random.set_rng_state(state)


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
    parser.add_argument("--addbmm-implementation", choices=("native", "composite"), help="Explicitly select the separately labelled bmm+FP32 accumulation implementation")
    parser.add_argument("--verify-reference", action="store_true", help="Check each case against the native CPU operation outside timing")
    parser.add_argument("--allow-arch-mismatch", action="store_true", help="Diagnostic override for a MUSA extension built for another GPU architecture")
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
    if args.addbmm_implementation:
        implementation = "native" if args.addbmm_implementation == "native" else "bmm_fp32_sum_v1"
        cases = [normalize_case({**case, "implementation": implementation, "name": None})
                 if case["operator"] == "addbmm" else case for case in cases]
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
    arch_list = api.get_arch_list() if api and hasattr(api, "get_arch_list") else []
    device_arch = None
    if args.device.split(":")[0] == "musa" and hasattr(api, "get_device_capability"):
        major, minor = api.get_device_capability()
        device_arch = f"{major}{minor}"
        if device_arch not in arch_list and not args.allow_arch_mismatch:
            parser.error(f"MUSA extension architectures {arch_list} do not include device architecture {device_arch}; rebuild TORCH_MUSA_ARCH_LIST={device_arch}")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    precision_settings = configure_precision(torch)
    device_name = api.get_device_name() if api and hasattr(api, "get_device_name") else platform.processor() or args.device
    run = {"name": args.name, "device": {"name": device_name, "backend": args.device.split(":")[0], "platform": platform.platform()},
           "torch_version": torch.__version__, "backend_version": args.backend_version,
           "timing_method": "synchronized_wall_per_iteration", "precision_policy": "highest; TF32 disabled where exposed",
           "precision_settings": precision_settings,
           "input_initialization": "cpu_then_copy",
           "operator_implementations": {c["operator"]: c["implementation"] for c in cases if c["implementation"] != "native"},
           "reference_check": args.verify_reference, "reference_method": "native_cpu_float64_quantized_inputs_v1" if args.verify_reference else "none", "backend_arch_list": arch_list, "device_arch": device_arch,
           "runtime_image": os.environ.get("OPBENCH_RUNTIME_IMAGE", "unknown"),
           "cpu_threads": args.threads, "warmup": args.warmup, "iterations": args.iterations, "seed": args.seed,
           "template": template.get("name", args.template), "synthetic": False,
           "timing_note": "Median synchronized wall includes dispatch and terminal synchronization. GPU events may include stream idle time. CPU enqueue excludes final synchronization. Backward graph prepared before timing. CPU stream/enqueue/memory metrics are null."}
    report = {"schema_version": 1, "run": run, "results": []}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    for index, case in enumerate(cases):
        try:
            reference = verify_reference(torch, case, args.device) if args.verify_reference else None
            record = measure(torch, case, args.device, args.warmup, args.iterations)
            if reference:
                record.update(correctness="cpu_reference", reference=reference)
        except Exception as exc:
            message = str(exc)
            status = "oom" if "out of memory" in message.lower() else "unsupported" if isinstance(exc, NotImplementedError) or "not implemented" in message.lower() else "failed"
            record = {**case, "status": status, "error": message, "correctness": "not_checked"}
            if isinstance(exc, ReferenceMismatch):
                record.update(correctness="cpu_reference_failed", reference=exc.reference)
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

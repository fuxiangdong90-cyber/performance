"""Deterministic synthetic fixtures, never represented as real hardware results."""
import json
import random
from pathlib import Path
from .catalog import expand_template, workload


def reports():
    cases = expand_template(json.loads((Path(__file__).resolve().parent.parent/"templates"/"standard.json").read_text(encoding="utf-8")))
    for variant in (0, 1):
        rng = random.Random(17)
        results = []
        for index, case in enumerate(cases):
            w = workload(case)
            base = 8 + (w["flops"] or 0)/2e7 + (w["bytes"] or 1000000)/2e5
            if case["stage"] == "backward":
                base *= 2.5
            t = base/(rng.uniform(0.65, 3.2) if variant else 1)
            status = "unsupported" if variant and index % 37 == 0 else "pass"
            results.append({**case, "status": status, "wall_us": t if status == "pass" else None,
                            "gpu_us": t*0.86, "cpu_us": t*0.25, "p95_us": t*1.08, "stddev_us": t*0.025,
                            "peak_allocated_bytes": (w["bytes"] or 2097152)*2,
                            "peak_delta_bytes": w["bytes"] or 2097152, "samples_us": [], "correctness": "synthetic",
                            "error": "Synthetic unsupported example" if status != "pass" else ""})
        yield {"schema_version": 1, "run": {"name": f"DEMO {'B' if variant else 'A'} · 合成演示 · Eager", "synthetic": True,
                "device": {"name": f"Synthetic Accelerator {'B' if variant else 'A'}", "backend": "demo"},
                "timing_method": "synchronized_wall_per_iteration", "precision_policy": "synthetic", "torch_version": "demo",
                "warmup": 10, "iterations": 50}, "results": results}

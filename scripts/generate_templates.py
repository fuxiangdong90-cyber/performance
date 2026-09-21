"""Rebuild the committed templates; run from repository root."""
import json
from pathlib import Path
from opbench.catalog import OPERATORS

specs = []
for op, category in OPERATORS.items():
    if category == "matrix":
        params = [{"m": m, "n": n, "k": k, **({"batch": b} if op in ("bmm", "matmul", "addbmm", "baddbmm") else {})}
                  for m, n, k, b in [(64, 64, 64, 4), (256, 256, 64, 8), (256, 1024, 128, 8), (1024, 1024, 1024, 4)]]
        dtypes = ["float32", "float16", "bfloat16"]
    elif category in ("convolution", "transpose_convolution"):
        dims = 2 if op == "Conv2dPointwise" else int(op[-2])
        params = [{"batch": 2, "cin": c, "cout": c, "spatial": [16]*dims, "kernel": 1 if op == "Conv2dPointwise" else 3,
                   "stride": stride, "padding": 0, "groups": 1} for c in (8, 32) for stride in (1, 2)]
        dtypes = ["float32"]
    elif category == "normalization":
        shape = {"BatchNorm1d": [4, 16, 64], "BatchNorm2d": [4, 16, 16, 16], "BatchNorm3d": [2, 8, 8, 8, 8],
                 "GroupNorm": [4, 16, 16, 16], "LayerNorm": [8, 256, 64], "RMSNorm": [8, 256, 64]}[op]
        params = [{"shape": shape, **({"groups": 4} if op == "GroupNorm" else {})}]
        dtypes = ["float32", "bfloat16"]
    else:
        params = [{"elements": n} for n in ([1, 64, 4096, 131072, 1048576] if category == "activation" else [10000, 100000, 1000000])]
        dtypes = ["float32"]
    specs.append({"operator": op, "params": params, "dtypes": dtypes, "stages": ["optimizer"] if category == "optimizer" else ["forward", "backward"]})
standard = {"schema_version": 1, "name": "Standard · 28 operators", "description": "28 类算子，多形状/精度/阶段；运行前用 --dry-run 检查规模。", "operators": specs}
smoke = {"schema_version": 1, "name": "CPU smoke · 28 operators", "description": "所有算子的最小 CPU 冒烟测试；建议 PyTorch >= 2.4。", "operators": []}
for spec in specs:
    p = spec["params"][0].copy()
    if "elements" in p:
        p["elements"] = 64
    if "shape" in p:
        p["shape"] = [max(2, min(x, 8)) for x in p["shape"]]
    smoke["operators"].append({**spec, "params": [p], "dtypes": ["float32"]})
Path("templates").mkdir(exist_ok=True)
for name, value in [("standard", standard), ("smoke", smoke)]:
    Path(f"templates/{name}.json").write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")

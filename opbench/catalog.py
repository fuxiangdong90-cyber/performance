"""Declarative test catalog; no arbitrary code is accepted from templates."""
import hashlib
import itertools
import json
import math

CATEGORIES = {
    "matrix": ["mm", "matmul", "bmm", "addmm", "addbmm", "baddbmm"],
    "convolution": ["Conv1d", "Conv2d", "Conv3d", "Conv2dPointwise"],
    "transpose_convolution": ["ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d"],
    "normalization": ["BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "GroupNorm", "LayerNorm", "RMSNorm"],
    "activation": ["ReLU", "LeakyReLU", "GELU", "SiLU"],
    "optimizer": ["sgd", "adagrad", "adam", "adamw", "rmsprop"],
}
OPERATORS = {op: category for category, ops in CATEGORIES.items() for op in ops}
DTYPES = {"float32": 4, "float16": 2, "bfloat16": 2, "float64": 8}
STAGES = {"forward", "backward", "optimizer"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def case_key(case):
    # Display names, device and timings must never influence matching.
    identity = {k: case[k] for k in ("operator", "params", "dtype", "stage", "execution", "module_mode", "gradient_scope")}
    if case.get("implementation", "native") != "native":
        identity["implementation"] = case["implementation"]
    return hashlib.sha256(canonical(identity).encode()).hexdigest()[:24]


def validate_params(op, p):
    cat = OPERATORS[op]
    if not isinstance(p, dict):
        raise ValueError("params must be an object")
    required = {"matrix": ["m", "n", "k"], "convolution": ["batch", "cin", "cout", "spatial", "kernel", "stride", "padding", "groups"],
                "transpose_convolution": ["batch", "cin", "cout", "spatial", "kernel", "stride", "padding", "groups"],
                "activation": ["elements"], "optimizer": ["elements"], "normalization": ["shape"]}[cat]
    for key in required:
        if key not in p:
            raise ValueError(f"{op}: missing parameter {key}")
    allowed = set(required) | ({"batch"} if cat == "matrix" else {"groups"} if op == "GroupNorm" else set())
    if set(p) - allowed:
        raise ValueError(f"{op}: unknown parameters {sorted(set(p)-allowed)}")
    for key, value in p.items():
        if key in ("shape", "spatial") and (not isinstance(value, list) or not 1 <= len(value) <= 8):
            raise ValueError("shape/spatial must be a list with 1..8 dimensions")
        if key not in ("shape", "spatial") and type(value) is not int:
            raise ValueError(f"{key} must be an integer")
        values = value if isinstance(value, list) else [value]
        if not values or any(type(x) is not int or x < (0 if key == "padding" else 1) or x > 100000000 for x in values):
            raise ValueError(f"{op}: invalid parameter {key}")
    if cat in ("convolution", "transpose_convolution"):
        dims = 2 if op == "Conv2dPointwise" else int(op[-2])
        if len(p["spatial"]) != dims or p["cin"] % p["groups"] or p["cout"] % p["groups"]:
            raise ValueError("invalid convolution dimensions/groups")
        if op == "Conv2dPointwise" and p["kernel"] != 1:
            raise ValueError("pointwise convolution requires kernel=1")
        if cat == "convolution" and any((x + 2*p["padding"] - p["kernel"]) // p["stride"] + 1 < 1 for x in p["spatial"]):
            raise ValueError("convolution output is empty")
        if cat == "transpose_convolution" and any((x-1)*p["stride"]-2*p["padding"]+p["kernel"] < 1 for x in p["spatial"]):
            raise ValueError("transpose convolution output is empty")
    if cat == "normalization":
        shape = p["shape"]
        ranks = {"BatchNorm1d": (2, 3), "BatchNorm2d": (4,), "BatchNorm3d": (5,)}
        if op in ranks and len(shape) not in ranks[op]:
            raise ValueError("invalid BatchNorm rank")
        if op == "GroupNorm" and (len(shape) < 2 or shape[1] % p.get("groups", 1)):
            raise ValueError("invalid GroupNorm channels/groups")


def normalize_case(raw):
    if not isinstance(raw, dict):
        raise ValueError("result/case must be an object")
    op = raw.get("operator")
    if op not in OPERATORS:
        raise ValueError(f"unknown operator: {op}")
    case = {"operator": op, "category": OPERATORS[op], "params": raw.get("params", {}),
            "dtype": raw.get("dtype", "float32"), "stage": raw.get("stage", "forward"),
            "execution": raw.get("execution", "eager"), "module_mode": raw.get("module_mode", "eval"),
            "implementation": raw.get("implementation", "native"),
            "gradient_scope": raw.get("gradient_scope", "all" if raw.get("stage") == "backward" else "none")}
    if case["dtype"] not in DTYPES or case["stage"] not in STAGES or case["execution"] != "eager" or case["module_mode"] not in ("train", "eval"):
        raise ValueError("invalid dtype, stage, execution or module_mode")
    if (case["category"] == "optimizer") != (case["stage"] == "optimizer"):
        raise ValueError("optimizer operators require optimizer stage")
    if case["implementation"] not in ("native", "addmm_loop_v1", "bmm_fp32_sum_v1") or (case["implementation"] != "native" and op != "addbmm"):
        raise ValueError("unsupported operator implementation")
    if case["gradient_scope"] != ("all" if case["stage"] == "backward" else "none"):
        raise ValueError("gradient_scope must be all for backward, none otherwise")
    validate_params(op, case["params"])
    case["case_key"] = case_key(case)
    case["name"] = raw.get("name") or f"{op} · {canonical(case['params'])} · {case['dtype']} · {case['stage']}"
    if not isinstance(case["name"], str):
        raise ValueError("invalid case name")
    label = {"addmm_loop_v1": "addmm-loop", "bmm_fp32_sum_v1": "bmm+FP32-sum"}.get(case["implementation"])
    if label and label not in case["name"]:
        case["name"] += f" · {label} [composite]"
    if not isinstance(case["name"], str) or len(case["name"]) > 2000:
        raise ValueError("invalid case name")
    return case


def workload(case):
    """Algorithmic FLOPs/compulsory logical bytes. Unknown remains null."""
    p, op, stage = case["params"], case["operator"], case["stage"]
    size = DTYPES[case["dtype"]]
    f = q = None
    cat = OPERATORS[op]
    note = "Logical minimum traffic, not measured device memory traffic."
    if case.get("implementation", "native") != "native":
        note += " Explicit GPU composition; casts and intermediate reads/writes are excluded from logical traffic."
    if cat == "matrix":
        m, n, k = (p[x] for x in ("m", "n", "k"))
        b = p.get("batch", 1) if op in ("bmm", "matmul", "addbmm", "baddbmm") else 1
        f = 2*b*m*n*k
        out = m*n*(1 if op == "addbmm" else b)
        q = size*(b*m*k+b*k*n+out*(2 if op in ("addmm", "addbmm", "baddbmm") else 1))
        note += " FLOPs count matrix products only (FMA=2); excludes epilogue/reduction."
        if stage == "backward":
            f *= 2
    elif cat in ("convolution", "transpose_convolution"):
        dims = len(p["spatial"])
        spatial_out = [(x+2*p["padding"]-p["kernel"])//p["stride"]+1 for x in p["spatial"]]
        if cat == "transpose_convolution":
            spatial_out = [(x-1)*p["stride"]-2*p["padding"]+p["kernel"] for x in p["spatial"]]
        weight = p["cout"]*p["cin"]//p["groups"]*p["kernel"]**dims
        count = math.prod(spatial_out if cat == "convolution" else p["spatial"])
        f = 2*p["batch"]*count*weight
        q = size*(p["batch"]*p["cin"]*math.prod(p["spatial"])+weight+p["batch"]*p["cout"]*math.prod(spatial_out))
        if stage == "backward":
            f = None  # Do not pretend training FLOPs equal forward FLOPs.
        note += " Direct convolution equivalent FLOPs; backward FLOPs unmodeled."
    elif cat in ("activation", "normalization"):
        q = 2*size*(p["elements"] if "elements" in p else math.prod(p["shape"]))
        note += " Main input/output only; normalization statistics/affine parameters excluded."
    elif cat == "optimizer":
        q = size*p["elements"]*{"sgd": 3, "adagrad": 5, "adam": 7, "adamw": 7, "rmsprop": 5}[op]
        note += " Default optimizer without momentum/AMSGrad; fused intermediates excluded."
    if stage == "backward":
        q = None
        note += " Backward traffic unmodeled; bandwidth intentionally null."
    return {"flops": f, "bytes": q, "arithmetic_intensity": f/q if f is not None and q else None, "workload_note": note}


def expand_template(template):
    if not isinstance(template, dict) or template.get("schema_version") != 1 or not isinstance(template.get("operators"), list) or not template["operators"]:
        raise ValueError("template requires schema_version=1 and operators list")
    cases = []
    for spec in template["operators"]:
        if not isinstance(spec, dict) or not isinstance(spec.get("params"), list) or not spec["params"]:
            raise ValueError("operator template requires a nonempty params list")
        for key in ("dtypes", "stages"):
            if key in spec and (not isinstance(spec[key], list) or not spec[key]):
                raise ValueError(f"{key} must be a nonempty list")
        for params, dtype, stage in itertools.product(spec["params"], spec.get("dtypes", ["float32"]), spec.get("stages", ["forward"])):
            cases.append(normalize_case({"operator": spec["operator"], "params": params, "dtype": dtype, "stage": stage,
                                         "implementation": spec.get("implementation", "native"),
                                         "module_mode": spec.get("module_mode", "train" if stage != "forward" else "eval")}))
            if len(cases) > 20000:
                raise ValueError("template exceeds 20000 cases")
    keys = [c["case_key"] for c in cases]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate template configurations")
    return cases

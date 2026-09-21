# OpBench Report v1

编码 UTF-8。单文件不超过 32 MiB、1–20000 个结果；每个用例最多 10000 个 wall 样本。浮点数必须有限，禁止 NaN/Infinity。导入整份报告成功才提交数据库；同一报告中重复配置拒绝导入。

```json
{
  "schema_version": 1,
  "run": {
    "name": "CPU smoke",
    "device": {"name": "CPU", "backend": "cpu"},
    "timing_method": "synchronized_wall_per_iteration",
    "torch_version": "2.x",
    "backend_version": "unknown",
    "precision_policy": "highest; TF32 disabled where exposed",
    "warmup": 10,
    "iterations": 50,
    "cpu_threads": 1,
    "synthetic": false
  },
  "results": [{
    "operator": "mm",
    "params": {"m": 64, "n": 64, "k": 64},
    "dtype": "float32",
    "stage": "forward",
    "execution": "eager",
    "module_mode": "eval",
    "gradient_scope": "none",
    "status": "pass",
    "correctness": "finite_only",
    "wall_us": 20.0,
    "gpu_us": null,
    "cpu_us": null,
    "p95_us": 22.0,
    "stddev_us": 1.0,
    "peak_allocated_bytes": null,
    "peak_delta_bytes": null,
    "samples_us": [19, 20, 22]
  }]
}
```

以上数值仅用于说明格式，不是硬件结果。示例文件必须标 synthetic=true。

## 字段

| 字段 | 规则 |
|---|---|
| run.name | 必填，1–300 字符 |
| run.device | 必填对象，必须包含设备 name，建议包含 backend、platform |
| run.timing_method | v1 必须为 synchronized_wall_per_iteration，避免无标记地混合 Event / wall |
| run.precision_settings | 采集器记录实际精度开关值，包括厂商暴露的 muDNN TF32；历史报告可缺省，缺省不代表关闭 |
| run.input_initialization | 新采集器为 cpu_then_copy，输入生成和传输在计时外；历史报告可缺省 |
| operator | catalog.py 中的 28 个名称之一，大小写敏感 |
| params | 算子参数对象，格式见下表 |
| dtype | float32 / float16 / bfloat16 / float64 |
| stage | forward / backward / optimizer；优化器仅 optimizer |
| execution | 当前固定 eager |
| module_mode | train / eval；与 stage 分离 |
| gradient_scope | backward 为 all，其他为 none |
| status | pass / failed / unsupported / oom |
| correctness | not_checked / finite_only / cpu_reference / cpu_reference_failed / synthetic；不由 pass 推定正确性 |
| implementation | 默认 native；addbmm 可显式选择 bmm_fp32_sum_v1 或历史 addmm_loop_v1，与 native 使用不同 case_key |
| reference | cpu_reference 必须提供 device=cpu、implementation=native、rtol、atol、max_abs_error、max_scaled_error；cpu_reference 的归一化误差不超过 1；cpu_reference_failed 必须 status=failed 且归一化误差大于 1 |
| run.backend_arch_list / device_arch | 后端编译支持的架构与实测设备架构；MUSA 不匹配默认拒绝启动 |
| run.runtime_image | 运行镜像内容 ID（通过容器包装脚本采集） |
| run.operator_implementations | 非原生实现映射；为空表示全部默认 native |
| wall_us | pass 必须为正数，其他允许 null |
| gpu_us / cpu_us | 可空；CPU 采集器输出 null；存在的 GPU 时间必须为正 |
| samples_us | 正数数组，原始逐次同步 wall 延迟，单位 μs |
| peak_allocated_bytes | 可空，框架峰值 allocated 字节数 |
| peak_delta_bytes | 可空，相对测量前 allocated 的增量 |
| error | 失败说明，最多保留 4000 字符 |

FLOPs、bytes、arithmetic_intensity、TFLOPS、bandwidth 在服务端根据已知模型重新计算，不信任导入文件里提供的派生数值。case_key 由语义字段 SHA-256 前 24 位生成，名称不参与匹配。重复上传创建新的 run ID，便于保留重复实验。

## params

| 算子族 | 必填参数 | 说明 |
|---|---|---|
| 矩阵 | m,n,k；批次变体可提供 batch（默认 1） | matmul 当前模板仅使用 3D batched dense 矩阵 |
| 卷积/转置卷积 | batch,cin,cout,spatial,kernel,stride,padding,groups | spatial 为对应维数数组；dilation=1、bias=false、output_padding=0 |
| 激活/优化器 | elements | 一维张量元素数 |
| 归一化 | shape | BatchNorm 用第二维作为通道；LN/RMSNorm 用最后一维；GroupNorm 可给 groups |

模板结构为 `schema_version=1`、`operators=[{operator, params:[...], dtypes:[...], stages:[...]}]`。采集器对三个列表做笛卡尔积，先验证再执行。模板不接受脚本字符串或 eval 表达式。

## 历史数据

原页面中的 bwd1/bwd2/bwd3 不应猜测为单独输入梯度；原报告已声明矩阵历史用例实际计算全部梯度。迁移前必须取得原始 JSON 和采集代码，确认单位、时间范围、梯度范围及重复策略。当前不提供猜测性自动迁移。

## CPU 参考校验

`--verify-reference` 的逐元素判据为 `abs(actual-reference) <= atol + rtol*abs(reference)`。float64 的 (rtol,atol) 为 (1e-8,1e-8)，float32 为 (1e-3,1e-4)，float16 为 (0.01,0.01)，bfloat16 为 (0.05,0.05)。`max_scaled_error` 是左侧除以右侧后的最大值。NaN/Inf 或任一元素超限均失败；校验与 CPU/GPU 传输均在计时外。默认不启用，不给历史报告追认参考校验。

推荐参考方法为 `run.reference_method=native_cpu_float64_quantized_inputs_v1`：输入与初始参数按目标 dtype 量化后转 float64；保留目标 dtype 对 RMSNorm 默认 epsilon 的影响。reference 额外记录 `compute_dtype=float64`、`input_dtype`。旧报告没有这些字段时，参考计算使用原始 dtype；两者不能混为同一精度验证方法。

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
| operator | catalog.py 中的 28 个名称之一，大小写敏感 |
| params | 算子参数对象，格式见下表 |
| dtype | float32 / float16 / bfloat16 / float64 |
| stage | forward / backward / optimizer；优化器仅 optimizer |
| execution | 当前固定 eager |
| module_mode | train / eval；与 stage 分离 |
| gradient_scope | backward 为 all，其他为 none |
| status | pass / failed / unsupported / oom |
| correctness | 建议 not_checked / finite_only / synthetic；不由 pass 推定正确性 |
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

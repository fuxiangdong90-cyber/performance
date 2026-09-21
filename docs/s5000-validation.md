# S5000 全部 28 种算子测试结果

2026-09-21，单张 MTT S5000（GPU 0），PyTorch 2.5.0 + torch_musa 2.5.0+aed8b42（重编译 arch31）、SDK 4.3.8、驱动 3.3.8-server。

**28 种算子全部执行；两轮标准模板各 279 配置，均为 273 通过、6 数值超差、0 unsupported、0 OOM。冒烟模板 51/51 通过。不是 279 项全部通过。**

每配置预热 10 次、计时 50 次，CPU 线程 1，种子 2026，eager，TF32 关闭。前向、全部输入/参数梯度及优化器更新均覆盖。GPU 实测前，使用相同量化输入做 CPU float64 原生参考校验；超差配置不计性能成绩、不参与加速比。

27 种使用原生 MUSA 路径；addbmm 使用明确标记的 `bmm_fp32_sum_v1` GPU 组合实现（低精度提升到 FP32 累加），不能当作原生融合 addbmm 的性能。

## 覆盖清单

| 算子 | 配置 | 通过 | 数值超差 |
|---|---:|---:|---:|
| BatchNorm1d | 4 | 3 | 1 |
| BatchNorm2d | 4 | 3 | 1 |
| BatchNorm3d | 4 | 3 | 1 |
| Conv1d | 8 | 8 | 0 |
| Conv2d | 8 | 8 | 0 |
| Conv2dPointwise | 8 | 8 | 0 |
| Conv3d | 8 | 8 | 0 |
| ConvTranspose1d | 8 | 8 | 0 |
| ConvTranspose2d | 8 | 8 | 0 |
| ConvTranspose3d | 8 | 8 | 0 |
| GELU | 10 | 10 | 0 |
| GroupNorm | 4 | 4 | 0 |
| LayerNorm | 4 | 3 | 1 |
| LeakyReLU | 10 | 10 | 0 |
| RMSNorm | 4 | 3 | 1 |
| ReLU | 10 | 10 | 0 |
| SiLU | 10 | 10 | 0 |
| adagrad | 3 | 3 | 0 |
| adam | 3 | 3 | 0 |
| adamw | 3 | 3 | 0 |
| addbmm | 24 | 23 | 1 |
| addmm | 24 | 24 | 0 |
| baddbmm | 24 | 24 | 0 |
| bmm | 24 | 24 | 0 |
| matmul | 24 | 24 | 0 |
| mm | 24 | 24 | 0 |
| rmsprop | 3 | 3 | 0 |
| sgd | 3 | 3 | 0 |

## 未通过的配置

下表为第二轮原始参考误差；归一化误差是 `abs(actual-reference)/(atol+rtol*abs(reference))` 的最大值，超过 1 即失败。容差保持预设值，没有为得到通过而放宽。最大绝对误差与最大归一化误差可能出现在不同元素。

| 算子 / 阶段 | dtype | 参数 | 最大绝对误差 | 最大归一化误差 |
|---|---|---|---:|---:|
| addbmm / forward | float32 | `{"m":1024,"n":1024,"k":1024,"batch":4}` | 0.0002767118 | 1.13683 |
| BatchNorm1d / backward | bfloat16 | `{"shape":[4,16,64]}` | 0.25585938 | 5.11719 |
| BatchNorm2d / backward | bfloat16 | `{"shape":[4,16,16,16]}` | 0.8984375 | 17.9688 |
| BatchNorm3d / backward | bfloat16 | `{"shape":[2,8,8,8,8]}` | 0.72265625 | 14.4531 |
| LayerNorm / backward | bfloat16 | `{"shape":[8,256,64]}` | 0.30937571 | 1.79651 |
| RMSNorm / backward | bfloat16 | `{"shape":[8,256,64]}` | 0.48502185 | 1.64835 |

addbmm FP32 大矩阵结果略超当前阈值；BF16 归一化反向也超差。这里确认的是当前测试输入、软件版本和容差下的差异，不能据此推断所有形状/版本均错误。

早期同精度 CPU 参考存在 LayerNorm BF16 梯度缺陷，详见 [环境与参考诊断](musa-testing.md)。这些历史报告已保留；最终结论使用上面的 float64 参考方法。

## 复现

```bash
cd /opt/opbench-test
bash scripts/run_s5000_suite.sh smoke
bash scripts/run_s5000_suite.sh standard
```

测试机结果位于 `results/`，同一套脚本可再次执行并产生新时间戳报告。完整 JSON 保存到部署数据库，可在运行记录导出；Git 仓库仅提交代码、方法及验证汇总。

最终报告文件及 SHA-256：

- `s5000-arch31-smoke-20260921-054203.json`：`a81d0dd04b742a9a805fcd3eb76af21421a612f669ee68bce211de060887c26d`
- `s5000-arch31-standard-20260921-053947.json`：`a81ba795bcc7cf011d5aee220fde1c8c7e6f36a79eb5024f927027b492848b76`
- `s5000-arch31-standard-20260921-054104.json`：`53f13b31880eef144842c46e7703c1d6ced284c925c65542e9e50dc731de18eb`

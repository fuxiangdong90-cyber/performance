# MTT S5000 测试环境

测试环境由厂商 PyTorch 镜像、宿主机 GPU 驱动和宿主机 MUSA SDK 组成。后端数据库服务与测试容器分开运行。

已验证启动的组合：PyTorch 2.5.0、torch_musa 2.5.0+aed8b42、宿主机 SDK 4.3.8、驱动 3.3.8-server、MTT S5000（计算能力 3.1）。固定镜像：

```text
registry.mthreads.com/mcconline/musa-pytorch-release-public:rc4.3.0-v2.5.0-qy2
sha256:58e39c5bd27eaae4df479aa0c36db6fb2afcd4390f11e658db606a4d71248d9f
```

`scripts/run_musa_container.sh` 使用摘要固定镜像。驱动库以只读方式挂载，并优先加载宿主机 SDK 的库；镜像中仍保留依赖特定 SONAME 的旧库。因此这是明确记录的组合环境，不是未经改动的镜像性能结果。进程映射核对实际加载了宿主机 muDNN 3.1.8.0、muBLAS 1.10.8、muSART 4.3.8、MCCL 2.11.4，并保留镜像内 muSolver 1.4.0；框架的 muDNN 版本接口返回 3100，不能据此代替实际库文件版本。

## 运行

默认参数对应此次测试的物理 GPU 0：`/dev/mtgpu.0`、`/dev/dri/card1`、`/dev/dri/renderD128`。在其他机器上先核对设备节点与 PCI 地址映射；可通过 `OPBENCH_MUSA_DEVICE_NODE`、`OPBENCH_MUSA_DRM_CARD`、`OPBENCH_MUSA_DRM_RENDER`、`OPBENCH_MUSA_VISIBLE_DEVICES` 覆盖。宿主机 SDK 和驱动路径分别用 `OPBENCH_MUSA_SDK`、`OPBENCH_MUSA_DRIVER` 覆盖。

```bash
bash scripts/run_musa_container.sh -m opbench.runner \
  --template templates/smoke.json --device musa:0 --backend-module torch_musa \
  --backend-version 'torch_musa 2.5.0+aed8b42; host SDK 4.3.8; driver 3.3.8-server' \
  --warmup 10 --iterations 50 --name s5000-smoke \
  --output results/s5000-smoke.json

# 标准测试将模板替换为 templates/standard.json，并使用新的 name/output。
```

容器只暴露指定设备，关闭网络，测试结果通过源码目录挂载写回 `results/`。需要上传时，在容器外通过后端导入接口或网页导入 JSON；访问令牌无需传入测试容器。

## 本次发现的兼容问题

1. 镜像内 `libmusa.so`/`libmusa.so.4` 是占位文件，直接运行会报 `file too short`。脚本挂载实际驱动库，同时覆盖动态加载器别名和显式绝对路径。
2. 仅使用镜像旧 SDK 时，基础 Fill 操作报 `invalid device function`。使用宿主机 SDK 库优先路径后，基础矩阵和激活计算检查通过。
3. GPU `torch.randn` 仍报 `invalid device function`，导致最初 51 个配置全部在输入初始化阶段失败。采集器改为 CPU 生成随机输入后复制到目标设备，标记 `input_initialization=cpu_then_copy`。这些操作发生在计时外；没有用 CPU 执行来替代被测 GPU 算子。原始失败报告仍保留以供诊断。
4. MUSA 使用独立的 `torch.backends.mudnn.allow_tf32`；采集器显式关闭，并将读回值写入 `precision_settings`。

## 实机验证结果

2026-09-21，单张 S5000，CPU 线程数 1，固定种子 2026，eager，每配置预热 10 次、测量 50 次，TF32 关闭：

| 模板 | 配置数 | pass | unsupported | failed |
|---|---:|---:|---:|---:|
| smoke，CPU 初始化输入 | 51 | 45 | 2 | 4 |
| standard，第一次 | 279 | 242 | 24 | 13 |
| standard，第二次 | 279 | 242 | 24 | 13 |

成功配置均采集到 GPU 流时间、CPU 提交时间和框架 allocated 峰值内存。`unsupported` 全部来自当前扩展未注册 `addbmm`（前向/反向）；`failed` 来自 GroupNorm 反向、LeakyReLU 反向、Adam、AdamW 的 `invalid device function`。标准模板包含多种形状和精度，因此失败配置数更多。

两轮标准测试逐配置状态一致，成功的前向/反向/优化器配置分别为 120/113/9。同步 wall 延迟的第一次除以第二次几何均值分别约 0.998/0.982/1.069；这是同机重复测量波动，不是优化收益或跨硬件结论。

初始 GPU 随机数初始化方案的 51 个失败结果，以及 CPU 初始化后的结果分别保存；不覆盖失败历史。原始 JSON 与日志保存在测试机，本机也留有副本；JSON 已导入部署数据库。仓库只包含环境、方法及验证汇总，不包含 SSH 凭据或服务器访问令牌。

结果中的 `pass` 表示该配置完成计时且输出/梯度通过有限值检查；不是独立参考实现的数值正确性认证。不能将这个组合的支持范围直接推广到所有 S5000 软件版本。

厂商来源：[torch_musa](https://github.com/MooreThreads/torch_musa)、[v2.5.0 muDNN 开关实现](https://github.com/MooreThreads/torch_musa/blob/v2.5.0/torch_musa/core/mudnn.py)。

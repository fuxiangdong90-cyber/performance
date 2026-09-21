# MTT S5000 测试环境

测试环境由厂商 PyTorch 镜像、宿主机 GPU 驱动和宿主机 MUSA SDK 组成。后端数据库服务与测试容器分开运行。

已验证启动的组合：PyTorch 2.5.0、torch_musa 2.5.0+aed8b42、宿主机 SDK 4.3.8、驱动 3.3.8-server、MTT S5000（计算能力 3.1）。固定镜像：

```text
registry.mthreads.com/mcconline/musa-pytorch-release-public:rc4.3.0-v2.5.0-qy2
sha256:58e39c5bd27eaae4df479aa0c36db6fb2afcd4390f11e658db606a4d71248d9f
```

`scripts/run_musa_container.sh` 使用摘要固定镜像。驱动库以只读方式挂载，并优先加载宿主机 SDK 的库；镜像中仍保留依赖特定 SONAME 的旧库。因此这是明确记录的组合环境，不是未经改动的镜像性能结果。进程映射核对实际加载了宿主机 muDNN 3.1.8.0、muBLAS 1.10.8、muSART 4.3.8、MCCL 2.11.4，并保留镜像内 muSolver 1.4.0；框架的 muDNN 版本接口返回 3100，不能据此代替实际库文件版本。

## 推荐：S5000 架构匹配环境

原厂镜像扩展的 `torch.musa.get_arch_list()` 实际为 `["22"]`，本机为 `(3,1)`。仅替换动态库不能修复这些编译内核。现已用同一厂商源码提交 `aed8b424`、厂商补丁 PyTorch 2.5.0 和宿主机 SDK 4.3.8，以 `TORCH_MUSA_ARCH_LIST=31` 重编译，运行时读回 `["31"]`。

已构建本地运行镜像 `opbench/musa:s5000-arch31`；仅修改隔离容器内扩展，未更改宿主机驱动或 SDK。采集器默认拒绝 MUSA 架构不匹配，`--allow-arch-mismatch` 仅用于诊断历史环境。

```bash
cd /opt/opbench-test
bash scripts/run_s5000_suite.sh smoke
bash scripts/run_s5000_suite.sh standard
```

默认 GPU 0，每配置预热 10 次、计时 50 次，打开 CPU 参考校验，结果按时间戳保存。脚本退出 2 表示仍有失败，必须检查 JSON，不代表未执行后续配置。

addbmm 未在此扩展注册，套件显式选择 `bmm_fp32_sum_v1`：在 GPU 上执行 bmm、求和、加偏置，低精度输入提升到 FP32 累加后转回原 dtype，前向和全部输入梯度均测试。其余 27 种使用原生 PyTorch MUSA 路径。组合实现单独标记并参与 case_key，不与原生 addbmm 配对。

## 旧镜像诊断命令

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

## 历史环境结果（arch22 扩展，保留诊断记录）

2026-09-21，单张 S5000，CPU 线程数 1，固定种子 2026，eager，每配置预热 10 次、测量 50 次，TF32 关闭：

| 模板 | 配置数 | pass | unsupported | failed |
|---|---:|---:|---:|---:|
| smoke，CPU 初始化输入 | 51 | 45 | 2 | 4 |
| standard，第一次 | 279 | 242 | 24 | 13 |
| standard，第二次 | 279 | 242 | 24 | 13 |

成功配置均采集到 GPU 流时间、CPU 提交时间和框架 allocated 峰值内存。`unsupported` 全部来自当前扩展未注册 `addbmm`（前向/反向）；`failed` 来自 GroupNorm 反向、LeakyReLU 反向、Adam、AdamW 的 `invalid device function`。标准模板包含多种形状和精度，因此失败配置数更多。

两轮标准测试逐配置状态一致，成功的前向/反向/优化器配置分别为 120/113/9。同步 wall 延迟的第一次除以第二次几何均值分别约 0.998/0.982/1.069；这是同机重复测量波动，不是优化收益或跨硬件结论。

初始 GPU 随机数初始化方案的 51 个失败结果，以及 CPU 初始化后的结果分别保存；不覆盖失败历史。原始 JSON 与日志保存在测试机，本机也留有副本；JSON 已导入部署数据库。仓库只包含环境、方法及验证汇总，不包含 SSH 凭据或服务器访问令牌。

上述历史结果中的 `pass` 只表示完成计时和有限值检查；新架构环境推荐套件另外执行 CPU 参考校验。不能将这个组合的支持范围直接推广到所有 S5000 软件版本。

厂商来源：[torch_musa](https://github.com/MooreThreads/torch_musa)、[v2.5.0 muDNN 开关实现](https://github.com/MooreThreads/torch_musa/blob/v2.5.0/torch_musa/core/mudnn.py)。

## 重建架构匹配的扩展

`bash scripts/build_musa_arch31.sh` 使用 `vendor-source/home/torch_musa` 和 `vendor-source/home/pytorch`，构建 wheel 后用 `Dockerfile.musa` 安装到原厂固定镜像；不把编译目录装进运行容器。源码应与镜像内 PyTorch 匹配，不能用缺少厂商补丁的上游 PyTorch 替换。

本次源码来自同一固定镜像的不可变层；使用 Docker containerd 存储的服务器上，层文件位于 `/var/lib/containerd/io.containerd.content.v1.content/blobs/sha256/`。将以下层中指定前缀按顺序解压到空的 `vendor-source/` 目录（目录不提交 Git）：

| 层 SHA-256 | 解压前缀 | 用途 |
|---|---|---|
| `2d3f907617b78976b1aac5406aa020aa8103db6dfab0ecc1ef8e4957163cc092` | `home/torch_musa` | 扩展源码 |
| `f66656038a6708e32dbc98847563d8373450bc4781ee8b9b94829763c274f5e0` | `home/pytorch` | 框架源码 |
| `67b51bbb6132fc7447ecf7c326869ee63acc2193f642038d2bc23a21dfa4b57d` | `home/pytorch` | 必须覆盖的厂商补丁 |

例如 `tar -xzf "$LAYER_STORE/$DIGEST" -C vendor-source home/pytorch`。仅适用于此固定镜像，先核验层 SHA-256；不同 Docker 存储后端应从镜像归档获取对应层。

构建使用宿主机 SDK 头文件优先、`CPLUS_INCLUDE_PATH=/usr/local/musa/include` 补充旧镜像内 Thrust。不要改用 CPATH：它会使旧 MCCL 头文件覆盖新 SDK，导致 FP8 枚举编译失败。若已有其他配置的生成代码/缓存，先在相同编译容器内执行 `cmake --build build --target clean` 再构建。

本次 wheel SHA-256：`efdb32b712e120b34e2b2eb08b1a9fc6c62ae1df502389de4226b4285fdf70d6`。运行镜像 ID：`sha256:5c32fb4ce69bdbb9b6ccb3920597d348c74a399d9fe763e48ed8193693839651`。不同构建时间产生的包/镜像摘要可能不同，应记录自己的摘要与实际架构。

## 参考实现的诊断

重编译后，所有 28 种算子的 smoke 51 配置均通过同精度 CPU 参考；两次 standard 均为 271 pass / 8 数值超差（addbmm 使用当时的 addmm_loop_v1）。这些是诊断历史，不能直接视为 8 个 GPU 错误。

进一步用解析梯度发现：此镜像内 CPU PyTorch 2.5.0 的 BF16 LayerNorm，shape=[8,256,64]、上游梯度全 1 时，bias 梯度为 256；解析结果及 S5000 均为 2048。推荐套件因此改用量化输入后的 CPU float64 参考，保持原有容差不变。addbmm 改用 FP32 累加组合实现，避免逐批低精度舍入；两种实现保留独立标识，历史结果不覆盖。

## 最终高精度参考测试

两轮 standard 均为 273/279 通过、6 数值超差；最新 smoke 为 51/51 通过，覆盖全部 28 种算子。失败清单、误差和报告摘要见 [最终验证报告](s5000-validation.md)。完成测试不代表全部配置通过。

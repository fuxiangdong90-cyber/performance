# 逆向依据、实现范围与测试口径

## 观察依据

对用户指定的内网页面进行只读交互，观察到 OpBench HUD：深色导航栏、左侧筛选、左右运行选择、算子筛选带、22 项明细列、几何平均加速比、5% 胜负阈值、失败状态及分页。初次分析时页面列出 11 个可选运行，默认对比包含 2651 配置、1799 个参与汇总用例。

原系统声明历史矩阵 bwdN 实际计算全部输入梯度、仅 bwdall 纳入汇总；反向访存不估算，但部分历史行仍显示带宽。复现系统通过新的明确阶段协议消除这种歧义。

没有获得原站点后端、数据库、源码或全部 shape 配置。本实现是根据可见交互独立重建，不声称原字节级源码还原，不将原内网硬件测试结果发布到 GitHub。

## 覆盖与差异

| 项目 | 实现 |
|---|---|
| 前端 | 保持主要布局和指标列；列过滤用内嵌输入，支持数值比较 |
| 持久化 | 原页面提示本地数据模式；新系统改为服务器 SQLite，页面明确告知上传保存位置 |
| 模式 | 原 inference/training 拆为 stage 与 module_mode |
| 计时 | 所有新结果要求同步 wall 方法，另外记录事件和提交时间 |
| 模板 | 同样覆盖 28 种算子；标准模板 279 配置，smoke 51 配置；未猜测原 2651 配置的完整参数集 |
| 数据 | 原始实测记录不随源码发布；演示数据可复现且醒目标注 |
| 历史 JSON | 未获得 schema；只支持本文档定义的 v1，不臆造字段映射 |
| 语言 | 对比表头中英文；其余以中文为主 |
| 后台页面 | 实现运行记录、归档恢复、设备、模板下载/校验、文档 |

## 性能分类

语义分类为 matrix / convolution / transpose_convolution / normalization / activation / optimizer。不要仅凭算子名称自动标 compute-bound 或 memory-bound：具体瓶颈受 shape、精度、布局、启动延迟、缓存和后端算法影响。

I=F/Q；如果有可比精度的峰值算力 P 与带宽 B，可使用 Roofline 临界值 I*=P/B。这个关系是分析模型，不是实测瓶颈证明。本系统不在没有峰值/硬件计数器的情况下显示伪造利用率或瓶颈标签。

## 测量边界

1. 随机种子、CPU 线程数、预热和重复次数写入结果。
   新采集报告的 `precision_settings` 记录实际读取的开关；除 CUDA/cuDNN 外也关闭厂商暴露的 MUSA/muDNN TF32 开关。没有暴露的开关不假定其状态，不同运行的精度设置会在对比页提示。
   随机输入在 CPU 上生成后复制到目标设备，报告标记 `input_initialization=cpu_then_copy`；初始化与传输均在计时外。被测前向、反向、优化器计算仍在目标设备执行。这避免随机初始化内核的可用性影响其他算子的覆盖，也便于同版本、同种子复现输入。
2. CPU 在调用前/后读单调高分辨率时钟；GPU 每轮开始前和末尾同步。
3. wall 包含 operation 与终止事件提交/同步，GPU events 包含对应流区间，CPU enqueue 只覆盖 operation 调用。
4. 使用每轮测量的中位数、最近秩 P95、总体标准差；不把其中一种统计量伪称另一种。
5. backward 计算图在计时外构造，重复 autograd.grad，不积累 .grad；retain_graph 保持输入一致。此模式不同于完整训练 step。
6. optimizer 默认非 foreach，预先初始化状态；每轮恢复状态后再测量。不是不同参数状态下连续训练的吞吐。
7. 默认 finite_only 只验证没有 NaN/Inf。`--verify-reference` 在计时外用相同 CPU 随机状态重建输入，先按目标 dtype 量化输入和初始参数，再转 float64 进行原生 CPU 参考计算，比较前向输出、全部梯度或优化器参数；计时执行再次恢复随机状态。容差和最大绝对/归一化误差写入结果，任何元素超限都记 failed。反向参考使用全 1 上游梯度，与计时定义一致；这不是覆盖所有输入分布的数学证明。
   addbmm 推荐使用显式 `bmm_fp32_sum_v1`：低精度输入在 GPU 上转 FP32，bmm 后求和并加偏置，最后转回输出 dtype。原有 `addmm_loop_v1` 逐批 addmm 实现保留用于解释历史诊断数据。它是组合实现，非原生融合内核；计时包含全部调用，逻辑流量不含类型转换及中间读写，不同实现不配对。
   RMSNorm 参考显式保留目标 dtype 的默认 epsilon。参考计算不使用 BF16 CPU 累加，以避开已复现的该版本 CPU LayerNorm 低精度梯度问题；参考方法写入 run.reference_method。
8. 内存指标为框架 allocated 峰值，包含常驻输入/模块/图等；增量另外报告。共享设备上的其他任务可能干扰结果。

## 参考

- [PyTorch autograd.grad](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html)
- [PyTorch 异步执行语义](https://docs.pytorch.org/docs/stable/notes/cuda.html)
- [PyTorch baddbmm](https://docs.pytorch.org/docs/stable/generated/torch.baddbmm.html)

未实现原系统不可见的内部行为。厂商插件/驱动组合需在对应设备验证；本地 CPU 测试通过不代表所有加速器正确或性能一致。

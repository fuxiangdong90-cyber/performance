# OpBench Performance

基于指定 OpBench HUD 页面可见行为独立实现的算子性能平台：**前端看板 + Python HTTP API + SQLite 数据库 + PyTorch 测试模板/采集脚本**。

复现深色顶栏、左侧筛选、双运行选择、几何平均加速比、胜负/失败统计、算子选择带、22 列横向对比表；补齐运行记录、设备、测试模板、文档和结果持久化。无前端构建步骤，无 CDN，服务端仅依赖 Python 标准库。

> 本仓库不包含原系统源代码、原数据库或内网实测数据。`--demo` 生成的记录明确标为 SYNTHETIC。原站点未提供原始 JSON schema，因此不能宣称原始结果文件直接兼容；新格式有显式字段、单位和校验规则。

## 1. 启动看板

需要 Python 3.10+，推荐 3.12。以下命令在仓库根目录执行：

```bash
git clone https://github.com/fuxiangdong90-cyber/performance.git
cd performance
python -m opbench.server --demo
```

打开 **http://127.0.0.1:30000/**。`--demo` 仅在没有活动运行时导入两组确定性合成数据（每组 279 配置），不代表任何实际设备性能。

```bash
# 空库启动 / 自定义端口和数据库
python -m opbench.server --port 30001 --db data/production.sqlite3
```

无需安装 PyTorch 即可查看、导入、比较结果。Windows 若没有 `python` 命令，可用已安装 Python 的完整路径或 `py -3`。

## 2. 实际采集

测试在目标机器上执行；服务端不执行用户上传的代码。先安装设备厂商支持的 PyTorch 版本。CPU 环境可使用：

```bash
python -m venv .venv
# Linux / macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# 无需 PyTorch 即可展开和校验模板
python -m opbench.runner --template templates/standard.json --dry-run

# 全部 28 种算子的最小实测：51 个前向/反向/更新配置
python -m opbench.runner --template templates/smoke.json --device cpu --warmup 3 --iterations 10 --name cpu-smoke --output results/cpu.json

# 目标设备已安装兼容的 PyTorch 后运行
python -m opbench.runner --template templates/standard.json --device cuda --warmup 10 --iterations 50 --name accelerator-run --output results/gpu.json

# 导入：页面选择 JSON，或采集后直接上传
python -m opbench.runner --device cpu --output results/cpu.json --upload http://127.0.0.1:30000
```

厂商扩展通过 `--backend-module` 显式加载，设备通过 `--device` 选择，例如已安装对应扩展的环境使用 `--backend-module torch_mlu --device mlu`。该通用接口要求扩展提供 PyTorch 对应的 device / synchronize / Event API；**未在本地验证厂商加速器，不能保证所有版本可用**。不支持的算子会记录状态和错误，继续执行后续用例。退出码：0 全部成功，2 存在 failed/unsupported/oom。

## 3. 已实现的功能

| 模块 | 功能 |
|---|---|
| 看板 | 两次运行对比/交换、同配置配对、分阶段几何平均、5% 胜负阈值、失败明细 |
| 筛选 | 名称、六大算子族、精度、阶段、状态、算式过滤、逐列文本/数值过滤 |
| 明细 | 22 列、排序、分页、原始样本和运行结果详情、CSV 导出 |
| 导入 | 多 JSON 文件/目录、32 MiB 限制、整份报告事务导入、错误提示 |
| 运行记录 | 元数据、JSON 导出、软归档与恢复 |
| 设备 | 从报告元数据建立设备目录 |
| 模板 | 下载、服务端校验和用例展开、28 种算子 |
| 数据库 | SQLite WAL，设备/运行/用例/测量独立表，索引与外键 |
| 采集 | 预热、逐次同步、wall/stream/enqueue、median/P95/stddev、显存峰值、原始样本、失败隔离 |
| 访问控制 | 默认本机访问；远程绑定必须配置 Bearer token；不允许跨域写入 |

表头提供中英文切换；说明文档和主要操作界面以中文为主。当前执行模式为 Eager；未实现 torch.compile、分布式通信算子、远程任务队列和多用户权限管理。

## 4. 指标口径

- **主延迟**：每次同步执行的 wall time 中位数，包含调度与末尾同步，不是纯 kernel 时间。
- **GPU Stream**：设备事件区间；可能包含 stream 空闲。后端无事件接口时为空。
- **CPU 提交**：operation 调用时间，不含末尾同步；CPU 测试该项为空。
- **FLOPs**：算法估算，FMA=2；矩阵类计算主体乘法，附加归约/epilogue 不计。
- **流量**：逻辑最低读写估算，不等于实测 HBM 字节数。
- **TFLOPS** = FLOPs / wall_us / 10^6；**GB/s** = bytes / wall_us / 1000。
- **反向**：图在计时外准备，计时区间仅包含全部输入/参数梯度。矩阵主体 FLOPs 为前向两倍；反向流量、卷积反向 FLOPs 暂不建模，显示空值。
- **优化器**：参数和状态初始化在计时外，每轮恢复到相同状态；不将更新操作标成 inference。
- **显存**：框架 `max_memory_allocated`；另存相对基线增量。不是 reserved 或进程总显存；CPU 为 null。
- **正确性**：当前为输出/梯度有限性检测 `finite_only`，不等于已通过独立参考精度对比。

同配置匹配包含算子、全部参数、dtype、stage、execution、module_mode 和 gradient_scope。仅双方 pass 用例参与几何平均；失败/缺失不填零，不按无穷加速比汇总。不同环境的差异在页面提示。详细说明见 [指标与逆向说明](docs/methodology.md)。

## 5. 数据格式与 API

- [JSON 结果格式](docs/report-format.md)
- [接口说明与数据库设计](docs/api.md)
- [声明式测试模板](templates/standard.json)
- [最小 CPU 模板](templates/smoke.json)
- [可导入示例](examples/minimal-report.json)

## 6. 部署

单机默认启动即可。内网共享时使用 HTTPS 反向代理并设置 token，前端右上角“访问令牌”填写相同值：

```bash
export OPBENCH_API_TOKEN='replace-with-a-long-random-secret'
python -m opbench.server --host 0.0.0.0 --port 30000
```

PowerShell 对应设置方式为 `$env:OPBENCH_API_TOKEN='...'`。CLI 上传从同名环境变量读取 token。该服务采用标准库 HTTP server，面向可信内部团队；对公网或高并发生产环境，应增加成熟的应用网关、限流、TLS、进程管理和身份体系。

也可使用 Docker Compose：复制 `.env.example` 为 `.env`，填写随机 token 后运行 `docker compose up --build -d`。默认宿主机只发布到 127.0.0.1，数据库挂载到持久卷。

在线备份：

```bash
python -m opbench.backup --db data/opbench.sqlite3 --output backups/opbench.sqlite3
```

不要只复制正在写入的 WAL 数据库主文件。恢复时先停服务，再使用完整备份替换目标数据库。

## 7. 测试

```bash
python -m unittest discover -s tests -v
node --test tests/metrics.test.js
node --check web/app.js
```

未安装 PyTorch 时，仅跳过三个真实算子测试；安装后验证 28 种算子的 51 个配置、梯度不累积和优化器重置。GitHub Actions 包含 Linux/Windows 服务测试、Node 指标测试和 Linux CPU 实测。

## 目录

```text
opbench/       catalog、指标模型、SQLite、HTTP 服务、采集、演示、备份
web/           无构建依赖前端
templates/     28 算子标准模板、CPU 冒烟模板
scripts/       模板生成工具
tests/         API/数据库/算子/前端统计测试
docs/          数据协议、API、逆向依据、指标定义
examples/      最小可导入 JSON
```

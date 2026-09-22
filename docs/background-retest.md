# 服务器后台复测

后台执行分为两部分：服务器可独立完成固定脚本的复测和规则化比对；涉及根因判断、修改测试口径和跨硬件适配的工作仍需后续审查。不得把后台进程启动等同于全部任务完成。

## 已启动任务

- 基线：49122，BW1100，原 OpBench HUD commit `c8bbc83e6af6277bcd8f5d72f64022950fea3e53`。
- systemd 单元：`opbench-retest-20260922-v2.service`。
- 控制目录：`/root/opbench-retest-20260922-v2/`。
- 结果目录：`/root/opbench-formal-c8bbc83/results/opbench-retest-20260922-v2-r1` 和 `-r2`。
- 原正式测试源码、wheel、镜像和测试结果均保留；新任务使用新的 work/result/state 目录。
- 原始 full-run 保留八卡调度、每设备两枚 CPU、单线程及 7200 秒任务超时。历史 scaled_mm/scaled_grouped_mm 不支持和 Conv3d 超时可能复现，必须查看退出码与缺失覆盖，不能只读成功行。

```bash
systemctl status opbench-retest-20260922-v2
cat /root/opbench-retest-20260922-v2/status.json
tail -n 40 /root/opbench-retest-20260922-v2/opbench-retest-20260922-v2-r1.log
```

两轮完成后生成 historical-comparison-r1、historical-comparison-r2 和 repeat-comparison，每个目录包含 comparison.json/csv/md。比对键包含原模型名称、输入配置、dtype、阶段、compile 状态和指标名称；不同指标/阶段分别汇总。重复键、缺失记录不填零、不参与加速比。20% 是排查阈值，不是显著性或正确性证明。

## 当前未完成条件

49112、49154 使用独立临时 PEM 副本以 root 登录仍被服务器拒绝公钥认证。49122 到目标内网数据库的 HTTP 连接不通。后台任务不会绕过认证，不会把原 HUD JSON 冒充本项目 v1（两者的计时策略和数据结构不同），也不会因性能差异自动放宽精度阈值。

复测完成后需审查覆盖、误差、环境与 NUMA/频率记录，必要时对异常项复测并修订脚本，再固定版本并部署到另外两台机器。原始数据入库需完成显式格式适配与服务器到数据库的数据传输。

用户后续将当前范围收敛为仅完善 49122，其他机器暂缓。首次后台任务因工作目录导致 smoke 相对路径校验失败，已保留记录；v2 显式指定容器工作目录 /workspace/opbench-hud，保留原来源校验。

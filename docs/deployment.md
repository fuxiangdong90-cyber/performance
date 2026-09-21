# 后端与数据库部署

后端只依赖 Python 3.10+ 标准库。测试机单独安装厂商 PyTorch，生成 JSON 后上传；后端无需 GPU 或 PyTorch。

## Ubuntu + systemd + Nginx

在装有 Python 3.10+、Nginx 和 systemd 的 Ubuntu 服务器上，将可信源码解压到临时目录：

```bash
# 地址替换为服务器实际内网 IPv4；默认仅监听 127.0.0.1。
sudo OPBENCH_LISTEN_ADDRESS=192.168.1.10 bash scripts/deploy_backend.sh
```

该脚本安装到 `/opt/opbench`，以专用 `opbench` 用户运行，创建独立 Nginx 站点（默认端口 30000，`OPBENCH_PUBLIC_PORT` 可覆盖），代理本机 30002 端口。不会修改默认 80 端口站点。重新运行会更新本应用代码、systemd 单元和专用站点并重启服务，保留数据库及现有令牌。安装前应确认上述目录、服务名及端口没有用于其他应用。

| 内容 | 位置 |
|---|---|
| SQLite 数据库及 WAL | `/var/lib/opbench/opbench.sqlite3` |
| API 令牌 | `/etc/opbench/service.env`，仅 root 可读 |
| systemd 单元 | `/etc/systemd/system/opbench.service` |
| Nginx 站点 | `/etc/nginx/sites-available/opbench` |
| 数据库备份目录 | `/var/lib/opbench/backups` |

首次安装会生成随机令牌。在前端“API 访问令牌”输入它；令牌仅存于当前标签页 sessionStorage。CLI 上传通过 `OPBENCH_API_TOKEN` 环境变量授权。不要把 `service.env`、密钥、密码或含令牌的命令提交到 Git。

脚本配置的是 HTTP 内网入口；跨不可信网络访问时，应在现有网关配置 HTTPS，或通过 SSH 隧道访问，避免令牌明文传输。

## 检查与维护

```bash
systemctl status opbench
journalctl -u opbench -n 50 --no-pager
curl http://127.0.0.1:30002/api/health
nginx -t

cd /opt/opbench
runuser -u opbench -- python3 -m opbench.backup \
  --db /var/lib/opbench/opbench.sqlite3 \
  --output /var/lib/opbench/backups/backup-YYYYMMDD.sqlite3
```

备份命令不覆盖同名文件。服务设置为开机启动；代码只读，只有数据目录允许服务写入。升级前先做在线备份。需要恢复时停服务，确认已保存现有数据库及 WAL/SHM，再恢复完整备份并将属主设为 `opbench:opbench`，最后启动并验证运行记录。

## 分离测试机

在测试机的可信厂商运行环境内执行：

```bash
python -m opbench.runner --template templates/smoke.json \
  --device musa:0 --backend-module torch_musa \
  --backend-version YOUR_VALIDATED_VERSION \
  --warmup 10 --iterations 50 --output results/musa-smoke.json
```

测试机可以添加 `--upload http://BACKEND:30000` 直接上传；两台主机不可互通时，也可以将 JSON 下载后通过前端导入。切勿将用户的 SSH 私钥复制到测试机。所有失败和不支持的配置会保留在结果中，CLI 返回退出码 2；不要将这类结果宣称为全通过。

#!/usr/bin/env bash
# Run on the backend host, from an extracted trusted source release.
set -euo pipefail
SOURCE=$(cd "$(dirname "$0")/.." && pwd)
APP=/opt/opbench
STATE=/var/lib/opbench
CONF=/etc/opbench
PUBLIC_ADDRESS=${OPBENCH_LISTEN_ADDRESS:-127.0.0.1}
PUBLIC_PORT=${OPBENCH_PUBLIC_PORT:-30000}
if [[ $(id -u) != 0 ]]; then echo 'Run as root'; exit 1; fi
command -v python3 >/dev/null
command -v nginx >/dev/null
python3 -c 'import sys; assert sys.version_info >= (3,10)'
[[ "$PUBLIC_ADDRESS" =~ ^[0-9.]+$ ]] || { echo 'An IPv4 listen address is required'; exit 1; }
[[ "$PUBLIC_PORT" =~ ^[0-9]+$ ]] || exit 1
if ! id opbench >/dev/null 2>&1; then useradd --system --home-dir "$STATE" --shell /usr/sbin/nologin opbench; fi
install -d -m 755 "$APP" "$CONF"
install -d -m 750 -o opbench -g opbench "$STATE" "$STATE/backups"
cp -a "$SOURCE/opbench" "$SOURCE/web" "$SOURCE/templates" "$APP/"
cp "$SOURCE/README.md" "$APP/README.md"
if [[ ! -f "$CONF/service.env" ]]; then
  (umask 077; python3 -c 'import secrets; print("OPBENCH_API_TOKEN="+secrets.token_urlsafe(36))' > "$CONF/service.env")
fi
cat > /etc/systemd/system/opbench.service <<'UNIT'
[Unit]
Description=OpBench performance dashboard and database API
After=network.target
[Service]
Type=simple
User=opbench
Group=opbench
WorkingDirectory=/opt/opbench
EnvironmentFile=/etc/opbench/service.env
Environment=PYTHONDONTWRITEBYTECODE=1
ExecStart=/usr/bin/python3 -m opbench.server --host 127.0.0.1 --port 30002 --db /var/lib/opbench/opbench.sqlite3
Restart=on-failure
RestartSec=3
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/opbench
UMask=0027
[Install]
WantedBy=multi-user.target
UNIT
cat > /etc/nginx/sites-available/opbench <<NGINX
server {
    listen ${PUBLIC_ADDRESS}:${PUBLIC_PORT};
    server_name _;
    client_max_body_size 32m;
    location / {
        proxy_pass http://127.0.0.1:30002;
        proxy_set_header Host \$http_host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 120s;
    }
}
NGINX
ln -sfn /etc/nginx/sites-available/opbench /etc/nginx/sites-enabled/opbench
nginx -t
systemctl daemon-reload
systemctl enable --now opbench
systemctl restart opbench
systemctl reload nginx
python3 - <<'PY'
import time,urllib.request,json
for attempt in range(20):
    try:
        print(json.load(urllib.request.urlopen('http://127.0.0.1:30002/api/health',timeout=2)))
        break
    except OSError:
        time.sleep(.25)
else:
    raise SystemExit('Service failed health check; inspect journalctl -u opbench')
PY
echo "Dashboard: http://${PUBLIC_ADDRESS}:${PUBLIC_PORT}/"
echo "Database: ${STATE}/opbench.sqlite3"
echo "API token is stored in ${CONF}/service.env (root only)."

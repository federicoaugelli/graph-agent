#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="${GRAPH_AGENT_DIR:-/opt/graph-agent}"
SERVICE_USER="${GRAPH_AGENT_USER:-graph-agent}"
ENV_DIR="/etc/graph-agent"
ENV_FILE="$ENV_DIR/env"
UNIT_DST="/etc/systemd/system/graph-agent.service"

if [[ $EUID -ne 0 ]]; then
    echo "install.sh must run as root" >&2
    exit 1
fi

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> system packages"
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl git bubblewrap rsync

echo "==> service user $SERVICE_USER"
if ! id -u "$SERVICE_USER" >/dev/null 2>&1; then
    useradd --system --home-dir "$INSTALL_DIR" --shell /usr/sbin/nologin "$SERVICE_USER"
fi

echo "==> uv"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
fi
UV_BIN="$(command -v uv)"

echo "==> sources -> $INSTALL_DIR"
install -d -o "$SERVICE_USER" -g "$SERVICE_USER" "$INSTALL_DIR"
if [[ "$SRC_DIR" != "$INSTALL_DIR" ]]; then
    rsync -a --delete \
        --exclude '.git' --exclude '.venv' --exclude 'data' \
        --exclude '.env' --exclude 'config.local.yaml' \
        --exclude '__pycache__' --exclude '.mypy_cache' \
        --exclude '.ruff_cache' --exclude '.pytest_cache' \
        "$SRC_DIR/" "$INSTALL_DIR/"
fi
chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"

echo "==> virtualenv"
runuser -u "$SERVICE_USER" -- env HOME="$INSTALL_DIR" "$UV_BIN" sync \
    --frozen --no-dev --directory "$INSTALL_DIR"

echo "==> environment file"
install -d -m 0755 -o root -g root "$ENV_DIR"
if [[ ! -f "$ENV_FILE" ]]; then
    install -m 0600 -o root -g root "$SRC_DIR/deploy/graph-agent.env.example" "$ENV_FILE"
    echo "    created $ENV_FILE, fill in the secrets"
fi

echo "==> systemd unit"
sed -e "s|@GRAPH_AGENT_DIR@|$INSTALL_DIR|g" \
    -e "s|@GRAPH_AGENT_USER@|$SERVICE_USER|g" \
    "$SRC_DIR/deploy/graph-agent.service" > "$UNIT_DST"
chmod 0644 "$UNIT_DST"

systemctl daemon-reload
systemctl enable graph-agent.service
echo "==> done. edit $ENV_FILE, then: systemctl start graph-agent"
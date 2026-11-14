#!/usr/bin/env bash
# Optional: run the station as a systemd *user* service that starts at login and restarts on failure.
# The loop still waits for START on the control page, so the arm never moves on boot by itself.
#   scripts/install_service.sh            install + start
#   scripts/install_service.sh --remove   stop + remove
#   journalctl --user -u brick-sorter -f  logs
set -euo pipefail
cd "$(dirname "$0")/.."
UNIT=$HOME/.config/systemd/user/brick-sorter.service
if [ "${1:-}" = --remove ]; then
  systemctl --user disable --now brick-sorter 2>/dev/null || true
  rm -f "$UNIT"; systemctl --user daemon-reload; echo removed; exit 0
fi
mkdir -p "$(dirname "$UNIT")"
cat > "$UNIT" <<EOF
[Unit]
Description=brick-sorter station (control page on :8080)
After=network-online.target

[Service]
WorkingDirectory=$PWD
ExecStart=$PWD/scripts/run.sh
Restart=on-failure
RestartSec=5
KillSignal=SIGINT
TimeoutStopSec=15

[Install]
WantedBy=default.target
EOF
systemctl --user daemon-reload
systemctl --user enable --now brick-sorter
loginctl enable-linger "$USER" 2>/dev/null || true
systemctl --user --no-pager status brick-sorter | head -5
echo "control page: http://$(hostname -I | awk '{print $1}'):8080"

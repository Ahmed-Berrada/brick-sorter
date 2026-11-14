#!/usr/bin/env bash
# One-shot setup of an Intel Core Ultra (Series 2/3, e.g. the Dell XPS 16 / Core Ultra X9 388H) Ubuntu machine.
#
#   git clone <this repo> ~/brick-sorter && cd ~/brick-sorter && scripts/setup_intel.sh
#
# What it does (each step is skipped when already done; safe to re-run):
#   1. apt packages (build tools, v4l-utils, curl, git, libusb)
#   2. Intel GPU compute runtime (OpenCL + Level Zero) so OpenVINO sees "GPU"
#   3. checks the NPU driver (/dev/accel/accel0) and prints how to install it if missing
#   4. adds you to dialout (arm serial), video (cameras), render (GPU/NPU)
#   5. udev rules: RealSense access, a stable /dev/so101-follower name (by USB serial), no ModemManager grabbing it
#   6. uv + the Python environment (.venv, CPU-only PyTorch for LeRobot's servo code)
#   7. downloads the models (scripts/download_models.sh)
#   8. runs `brick-sorter doctor`
#
# Options: --no-apt (skip 1-3, e.g. no sudo), --no-models, --follower-serial <USB serial> (see ls /dev/serial/by-id),
#          --xpu (PyTorch for the Intel GPU + SmolVLA deps: needed to run SmolVLA / LeRobot policies at usable speed)
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
NO_APT=0; NO_MODELS=0; FOLLOWER_SERIAL=""; XPU=0
while [ $# -gt 0 ]; do
  case "$1" in
    --no-apt) NO_APT=1 ;;
    --no-models) NO_MODELS=1 ;;
    --xpu) XPU=1 ;;
    --follower-serial) FOLLOWER_SERIAL="$2"; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!! %s\033[0m\n' "$*"; }

. /etc/os-release 2>/dev/null || true
step "System: ${PRETTY_NAME:-unknown}, $(uname -r), CPU: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)"

if [ "$NO_APT" = 0 ]; then
  step "1. apt packages"
  sudo apt-get update -y
  sudo apt-get install -y git curl ca-certificates build-essential pkg-config v4l-utils usbutils libusb-1.0-0 \
    libgl1 libglib2.0-0 clinfo

  step "2. Intel GPU compute runtime (OpenCL + Level Zero)"
  if ! clinfo -l 2>/dev/null | grep -qi intel; then
    # Ubuntu 24.04+ ships the runtime; Intel's PPA has newer builds for the newest iGPUs (Arc B390 / Xe3).
    if ! sudo apt-get install -y intel-opencl-icd libze1 libze-intel-gpu1 2>/dev/null; then
      sudo apt-get install -y software-properties-common
      sudo add-apt-repository -y ppa:kobuk-team/intel-graphics
      sudo apt-get update -y
      sudo apt-get install -y intel-opencl-icd libze1 libze-intel-gpu1
    fi
  fi
  clinfo -l 2>/dev/null | grep -i intel || warn "no Intel OpenCL device yet; reboot, then run scripts/setup_intel.sh again"

  step "3. NPU driver"
  if [ -e /dev/accel/accel0 ]; then
    echo "NPU present: $(ls -l /dev/accel/accel0)"
  else
    warn "no /dev/accel/accel0. Install the Intel NPU driver for your kernel from"
    warn "  https://github.com/intel/linux-npu-driver/releases  (the .deb files for your Ubuntu version + level-zero)"
    warn "then reboot. Inspection falls back to GPU/CPU until then."
  fi
fi

step "4. groups: dialout (arm), video (cameras), render (GPU/NPU)"
NEED_RELOGIN=0
for g in dialout video render; do
  if getent group "$g" >/dev/null && ! id -nG "$USER" | tr ' ' '\n' | grep -qx "$g"; then
    sudo usermod -aG "$g" "$USER"; NEED_RELOGIN=1; echo "added $USER to $g"
  fi
done

step "5. udev rules"
RULES=/etc/udev/rules.d/99-brick-sorter.rules
{
  echo '# brick-sorter: SO-101 CH343 USB serial adapters, readable without root, ignored by ModemManager'
  echo 'SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", MODE="0666", ENV{ID_MM_DEVICE_IGNORE}="1"'
  if [ -n "$FOLLOWER_SERIAL" ]; then
    echo "SUBSYSTEM==\"tty\", ATTRS{idVendor}==\"1a86\", ATTRS{serial}==\"$FOLLOWER_SERIAL\", SYMLINK+=\"so101-follower\""
  fi
  echo '# Intel RealSense D4xx'
  echo 'SUBSYSTEMS=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0ad3", MODE="0666", GROUP="plugdev"'
  echo 'KERNEL=="iio*", ATTRS{idVendor}=="8086", MODE="0666", GROUP="plugdev"'
  echo '# NPU'
  echo 'SUBSYSTEM=="accel", KERNEL=="accel*", GROUP="render", MODE="0660"'
} | sudo tee "$RULES" >/dev/null
sudo udevadm control --reload-rules && sudo udevadm trigger || true
echo "wrote $RULES"
ls -l /dev/serial/by-id/ 2>/dev/null || warn "no USB serial devices plugged in right now"
[ -n "$FOLLOWER_SERIAL" ] || echo "tip: re-run with --follower-serial <serial from /dev/serial/by-id> for a fixed /dev/so101-follower"

step "6. Python environment"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
if [ "$XPU" = 1 ]; then
  uv sync --extra smolvla
  # Intel GPU PyTorch replaces the CPU wheels; .venv/.xpu tells scripts/run.sh not to re-sync them away.
  uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/xpu
  touch .venv/.xpu
  uv run --no-sync python -c "import torch; print('torch', torch.__version__, 'xpu available:', torch.xpu.is_available())"
else
  uv sync
  rm -f .venv/.xpu
fi
uv run --no-sync python -c "import openvino as ov; print('OpenVINO', ov.__version__, ov.Core().available_devices)"

if [ "$NO_MODELS" = 0 ]; then
  step "7. models"
  scripts/download_models.sh
fi

if [ ! -f config/local.toml ]; then
  cat > config/local.toml <<'EOF'
# Machine-specific overrides of config/station.toml (only what differs). Examples:
# [arm]
# port = "/dev/so101-follower"
# [inspect]
# device = "NPU"
# crop = [160, 80, 320, 320]
EOF
  echo "created config/local.toml (edit for this machine)"
fi

step "8. doctor"
uv run --no-sync brick-sorter -c config/local.toml doctor || true

step "done"
[ "$NEED_RELOGIN" = 1 ] && warn "log out and back in (or reboot) so the new groups apply"
cat <<EOF
Next:
  uv run brick-sorter -c config/local.toml cameras                       # find/test/assign wrist, top, inspect
  uv run brick-sorter -c config/local.toml calib                         # motors vs calibration file
  uv run brick-sorter -c config/local.toml teach home yellow orange purple
  uv run brick-sorter -c config/local.toml bench --torch cpu             # OPTIMIZE table
  scripts/run.sh                                                         # http://$(hostname -I 2>/dev/null | awk '{print $1}'):8080
EOF

#!/usr/bin/env bash
# Download and verify the models into models/ from this repo's GitHub release:
#   models/act            LeRobot ACT checkpoint (pick-and-present policy, PyTorch baseline)
#   models/act_openvino   OpenVINO export of it (normalisation + top-camera crop inside the graph)
#   models/patchcore      Anomalib PatchCore (marked vs plain brick face, 320x320 crops)
#   models/yolo           YOLO classifier (plain / circle / cross)
# Re-running skips what is already there.
#   MODELS_REPO=<owner>/<repo>  MODELS_TAG=<release tag>  to use another release (e.g. your fine-tuned policy).
# The release is private while the repository is: run `gh auth login` first (gh is used when present).
set -euo pipefail
cd "$(dirname "$0")/.."
MODELS_REPO=${MODELS_REPO:-Ahmed-Berrada/brick-sorter}
MODELS_TAG=${MODELS_TAG:-models-v1}
M=models; mkdir -p "$M"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

fetch() {  # asset name -> $TMP
  if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
    gh release download "$MODELS_TAG" --repo "$MODELS_REPO" --pattern "$1" --dir "$TMP" --clobber
  else
    curl -fL --retry 3 -o "$TMP/$1" "https://github.com/$MODELS_REPO/releases/download/$MODELS_TAG/$1"
  fi
}

fetch SHA256SUMS
for name in act act_openvino patchcore yolo; do
  if [ -d "$M/$name" ] && [ -n "$(ls -A "$M/$name")" ]; then echo "$name: present"; continue; fi
  echo "$name: downloading"
  fetch "$name.tar.gz"
  (cd "$TMP" && grep " $name.tar.gz\$" SHA256SUMS | sha256sum -c -)
  tar -xzf "$TMP/$name.tar.gz" -C "$M"
  rm -f "$TMP/$name.tar.gz"
done
du -sh "$M"/* 2>/dev/null

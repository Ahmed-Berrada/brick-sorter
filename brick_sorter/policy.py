"""Pick-and-present policies. All take (state deg, {camera: BGR frame}) and return the next joint target.

* OpenVINOACT: the Physical AI Studio ACT export (act.xml). Normalisation and the top-camera training crop are
  inside the graph: feed the raw state and full RGB frames in [0, 1]. Verified against LeRobot PyTorch on the
  same inputs: max |diff| 6e-5 over the 100x6 chunk. No PyTorch needed; runs on CPU / GPU / NPU.
* TorchACT: the LeRobot checkpoint (baseline for the OPTIMIZE table; xpu or cpu).
* FakePolicy: scripted reach -> grasp -> present for dry runs.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from brick_sorter import CAMERAS

log = logging.getLogger(__name__)


class Policy:
    chunk_size = 100

    def __init__(self, n_action_steps: int = 100):
        self.n_action_steps = n_action_steps
        self.queue: list[np.ndarray] = []
        self.latencies_ms: list[float] = []  # per predicted chunk

    def reset(self) -> None:
        self.queue = []

    def predict_chunk(self, state: np.ndarray, frames: dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def act(self, state: np.ndarray, frames: dict[str, np.ndarray]) -> np.ndarray:
        if not self.queue:
            t = time.perf_counter()
            chunk = self.predict_chunk(state, frames)
            self.latencies_ms.append((time.perf_counter() - t) * 1e3)
            self.queue = list(chunk[: self.n_action_steps])
        return np.asarray(self.queue.pop(0), dtype=np.float32)

    def warmup(self, n: int = 2) -> None:
        z = {c: np.zeros((480, 640, 3), np.uint8) for c in CAMERAS}
        for _ in range(n):
            self.predict_chunk(np.zeros(6, np.float32), z)
        self.latencies_ms.clear()


def _available(core, device: str) -> str:
    devs = core.available_devices
    if any(d.startswith(device) for d in devs):
        return device
    log.warning("OpenVINO device %s not available (%s); using CPU", device, devs)
    return "CPU"


class OpenVINOACT(Policy):
    def __init__(self, export_dir: str | Path, device: str = "GPU", n_action_steps: int = 100,
                 cache_dir: str | Path | None = None):
        import openvino as ov

        super().__init__(n_action_steps)
        export_dir = Path(export_dir)
        xml = export_dir / "act.xml"
        if not xml.exists():
            raise FileNotFoundError(f"{xml} missing (scripts/download_models.sh, or convert with Studio)")
        core = ov.Core()
        self.device = _available(core, device)
        cfg = {"PERFORMANCE_HINT": "LATENCY"}
        if cache_dir:
            cfg["CACHE_DIR"] = str(cache_dir)
        if self.device == "GPU":
            cfg["INFERENCE_PRECISION_HINT"] = "f16"
        self.compiled = core.compile_model(str(xml), self.device, cfg)
        self.inputs = [i.any_name for i in self.compiled.inputs]
        self.image_hw: tuple[int, int] | None = None  # Studio letterbox-resizes every camera to this
        manifest = export_dir / "manifest.json"
        if manifest.exists():
            m = json.loads(manifest.read_text())["model"]
            for f in m["output_features"]:
                self.chunk_size = f["init_args"]["shape"][0]
            for pre in m.get("preprocessors", []):
                if pre.get("type") == "resize":
                    self.image_hw = tuple(pre["image_resolution"])
        missing = [f"images.{c}" for c in CAMERAS if f"images.{c}" not in self.inputs]
        if missing or "state" not in self.inputs:
            raise ValueError(f"{xml}: expected inputs state + images.wrist/top/inspect, got {self.inputs}: name the "
                             f"cameras wrist, top and inspect when recording/training in Studio")
        log.info("OpenVINO ACT on %s, inputs %s, images %s", self.device, self.inputs, self.image_hw)

    @staticmethod
    def letterbox(img: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
        h, w = img.shape[:2]
        if (h, w) == tuple(hw):
            return img
        s = min(hw[0] / h, hw[1] / w)
        nh, nw = round(h * s), round(w * s)
        out = np.zeros((hw[0], hw[1], 3), img.dtype)
        y, x = (hw[0] - nh) // 2, (hw[1] - nw) // 2
        out[y:y + nh, x:x + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        return out

    def predict_chunk(self, state, frames):
        feed = {"state": np.asarray(state, np.float32)[None]}
        for cam in CAMERAS:
            rgb = cv2.cvtColor(frames[cam], cv2.COLOR_BGR2RGB)
            if self.image_hw:
                rgb = self.letterbox(rgb, self.image_hw)
            feed[f"images.{cam}"] = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        return np.asarray(self.compiled(feed)[0])[0]


class TorchACT(Policy):
    def __init__(self, checkpoint: str | Path, device: str = "cpu", n_action_steps: int = 100,
                 top_crop=(0, 0, 560, 480)):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        super().__init__(n_action_steps)
        self.torch = torch
        cfg = PreTrainedConfig.from_pretrained(str(checkpoint))
        cfg.device = device
        self.device = device
        self.policy = ACTPolicy.from_pretrained(str(checkpoint), config=cfg).to(device).eval()
        self.pre, self.post = make_pre_post_processors(
            self.policy.config, pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": device}})
        self.chunk_size = cfg.chunk_size
        self.top_crop = top_crop

    def predict_chunk(self, state, frames):
        torch = self.torch
        obs = {"observation.state": torch.from_numpy(np.asarray(state, np.float32))[None]}
        for cam in CAMERAS:
            img = cv2.cvtColor(frames[cam], cv2.COLOR_BGR2RGB)
            if cam == "top" and self.top_crop:
                x, y, w, h = self.top_crop
                img = img[y:y + h, x:x + w]
            obs[f"observation.images.{cam}"] = torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255.0
        with torch.inference_mode():
            chunk = self.policy.predict_action_chunk(self.pre(obs))
            return self.post(chunk)[0].float().cpu().numpy()


class FakePolicy(Policy):
    """Scripted pick: reach down, close the gripper, lift to the inspect pose and hold still."""

    REACH = np.array([10.0, -20.0, 40.0, 60.0, 0.0, 40.0], np.float32)
    GRASP = np.array([10.0, -20.0, 40.0, 60.0, 0.0, 6.0], np.float32)
    PRESENT = np.array([-30.0, -60.0, 60.0, 20.0, 0.0, 6.0], np.float32)

    def __init__(self, n_action_steps: int = 100, latency_ms: float = 5.0):
        super().__init__(n_action_steps)
        self.latency_ms = latency_ms
        self.device = "sim"

    def predict_chunk(self, state, frames):
        time.sleep(self.latency_ms / 1e3)
        holding = 3.0 <= state[5] <= 15.0
        at_brick = float(np.max(np.abs(state[:5] - self.REACH[:5]))) < 2.0
        if holding:
            goal = self.PRESENT  # brick in the jaws: lift it to the inspect camera
        elif at_brick and state[5] > 20:
            goal = self.GRASP  # open at the brick: close
        else:
            goal = self.REACH  # anything else: open and reach
        # short chunks so the scripted phases react quickly
        return np.repeat(goal[None], 10, axis=0)


def make_policy(cfg: dict) -> Policy:
    p = cfg["policy"]
    n = p.get("n_action_steps", 100)
    if p["backend"] == "openvino":
        return OpenVINOACT(p["path"], p.get("device", "GPU"), n, cache_dir=Path(p["path"]) / "cache")
    if p["backend"] == "torch":
        return TorchACT(p["path"], p.get("device", "cpu"), n)
    if p["backend"] == "fake":
        return FakePolicy(n)
    raise ValueError(f"unknown policy backend {p['backend']!r}")

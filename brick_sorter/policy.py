"""Pick-and-present policies. All take (state deg, {camera: BGR frame}) and return the next joint target.

* OpenVINOACT: the Physical AI Studio ACT export (act.xml). Normalisation and the top-camera training crop are
  inside the graph: feed the raw state and full RGB frames in [0, 1]. Verified against LeRobot PyTorch on the
  same inputs: max |diff| 6e-5 over the 100x6 chunk. No PyTorch needed; runs on CPU / GPU / NPU.
* LeRobotPolicy: any LeRobot checkpoint in PyTorch (ACT, SmolVLA, ...), with a task string for VLAs.
  TorchACT is it preconfigured for the original ACT (baseline for the OPTIMIZE table).
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


def torch_device(want: str) -> str:
    import torch

    ok = {"cpu": True, "xpu": hasattr(torch, "xpu") and torch.xpu.is_available(),
          "cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available()}
    if ok.get(want.split(":")[0], False):
        return want
    log.warning("torch device %s not available; using cpu (this PyTorch is %s)", want, torch.__version__)
    return "cpu"


class LeRobotPolicy(Policy):
    """Any LeRobot checkpoint (ACT, SmolVLA, Diffusion, pi0/pi0.5 ...) in PyTorch, with its saved processors.

    * Camera keys: the checkpoint's `observation.images.<key>` inputs are fed from our cameras of the same name;
      `camera_map` renames ({"camera1": "wrist"} feeds wrist into observation.images.camera1).
    * Images are resized to the size the checkpoint expects; `crops` ({"top": [x, y, w, h]}) are applied first.
    * `task` is the language instruction for VLAs (SmolVLA, pi0...). It must be the exact task string the
      demos were recorded with; ACT ignores it.
    """

    def __init__(self, checkpoint: str | Path, device: str = "cpu", n_action_steps: int | None = None,
                 task: str = "", camera_map: dict | None = None, crops: dict | None = None,
                 robot_type: str = "so101_follower"):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        self.torch = torch
        checkpoint = Path(checkpoint)
        if not (checkpoint / "config.json").exists():
            raise FileNotFoundError(f"{checkpoint}: not a LeRobot checkpoint (no config.json); use the "
                                    "pretrained_model folder of the training output")
        self.device = torch_device(device)
        cfg = PreTrainedConfig.from_pretrained(str(checkpoint))
        cfg.device = self.device
        self.kind = cfg.type
        if n_action_steps and hasattr(cfg, "n_action_steps"):
            cfg.n_action_steps = min(n_action_steps, getattr(cfg, "chunk_size", n_action_steps))
        super().__init__(getattr(cfg, "n_action_steps", 1))
        self.chunk_size = getattr(cfg, "chunk_size", self.n_action_steps)
        self.policy = get_policy_class(cfg.type).from_pretrained(str(checkpoint), config=cfg).to(self.device).eval()
        self.pre, self.post = make_pre_post_processors(
            self.policy.config, pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": self.device}})
        self.task, self.robot_type = task, robot_type
        self.crops = {k: tuple(int(v) for v in b) for k, b in (crops or {}).items()}
        # policy image key -> (our camera, (H, W))
        rename = dict(camera_map or {})
        self.images: dict[str, tuple[str, tuple[int, int]]] = {}
        for key, feat in self.policy.config.input_features.items():
            if not key.startswith("observation.images."):
                continue
            suffix = key.removeprefix("observation.images.")
            cam = rename.get(suffix, suffix)
            if cam not in CAMERAS:
                raise ValueError(f"policy input {key} has no camera: name the dataset cameras wrist/top/inspect, "
                                 f"or set [policy] camera_map = {{ {suffix} = \"wrist\" }}")
            self.images[key] = (cam, tuple(feat.shape[-2:]))
        if self.kind in VLA_TYPES and not task:
            log.warning("%s is language-conditioned but [policy] task is empty: set it to the exact task string "
                        "used when recording", self.kind)
        log.info("LeRobot %s on %s: chunk %s, executing %d per chunk, cameras %s, task %r", self.kind, self.device,
                 self.chunk_size, self.n_action_steps, {k: v[0] for k, v in self.images.items()}, task)

    def _observation(self, state, frames) -> dict:
        obs = {"observation.state": np.asarray(state, np.float32)}
        for key, (cam, (h, w)) in self.images.items():
            img = cv2.cvtColor(frames[cam], cv2.COLOR_BGR2RGB)
            if cam in self.crops:
                x, y, cw, ch = self.crops[cam]
                img = img[y:y + ch, x:x + cw]
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            obs[key] = np.ascontiguousarray(img)
        return obs

    def reset(self) -> None:
        super().reset()
        self.policy.reset()

    def predict_chunk(self, state, frames):
        from lerobot.policies.utils import prepare_observation_for_inference

        torch = self.torch
        with torch.inference_mode():
            batch = prepare_observation_for_inference(self._observation(state, frames), torch.device(self.device),
                                                      self.task, self.robot_type)
            chunk = self.policy.predict_action_chunk(self.pre(batch))
            return self.post(chunk)[0].float().cpu().numpy()


VLA_TYPES = {"smolvla", "pi0", "pi05", "pi0_fast", "groot", "xvla", "molmoact2"}


class TorchACT(LeRobotPolicy):
    """The original downloaded ACT (top camera trained on the x 0-560 crop): the PyTorch baseline for `bench`."""

    def __init__(self, checkpoint: str | Path, device: str = "cpu", n_action_steps: int = 100):
        super().__init__(checkpoint, device, n_action_steps, crops={"top": (0, 0, 560, 480)})


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
    if p["backend"] in ("lerobot", "torch"):
        return LeRobotPolicy(p["path"], p.get("device", "cpu"), n, task=p.get("task", ""),
                             camera_map=p.get("camera_map"), crops=p.get("crops"))
    if p["backend"] == "fake":
        return FakePolicy(n)
    raise ValueError(f"unknown policy backend {p['backend']!r}")

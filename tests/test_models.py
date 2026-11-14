"""Real-model checks; skipped when the models are not downloaded (scripts/download_models.sh)."""

import numpy as np
import pytest

from brick_sorter import ROOT
from brick_sorter import config as C

ov = pytest.importorskip("openvino")
HAS_ACT = (ROOT / "models/act_openvino/act.xml").exists()
HAS_INSPECT = (ROOT / "models/patchcore/model.xml").exists() and (ROOT / "models/yolo/best.xml").exists()
FRAMES = {c: np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8) for c in ("wrist", "top", "inspect")}


@pytest.mark.skipif(not HAS_ACT, reason="ACT export not downloaded")
def test_openvino_act_chunk_shape_and_range():
    from brick_sorter.policy import OpenVINOACT

    p = OpenVINOACT(ROOT / "models/act_openvino", "CPU", n_action_steps=10)
    chunk = p.predict_chunk(np.array([0, -100, 95, 5, 0, 1], np.float32), FRAMES)
    assert chunk.shape == (100, 6) and np.isfinite(chunk).all()
    assert np.abs(chunk[:, :5]).max() < 360
    a = p.act(np.zeros(6, np.float32), FRAMES)
    assert a.shape == (6,) and len(p.queue) == 9


@pytest.mark.skipif(not (HAS_ACT and (ROOT / "models/act/model.safetensors").exists()), reason="checkpoint missing")
def test_openvino_matches_pytorch():
    pytest.importorskip("lerobot")
    from brick_sorter.policy import OpenVINOACT, TorchACT

    state = np.array([8.9, -102.6, 96.4, 75.1, 5.7, 47.2], np.float32)
    a = OpenVINOACT(ROOT / "models/act_openvino", "CPU").predict_chunk(state, FRAMES)
    b = TorchACT(ROOT / "models/act", "cpu").predict_chunk(state, FRAMES)
    assert np.abs(a - b).max() < 1e-3


@pytest.mark.skipif(not HAS_INSPECT, reason="inspection models not downloaded")
def test_inspector_runs_and_returns_an_outcome():
    from brick_sorter.inspector import OUTCOMES, Inspector

    cfg = C.load()
    cfg["inspect"]["device"] = "CPU"
    ins = Inspector(cfg).inspect([FRAMES["inspect"]] * 3)
    assert ins.outcome in OUTCOMES
    assert set(ins.probabilities) == {"plain", "circle", "cross"}
    assert abs(sum(ins.probabilities.values()) - 1) < 1e-2
    assert ins.heatmap_jpeg[:2] == b"\xff\xd8"


@pytest.mark.skipif(not (ROOT / "models/act/config.json").exists(), reason="checkpoint missing")
def test_lerobot_backend_loads_act_and_checks_cameras():
    pytest.importorskip("lerobot")
    from brick_sorter.policy import LeRobotPolicy

    p = LeRobotPolicy(ROOT / "models/act", "cpu", n_action_steps=10, crops={"top": (0, 0, 560, 480)})
    assert p.kind == "act" and p.n_action_steps == 10
    assert p.images["observation.images.top"] == ("top", (480, 560))
    a = p.act(np.array([0, -100, 95, 5, 0, 1], np.float32), FRAMES)
    assert a.shape == (6,) and len(p.queue) == 9
    with pytest.raises(ValueError, match="camera_map"):
        LeRobotPolicy(ROOT / "models/act", "cpu", camera_map={"wrist": "gripper_cam"})

"""DETECT: Anomalib PatchCore (marked or not + heatmap) and a YOLO classifier (plain / circle / cross).

Decision rule (validated on held-out captures): both models must agree, otherwise "uncertain".
    anomalous  + classifier cross/circle (confident)  -> that mark
    normal     + classifier plain (confident)         -> plain
    anything else                                     -> uncertain  (re-inspect, then the fallback bowl)
Without a classifier: anomalous -> cross (defective), normal -> plain.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)
OUTCOMES = ("plain", "circle", "cross", "uncertain")


@dataclass
class Inspection:
    outcome: str  # plain | circle | cross | uncertain
    score: float  # mean PatchCore score
    threshold: float
    anomalous: bool
    probabilities: dict[str, float] = field(default_factory=dict)
    reason: str = ""
    latency_ms: dict[str, float] = field(default_factory=dict)
    device: str = ""
    heatmap_jpeg: bytes | None = None  # crop with the anomaly map overlaid (for the web page)

    def record(self) -> dict:
        d = asdict(self)
        d.pop("heatmap_jpeg")
        d["score"] = round(d["score"], 6)
        d["probabilities"] = {k: round(v, 3) for k, v in d["probabilities"].items()}
        d["latency_ms"] = {k: round(v, 2) for k, v in d["latency_ms"].items()}
        return d


def decide(anomalous: bool, probs: dict[str, float] | None, min_conf: float, min_margin: float,
           near_threshold: bool = False) -> tuple[str, str]:
    if near_threshold:
        return "uncertain", "anomaly score inside the uncertainty band"
    if not probs:
        return ("cross", "anomalous (no classifier)") if anomalous else ("plain", "normal (no classifier)")
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    (label, conf), (_, second) = ranked[0], ranked[1]
    if conf < min_conf or conf - second < min_margin:
        return "uncertain", f"classifier unsure ({label} {conf:.2f}, margin {conf - second:.2f})"
    if anomalous:
        if label in ("cross", "circle"):
            return label, f"anomalous and classifier says {label} ({conf:.2f})"
        return "uncertain", f"anomalous but classifier says plain ({conf:.2f})"
    if label == "plain":
        return "plain", f"normal and classifier says plain ({conf:.2f})"
    return "uncertain", f"normal but classifier says {label} ({conf:.2f})"


def crop(frame: np.ndarray, box) -> np.ndarray:
    x, y, w, h = (int(v) for v in box)
    H, W = frame.shape[:2]
    x, y = max(0, min(x, W - 1)), max(0, min(y, H - 1))
    c = frame[y:min(H, y + h), x:min(W, x + w)]
    return cv2.resize(c, (320, 320)) if c.shape[:2] != (320, 320) else c


def _compile(core, xml: Path, device: str, shape=None):
    model = core.read_model(str(xml))
    if shape is not None:
        model.reshape({model.inputs[0]: shape})  # static shape: required by the NPU, faster elsewhere
    devs = core.available_devices
    if not any(d.startswith(device) for d in devs):
        log.warning("OpenVINO device %s not available (%s); using CPU", device, devs)
        device = "CPU"
    try:
        return core.compile_model(model, device, {"PERFORMANCE_HINT": "LATENCY",
                                                  "CACHE_DIR": str(xml.parent / "cache")}), device
    except Exception as e:  # noqa: BLE001  e.g. an op the NPU compiler rejects
        if device == "CPU":
            raise
        log.warning("compile on %s failed (%s); using CPU", device, e)
        return core.compile_model(model, "CPU", {"PERFORMANCE_HINT": "LATENCY"}), "CPU"


class Inspector:
    def __init__(self, cfg: dict):
        import openvino as ov

        c = cfg["inspect"]
        self.cfg = c
        core = ov.Core()
        self.patchcore, self.device = _compile(core, Path(c["patchcore"]) / "model.xml", c.get("device", "CPU"),
                                               [1, 3, 320, 320])
        self.classifier = None
        self.classes: list[str] = []
        if c.get("classifier"):
            import json

            meta = json.loads((Path(c["classifier"]) / "classifier.json").read_text())
            self.classes = meta["classes"]
            self.cls_size = int(meta.get("input_size", 320))
            self.classifier, self.cls_device = _compile(core, Path(c["classifier"]) / meta["model"],
                                                        c.get("device", "CPU"), [1, 3, self.cls_size, self.cls_size])
        log.info("inspector: PatchCore on %s, classifier %s", self.device,
                 f"{self.classes} on {self.cls_device}" if self.classifier else "none")

    @staticmethod
    def _tensor(bgr: np.ndarray, size: int = 320) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (size, size):
            rgb = cv2.resize(rgb, (size, size))
        return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0

    def inspect(self, frames: list[np.ndarray]) -> Inspection:
        c = self.cfg
        crops = [crop(f, c["crop"]) for f in frames]
        t0 = time.perf_counter()
        scores, maps = [], []
        for cr in crops:
            out = self.patchcore([self._tensor(cr)])
            vals = {n: v for p, v in out.items() for n in p.get_names()}  # outputs carry several names
            scores.append(float(np.asarray(vals["pred_score"]).reshape(-1)[0]))
            maps.append(np.asarray(vals["anomaly_map"]).squeeze())
        t1 = time.perf_counter()
        probs = None
        if self.classifier is not None:
            ps = [np.asarray(self.classifier([self._tensor(cr, self.cls_size)])[0]).reshape(-1) for cr in crops]
            probs = dict(zip(self.classes, map(float, np.mean(ps, axis=0))))
        t2 = time.perf_counter()
        score = float(np.mean(scores))
        thr = float(c["threshold"])
        near = c.get("uncertain_band", 0.0) > 0 and abs(score - thr) < c["uncertain_band"]
        outcome, reason = decide(score >= thr, probs, c["min_confidence"], c["min_margin"], near)
        return Inspection(outcome, score, thr, score >= thr, probs or {}, reason,
                          {"anomalib": (t1 - t0) * 1e3 / len(crops), "classifier": (t2 - t1) * 1e3 / len(crops)},
                          self.device, overlay(crops[-1], maps[-1]))


def overlay(crop_bgr: np.ndarray, amap: np.ndarray) -> bytes | None:
    a = np.clip(amap, 0, 1)
    heat = cv2.applyColorMap((cv2.resize(a, crop_bgr.shape[1::-1]) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    ok, buf = cv2.imencode(".jpg", cv2.addWeighted(crop_bgr, 0.55, heat, 0.45, 0))
    return buf.tobytes() if ok else None


class FakeInspector:
    """Reads the mark the simulated scene painted on the brick (or a scripted sequence)."""

    device = "sim"

    def __init__(self, scene=None, threshold: float = 0.999):
        self.scene, self.threshold = scene, threshold

    def inspect(self, frames: list[np.ndarray]) -> Inspection:
        time.sleep(0.02)
        truth = self.scene.current_mark() if self.scene is not None else "plain"
        if truth == "faint":  # unreadable the first time, readable after turning it
            truth = "circle" if self.scene.reinspections > 0 else "unclear"
        if truth == "unclear":
            return Inspection("uncertain", self.threshold, self.threshold, True, {"plain": .4, "circle": .3, "cross": .3},
                              "classifier unsure (sim)", {"anomalib": 8.0, "classifier": 2.0}, "sim")
        anomalous = truth != "plain"
        probs = {k: (0.9 if k == truth else 0.05) for k in ("plain", "circle", "cross")}
        outcome, reason = decide(anomalous, probs, 0.8, 0.2)
        return Inspection(outcome, 1.0 if anomalous else 0.9, self.threshold, anomalous, probs, reason + " (sim)",
                          {"anomalib": 8.0, "classifier": 2.0}, "sim")


def make_inspector(cfg: dict, scene=None):
    if cfg["inspect"]["backend"] == "fake":
        return FakeInspector(scene, cfg["inspect"]["threshold"])
    return Inspector(cfg)

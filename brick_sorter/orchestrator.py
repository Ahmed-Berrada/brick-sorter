"""The closed loop: HOME -> PICK (policy) -> INSPECT (Anomalib + classifier) -> DECIDE (rules) -> PLACE -> HOME.

Runs in its own thread. The web page (or tests) control it with start / pause / resume / stop / estop / next, and
read `snapshot()` for live status. Every brick is logged to runs/<timestamp>/bricks.jsonl; per-stage latencies
are kept for the OPTIMIZE table (median / p99).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

from brick_sorter import JOINTS
from brick_sorter.arm import Arm, Poses
from brick_sorter.inspector import Inspection
from brick_sorter.reasoner import Decision, decide, required_poses

log = logging.getLogger(__name__)


class Stop(Exception):
    """Raised inside the loop when the operator stops or e-stops."""


def pct(values, q) -> float | None:
    return round(float(np.percentile(values, q)), 2) if len(values) else None


class Orchestrator:
    def __init__(self, cfg: dict, arm: Arm, cameras: dict, policy, inspector, scene=None, run_dir: Path | None = None):
        self.cfg, self.arm, self.cameras, self.policy, self.inspector, self.scene = cfg, arm, cameras, policy, inspector, scene
        self.rules = cfg["rules"]
        self.poses = Poses(Path(cfg["arm"]["calibration_dir"]) / "poses.json")
        off = cfg["arm"].get("joint_offset", {})
        self.offset = np.array([off.get(j.removesuffix(".pos"), 0.0) for j in JOINTS], np.float32)
        self.run_dir = run_dir or Path(cfg["log"]["dir"]) / datetime.now().strftime("%Y%m%d-%H%M%S")
        self.state = "idle"
        self.detail = ""
        self.brick = 0
        self.counts: dict[str, int] = {}
        self.events: deque = deque(maxlen=200)
        self.last_inspection: Inspection | None = None
        self.last_decision: Decision | None = None
        self.lat: dict[str, list[float]] = {"policy_chunk_ms": [], "anomalib_ms": [], "classifier_ms": [],
                                            "pick_s": [], "cycle_s": []}
        self.t_start: float | None = None
        self.error: str | None = None
        self.step_mode = False
        self._go = threading.Event()  # step mode: NEXT pressed
        self._run = threading.Event()  # loop allowed to proceed (cleared by pause)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.max_bricks: int | None = None

    # ------------------------------------------------------------------ control
    def missing_poses(self) -> list[str]:
        return [p for p in required_poses(self.rules) if not self.poses.has(p)]

    def start(self, max_bricks: int | None = None) -> None:
        if self._thread and self._thread.is_alive():
            self._run.set()
            return
        missing = self.missing_poses()
        if missing:
            raise RuntimeError(f"teach these poses first: {missing}  (brick-sorter teach {' '.join(missing)})")
        self.max_bricks = max_bricks
        self._stop.clear()
        self._run.set()
        self.arm.abort.clear()
        self.error = None
        self._thread = threading.Thread(target=self._main, name="orchestrator", daemon=True)
        self._thread.start()

    def pause(self) -> None:
        self._run.clear()
        self.event("paused (the arm finishes the current move, then waits)")

    def resume(self) -> None:
        self._run.set()
        self.event("resumed")

    def stop(self) -> None:
        """Stop after aborting the current move; the arm holds where it is."""
        self._stop.set()
        self._run.set()
        self._go.set()
        self.arm.abort.set()
        self.event("stop requested")

    def estop(self) -> None:
        self.stop()
        try:
            self.arm.hold()
        except Exception as e:  # noqa: BLE001
            log.error("estop hold failed: %s", e)
        self.event("E-STOP: holding position")

    def next(self) -> None:
        self._go.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    # ------------------------------------------------------------------ status
    def event(self, msg: str, **data) -> None:
        e = {"t": round(time.time(), 2), "brick": self.brick, "state": self.state, "msg": msg, **data}
        self.events.append(e)
        log.info("[%s #%d] %s", self.state, self.brick, msg)
        self._write("events.jsonl", e)

    def _write(self, name: str, rec: dict) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with open(self.run_dir / name, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def snapshot(self) -> dict:
        elapsed = (time.time() - self.t_start) if self.t_start else 0.0
        placed = sum(self.counts.values())
        return {
            "state": self.state, "detail": self.detail, "brick": self.brick, "counts": self.counts,
            "error": self.error, "running": bool(self._thread and self._thread.is_alive()),
            "paused": not self._run.is_set(), "step_mode": self.step_mode,
            "tasks_per_min": round(placed / elapsed * 60, 2) if elapsed > 5 and placed else None,
            "inspection": self.last_inspection.record() if self.last_inspection else None,
            "decision": self.last_decision.__dict__ if self.last_decision else None,
            "latency": {k: {"n": len(v), "median": pct(v, 50), "p99": pct(v, 99)} for k, v in self.lat.items()},
            "devices": {"policy": getattr(self.policy, "device", "?"), "inspector": getattr(self.inspector, "device", "?")},
            "missing_poses": self.missing_poses(),
            "cameras": {n: c.info() for n, c in self.cameras.items()},
            "events": list(self.events)[-40:],
            "run_dir": str(self.run_dir),
        }

    # ------------------------------------------------------------------ loop
    def _set(self, state: str, detail: str = "") -> None:
        self.state, self.detail = state, detail

    def _checkpoint(self) -> None:
        if self._stop.is_set():
            raise Stop()
        if not self._run.is_set():
            self._set("paused", self.state)
            self._run.wait()
            if self._stop.is_set():
                raise Stop()

    def _move(self, pose: np.ndarray, speed: float | None = None) -> None:
        self._checkpoint()
        if not self.arm.move_to(pose, speed):
            raise Stop()

    def _frames(self) -> dict[str, np.ndarray]:
        return {n: c.latest()[0] for n, c in self.cameras.items()}

    def _main(self) -> None:
        self.t_start = time.time()
        self.event("loop started", config={"rules": self.rules, "policy": self.cfg["policy"]["backend"],
                                           "inspect": self.cfg["inspect"]["backend"]})
        misses = 0
        try:
            self._set("homing")
            self._move(self.poses.get("home"))
            while not self._stop.is_set():
                if self.max_bricks is not None and self.brick >= self.max_bricks:
                    self.event(f"done: {self.max_bricks} bricks")
                    break
                if self.step_mode:
                    self._set("waiting", "press NEXT for the next brick")
                    self._go.clear()
                    self._go.wait()
                self._checkpoint()
                self.brick += 1
                t_cycle = time.time()
                if not self._pick():
                    self.brick -= 1
                    misses += 1
                    self._move(self.poses.get("home"))
                    if misses >= 2:
                        self.event("no brick presented twice in a row: table empty? stopping")
                        break
                    continue
                misses = 0
                decision = self._inspect_and_decide()
                self._place(decision)
                self.lat["cycle_s"].append(time.time() - t_cycle)
            self._set("idle", "finished")
        except Stop:
            self._set("stopped", "operator stop; arm holding")
            try:
                self.arm.abort.clear()
                self.arm.hold()
            except Exception:  # noqa: BLE001
                pass
        except Exception as e:  # noqa: BLE001
            log.exception("loop failed")
            self.error = f"{type(e).__name__}: {e}"
            self._set("error", self.error)
            try:
                self.arm.hold()
            except Exception:  # noqa: BLE001
                pass
        self.event("loop ended", counts=self.counts)
        self._write("summary.json", self.snapshot())

    def _pick(self) -> bool:
        p = self.cfg["policy"]
        self._set("picking", "policy: pick up the brick and show it to the inspect camera")
        self.policy.reset()
        fps = self.cfg["arm"]["fps"]
        lo, hi = p["holding_gripper"]
        still_n = max(1, int(p["present_still_s"] * fps))
        recent: deque = deque(maxlen=still_n)
        t0 = time.time()
        n_chunks = len(self.policy.latencies_ms)
        while True:
            self._checkpoint()
            t = time.perf_counter()
            state = self.arm.read_state()
            target = self.policy.act(state - self.offset, self._frames()) + self.offset
            self.arm.send(target)
            if len(self.policy.latencies_ms) > n_chunks:
                self.lat["policy_chunk_ms"].extend(self.policy.latencies_ms[n_chunks:])
                n_chunks = len(self.policy.latencies_ms)
            recent.append(state)
            holding = lo <= state[5] <= hi
            still = len(recent) == still_n and float(np.ptp(np.stack(recent)[:, :5], axis=0).max()) < 3.0
            if holding and still and time.time() - t0 > 1.0:
                self.lat["pick_s"].append(time.time() - t0)
                self.event(f"brick presented after {time.time() - t0:.1f}s")
                self.arm.hold()
                return True
            if time.time() - t0 > p["pick_timeout"]:
                self.event(f"no brick presented within {p['pick_timeout']:.0f}s")
                self.arm.hold()
                return False
            time.sleep(max(0.0, 1.0 / fps - (time.perf_counter() - t)))

    def _grab_inspect_frames(self) -> list[np.ndarray]:
        cam = self.cameras["inspect"]
        frames, seq = [], cam.seq
        for _ in range(self.cfg["inspect"]["frames"]):
            seq = cam.wait_new(seq, 2.0)
            frames.append(cam.latest()[0])
        return frames

    def _inspect_and_decide(self) -> Decision:
        c = self.cfg["inspect"]
        attempt = 0
        while True:
            attempt += 1
            self._set("inspecting", f"inspection {attempt}")
            time.sleep(c["settle_s"])
            self._checkpoint()
            ins = self.inspector.inspect(self._grab_inspect_frames())
            self.last_inspection = ins
            self.lat["anomalib_ms"].append(ins.latency_ms.get("anomalib", 0.0))
            self.lat["classifier_ms"].append(ins.latency_ms.get("classifier", 0.0))
            d = decide(ins.outcome, attempt, self.rules)
            self.last_decision = d
            self.event(d.reason, inspection=ins.record())
            if d.action == "place":
                self._write("bricks.jsonl", {"brick": self.brick, "t": time.time(), "attempts": attempt,
                                             "decision": d.__dict__, "inspection": ins.record()})
                return d
            self._set("reinspecting", "turning the brick for another look")
            roll = self.rules.get("reinspect_roll_deg", 0.0)
            if roll:
                pose = self.arm.read_state()
                pose[4] += roll if attempt % 2 else -2 * roll
                self._move(pose, 30.0)
            if self.scene is not None:
                self.scene.reinspections += 1

    def _place(self, d: Decision) -> None:
        pl = self.cfg["place"]
        self._set("placing", f"{d.outcome} -> {d.bowl} bowl")
        grip = float(self.arm.read_state()[5])
        approach = f"{d.bowl}_approach"
        if self.poses.has(approach):
            pose = self.poses.get(approach)
            pose[5] = grip
            self._move(pose)
        pose = self.poses.get(d.bowl)
        pose[5] = grip  # keep holding while carrying
        self._move(pose)
        pose[5] = pl["open_gripper"]
        self._move(pose)
        time.sleep(pl["dwell_s"])
        if self.scene is not None:
            self.scene.place_done(d.bowl)
        self.counts[d.bowl] = self.counts.get(d.bowl, 0) + 1
        self.event(f"placed in the {d.bowl} bowl", bowl=d.bowl, outcome=d.outcome, forced=d.forced)
        self._set("returning")
        self._move(self.poses.get("home"))

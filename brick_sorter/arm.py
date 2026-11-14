"""SO-101 follower with safety checks, plus a simulated arm for dry runs.

Safety rules enforced here:
* Torque is never enabled unless the calibration in the motors matches calibration/follower/follower.json.
  LeRobot's connect(calibrate=False) silently uses the file even when the motors disagree, which turns every
  joint angle into a wrong one.
* Before torque goes on, every Goal_Position is set to the present position (a stale goal makes the arm lunge).
* Every command is rate-limited per joint (deg/s), and `stop()` can interrupt any move.
"""

from __future__ import annotations

import glob
import json
import logging
import threading
import time
from pathlib import Path

import numpy as np

from brick_sorter import JOINTS
from brick_sorter import calib

log = logging.getLogger(__name__)
FOLLOWER_MIN_VOLTS = 9.0  # follower servos run on 12 V, the leader's on 5 V


class CalibrationMismatch(RuntimeError):
    pass


def step_toward(cur: np.ndarray, target: np.ndarray, max_step: float) -> np.ndarray:
    return cur + np.clip(target - cur, -max_step, max_step)


def serial_ports() -> list[str]:
    return sorted(set(glob.glob("/dev/serial/by-id/*") or glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*")))


def supply_volts(port: str) -> float | None:
    """Servo 1's supply voltage on `port` (read-only), or None if no SO-101 servo answers."""
    import warnings

    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bus = FeetechMotorsBus(port, {"m1": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100)})
    try:
        bus.connect(handshake=False)
        return bus.read("Present_Voltage", "m1", normalize=False) / 10.0
    except Exception:  # noqa: BLE001
        return None
    finally:
        try:
            bus.disconnect(disable_torque=False)
        except Exception:  # noqa: BLE001
            pass


def find_follower() -> tuple[str | None, dict[str, float | None]]:
    volts = {p: supply_volts(p) for p in serial_ports()}
    followers = [p for p, v in volts.items() if v is not None and v >= FOLLOWER_MIN_VOLTS]
    if len(followers) > 1:
        raise RuntimeError(f"several 12 V arms {volts}; set arm.port")
    return (followers[0] if followers else None), volts


class Poses:
    """Named joint poses taught by hand, stored next to the calibration (poses.json)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.data: dict[str, dict[str, float]] = json.loads(self.path.read_text()) if self.path.exists() else {}

    def get(self, name: str) -> np.ndarray:
        if name not in self.data:
            raise KeyError(f"no taught pose '{name}' in {self.path}; run: brick-sorter teach {name}")
        return np.array([self.data[name][j] for j in JOINTS], dtype=np.float32)

    def has(self, name: str) -> bool:
        return name in self.data

    def save(self, name: str, pose: np.ndarray) -> None:
        self.data[name] = {j: round(float(v), 2) for j, v in zip(JOINTS, pose)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2) + "\n")


class Arm:
    """Common motion helpers on top of read_state() / _write(cmd)."""

    def __init__(self, fps: int = 30, max_speed: float = 40.0, move_speed: float = 50.0):
        self.fps, self.max_speed, self.move_speed = fps, max_speed, move_speed
        self.cmd: np.ndarray | None = None  # last commanded target
        self.abort = threading.Event()
        self.lock = threading.RLock()

    # --- hardware hooks
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def read_state(self) -> np.ndarray: raise NotImplementedError
    def _write(self, cmd: np.ndarray) -> None: raise NotImplementedError
    def torque(self, on: bool) -> None: ...

    # --- motion
    def send(self, target: np.ndarray, max_speed: float | None = None) -> np.ndarray:
        """One rate-limited step toward `target` (call once per control tick)."""
        with self.lock:
            if self.cmd is None:
                self.cmd = self.read_state()
            self.cmd = step_toward(self.cmd, np.asarray(target, np.float32), (max_speed or self.max_speed) / self.fps)
            self._write(self.cmd)
            return self.cmd

    def move_to(self, target: np.ndarray, speed: float | None = None, tol: float = 0.5) -> bool:
        """Ramp to `target`; returns False if aborted."""
        target = np.asarray(target, np.float32)
        with self.lock:
            if self.cmd is None:
                self.cmd = self.read_state()
        while float(np.max(np.abs(target - self.cmd))) > tol:
            if self.abort.is_set():
                return False
            t = time.perf_counter()
            self.send(target, speed or self.move_speed)
            time.sleep(max(0.0, 1.0 / self.fps - (time.perf_counter() - t)))
        return True

    def hold(self) -> None:
        """Command the present position (stop where we are)."""
        with self.lock:
            self.cmd = self.read_state()
            self._write(self.cmd)


class SO101(Arm):
    def __init__(self, port: str, arm_id: str, calibration_dir: str | Path, fps=30, max_speed=40.0, move_speed=50.0,
                 write_calibration: bool = False):
        super().__init__(fps, max_speed, move_speed)
        self.port, self.arm_id, self.cal_dir = port, arm_id, Path(calibration_dir)
        self.write_calibration = write_calibration
        self.robot = None

    def connect(self) -> None:
        from lerobot.robots import make_robot_from_config
        from lerobot.robots.so_follower import SO101FollowerConfig

        cal_file = self.cal_dir / f"{self.arm_id}.json"
        if not cal_file.exists():
            raise FileNotFoundError(f"{cal_file} missing: copy your follower calibration there")
        port = self.port
        if port == "auto":
            port, volts = find_follower()
            log.info("follower by supply voltage: %s %s", port, volts)
            if not port:
                raise RuntimeError(f"no follower found (servo voltages {volts}); check USB and the 12 V supply")
        self.robot = make_robot_from_config(SO101FollowerConfig(
            port=port, id=self.arm_id, calibration_dir=self.cal_dir, disable_torque_on_disconnect=False))
        bus = self.robot.bus
        bus.connect()  # handshake: all 6 servos must answer
        file_cal = calib.load(cal_file)
        diffs = calib.compare(file_cal, calib.read_motor_calibration(bus))
        if diffs:
            if not self.write_calibration:
                bus.disconnect(disable_torque=False)
                raise CalibrationMismatch(
                    "the motors hold a different calibration than " + str(cal_file) + ":\n  " + "\n  ".join(diffs)
                    + "\nRun `brick-sorter calib` to inspect, or pass --write-calibration to write the file to the"
                      " motors (only if the file is the right one for this arm).")
            log.warning("writing %s to the motors (%d differences)", cal_file.name, len(diffs))
            bus.disable_torque()
            bus.write_calibration(self.robot.calibration)
        self._hold_present_goal()
        self.robot.configure()  # LeRobot enables torque at the end
        self.cmd = self.read_state()
        log.info("follower connected on %s, state %s", port, np.round(self.cmd, 1))

    def _hold_present_goal(self) -> None:
        bus = self.robot.bus
        bus.disable_torque()
        present = bus.sync_read("Present_Position", normalize=False, num_retry=5)
        target = {m: int(p) for m, p in present.items()}
        for _ in range(5):
            bus.sync_write("Goal_Position", target, normalize=False)
            goal = bus.sync_read("Goal_Position", normalize=False, num_retry=5)
            if all(abs(int(goal[m]) - target[m]) <= 2 for m in target):
                return
        raise RuntimeError("could not set goal = present; refusing to enable torque")

    def read_state(self) -> np.ndarray:
        obs = self.robot.bus.sync_read("Present_Position", num_retry=3)
        return np.array([obs[j.removesuffix(".pos")] for j in JOINTS], dtype=np.float32)

    def _write(self, cmd: np.ndarray) -> None:
        self.robot.send_action({j: float(v) for j, v in zip(JOINTS, cmd)})

    def torque(self, on: bool) -> None:
        with self.lock:
            if on:
                self._hold_present_goal()  # the goal write switches torque on
                self.robot.bus.enable_torque()
                self.cmd = self.read_state()
            else:
                self.robot.bus.disable_torque()

    def disconnect(self) -> None:
        if self.robot is not None:
            try:
                self.robot.bus.disconnect(disable_torque=False)  # keep holding: never drop the arm
            except Exception as e:  # noqa: BLE001
                log.warning("disconnect: %s", e)


class FakeArm(Arm):
    """Kinematics-free simulation: the state follows commands instantly; a shared Scene tracks the brick."""

    def __init__(self, scene=None, fps=30, max_speed=40.0, move_speed=50.0, speedup: float = 1.0):
        super().__init__(fps, max_speed * speedup, move_speed * speedup)
        self.scene = scene
        self.state = np.array([0.0, -100.0, 95.0, 5.0, 0.0, 1.0], np.float32)

    def read_state(self) -> np.ndarray:
        return self.state.copy()

    def _write(self, cmd: np.ndarray) -> None:
        self.state = np.asarray(cmd, np.float32).copy()
        if self.scene is not None and not self.scene.has_brick() and self.state[5] < 15.0:
            self.state[5] = min(self.state[5], 1.0)  # closes on nothing: reads fully closed, never "holding"
        if self.scene is not None:
            self.scene.on_arm(self.state)


def make_arm(cfg: dict, scene=None, write_calibration: bool = False, speedup: float = 1.0) -> Arm:
    a = cfg["arm"]
    if a["port"] == "fake":
        return FakeArm(scene, a["fps"], a["max_speed"], a["move_speed"], speedup)
    return SO101(a["port"], a["id"], a["calibration_dir"], a["fps"], a["max_speed"], a["move_speed"],
                 write_calibration)

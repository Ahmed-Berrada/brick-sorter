"""Calibration maths and checks (no hardware needed for the maths).

LeRobot SO-101 conventions (lerobot 0.7, use_degrees=True):
  homed = raw_encoder - homing_offset                         (what Present_Position reads)
  body joint deg = (homed - (range_min + range_max) / 2) * 360 / 4095
  gripper       = (homed - range_min) / (range_max - range_min) * 100

For the same physical pose (same raw encoder value) the reading under calibration A minus the reading under B is
a constant for body joints: (hB - hA + midB - midA) * 360/4095. This is only meaningful if the servo horns were
not re-mounted between the two calibrations (same raw value for the same pose). Checked against a recorded
recalibration (2026-11-14): reproduces 20.35 / -7.96 / 0.18 / 0.04 exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

BODY = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
MOTORS = BODY + ["gripper"]
K = 360.0 / 4095.0


def load(path: str | Path) -> dict:
    cal = json.loads(Path(path).read_text())
    missing = [m for m in MOTORS if m not in cal]
    if missing:
        raise ValueError(f"{path}: missing motors {missing}")
    return cal


def wrap180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def offset_deg(current: dict, reference: dict) -> dict[str, float]:
    """Body-joint reading under `current` minus reading under `reference`, same physical pose (wrapped)."""
    out = {}
    for m in BODY:
        a, b = current[m], reference[m]
        mid_a = (a["range_min"] + a["range_max"]) / 2
        mid_b = (b["range_min"] + b["range_max"]) / 2
        out[m] = round(wrap180((b["homing_offset"] - a["homing_offset"] + mid_b - mid_a) * K), 2)
    return out


def compare(file_cal: dict, motor_cal: dict, tol: int = 2) -> list[str]:
    """Differences between a calibration file and what the motors hold (empty list = match)."""
    diffs = []
    for m in MOTORS:
        for key in ("homing_offset", "range_min", "range_max"):
            f, d = int(file_cal[m][key]), int(motor_cal[m][key])
            if abs(f - d) > tol:
                diffs.append(f"{m}.{key}: file {f} vs motor {d}")
    return diffs


def problems(cal: dict) -> list[str]:
    """Sanity checks on one calibration."""
    out = []
    for m in MOTORS:
        c = cal[m]
        span = c["range_max"] - c["range_min"]
        if m != "wrist_roll" and span < 300:
            out.append(f"{m}: range only {span} ticks (joint not moved through its travel?)")
        if c["range_min"] < 0 or c["range_max"] > 4095:
            out.append(f"{m}: range outside 0..4095")
    return out


def read_motor_calibration(bus) -> dict:
    """{motor: {homing_offset, range_min, range_max}} straight from the servos' EEPROM (read-only)."""
    cal = bus.read_calibration()
    return {m: {"id": c.id, "drive_mode": c.drive_mode, "homing_offset": int(c.homing_offset),
                "range_min": int(c.range_min), "range_max": int(c.range_max)} for m, c in cal.items()}

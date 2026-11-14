"""brick-sorter: SO-101 pick -> inspect (Anomalib + classifier) -> decide -> place, on Intel Core Ultra."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOINTS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
CAMERAS = ["wrist", "top", "inspect"]

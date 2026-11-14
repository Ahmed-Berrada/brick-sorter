"""Shared state for dry runs: which brick is on the table, whether the arm holds it, what the cameras see."""

from __future__ import annotations

import threading

import cv2
import numpy as np

COLOURS = {"yellow": (0, 220, 255), "orange": (0, 140, 255), "purple": (200, 60, 150)}


class Scene:
    """Bricks are a list of marks: plain | circle | cross | unclear (never readable) | faint (readable on 2nd look)."""

    def __init__(self, bricks: list[str], rules: dict | None = None):
        self.bricks = list(bricks)
        self.rules = rules or {}
        self.index = 0
        self.held = False
        self.reinspections = 0
        self.placed: list[tuple[str, str]] = []  # (mark, bowl)
        self.lock = threading.Lock()

    def current_mark(self) -> str:
        with self.lock:
            return self.bricks[self.index] if self.index < len(self.bricks) else "none"

    def has_brick(self) -> bool:
        return self.index < len(self.bricks)

    def on_arm(self, state: np.ndarray) -> None:
        with self.lock:
            g = float(state[5])
            if not self.held and 3.0 <= g <= 15.0 and self.index < len(self.bricks):
                self.held = True
            elif self.held and g > 25.0:
                self.held = False  # released: the orchestrator calls place_done with the bowl

    def place_done(self, bowl: str) -> None:
        with self.lock:
            if self.index < len(self.bricks):
                self.placed.append((self.bricks[self.index], bowl))
                self.index += 1
                self.reinspections = 0

    def paint(self, camera: str, img: np.ndarray) -> None:
        h, w = img.shape[:2]
        mark = self.current_mark()
        if mark == "none":
            return
        if camera == "inspect" and self.held:
            cx, cy = 320, 240
            cv2.rectangle(img, (cx - 110, cy - 110), (cx + 110, cy + 110), (230, 230, 230), -1)
            if mark in ("circle", "faint"):
                cv2.circle(img, (cx, cy), 45, (30, 30, 30), 6 if mark == "circle" else 2)
            elif mark in ("cross", "unclear"):
                t = 6 if mark == "cross" else 1
                cv2.line(img, (cx - 45, cy - 45), (cx + 45, cy + 45), (30, 30, 30), t)
                cv2.line(img, (cx - 45, cy + 45), (cx + 45, cy - 45), (30, 30, 30), t)
        elif camera == "top" and not self.held:
            cv2.rectangle(img, (w // 2 - 30, h // 2 - 30), (w // 2 + 30, h // 2 + 30), (230, 230, 230), -1)
        if camera == "top":
            for i, (name, col) in enumerate(COLOURS.items()):
                cv2.circle(img, (90 + i * 230, h - 70), 50, col, 4)
                n = sum(1 for _, b in self.placed if b == name)
                cv2.putText(img, f"{name} {n}", (50 + i * 230, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

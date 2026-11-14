"""Camera threads: each keeps its latest BGR frame (and a JPEG for the web page) and reconnects on failure."""

from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np

log = logging.getLogger(__name__)


class Camera:
    """Base: subclasses implement _open() -> handle and _read(handle) -> BGR frame or None, and _close(handle)."""

    def __init__(self, name: str, width: int, height: int, fps: int):
        self.name, self.width, self.height, self.fps = name, width, height, fps
        self.frame: np.ndarray | None = None
        self.stamp = 0.0
        self.seq = 0
        self.error: str | None = None
        self.measured_fps = 0.0
        self._jpeg: tuple[int, bytes] | None = None
        self._lock = threading.Condition()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"cam-{name}", daemon=True)

    # --- subclass hooks
    def _open(self):
        raise NotImplementedError

    def _read(self, handle) -> np.ndarray | None:
        raise NotImplementedError

    def _close(self, handle) -> None:
        pass

    # --- public
    def start(self) -> "Camera":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def latest(self) -> tuple[np.ndarray | None, float]:
        with self._lock:
            return self.frame, self.stamp

    def wait_new(self, after_seq: int, timeout: float = 1.0) -> int:
        with self._lock:
            self._lock.wait_for(lambda: self.seq > after_seq, timeout)
            return self.seq

    def jpeg(self) -> bytes | None:
        with self._lock:
            if self.frame is None:
                return None
            if self._jpeg is None or self._jpeg[0] != self.seq:
                ok, buf = cv2.imencode(".jpg", self.frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                self._jpeg = (self.seq, buf.tobytes()) if ok else None
            return self._jpeg[1] if self._jpeg else None

    def info(self) -> dict:
        return {"name": self.name, "fps": round(self.measured_fps, 1), "error": self.error,
                "ok": self.frame is not None and time.time() - self.stamp < 2.0}

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                handle = self._open()
            except Exception as e:  # noqa: BLE001  unplugged / busy: retry
                self.error = f"open failed: {e}"
                time.sleep(2.0)
                continue
            self.error = None
            t0, n = time.perf_counter(), 0
            try:
                while not self._stop.is_set():
                    frame = self._read(handle)
                    if frame is None:
                        self.error = "no frame; reopening"
                        break
                    n += 1
                    now = time.perf_counter()
                    if now - t0 >= 1.0:
                        self.measured_fps, t0, n = n / (now - t0), now, 0
                    with self._lock:
                        self.frame, self.stamp = frame, time.time()
                        self.seq += 1
                        self._lock.notify_all()
            except Exception as e:  # noqa: BLE001
                self.error = str(e)
                log.warning("camera %s: %s", self.name, e)
            finally:
                try:
                    self._close(handle)
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(0.5)


class UVCCamera(Camera):
    def __init__(self, name, source, width, height, fps, fourcc: str | None = "MJPG"):
        super().__init__(name, width, height, fps)
        self.source = int(source) if str(source).isdigit() else str(source)
        self.fourcc = fourcc

    def _open(self):
        cap = cv2.VideoCapture(self.source, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {self.source}")
        if self.fourcc:  # set before the size: uncompressed YUYV times out on WSL / shared USB 2 hubs
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _read(self, cap):
        ok, frame = cap.read()
        if not ok:
            return None
        if frame.shape[:2] != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height))
        return frame

    def _close(self, cap):
        cap.release()


class RealSenseCamera(Camera):
    def __init__(self, name, serial, width, height, fps):
        super().__init__(name, width, height, fps)
        self.serial = str(serial)

    def _open(self):
        import pyrealsense2 as rs

        cfg = rs.config()
        cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        pipe = rs.pipeline()
        pipe.start(cfg)
        return pipe

    def _read(self, pipe):
        fs = pipe.wait_for_frames(3000)
        c = fs.get_color_frame()
        return np.asanyarray(c.get_data()).copy() if c else None

    def _close(self, pipe):
        pipe.stop()


class FakeCamera(Camera):
    """Synthetic scene for dry runs. `scene` (shared with the fake arm) can paint a brick with a mark."""

    def __init__(self, name, width, height, fps, scene=None):
        super().__init__(name, width, height, fps)
        self.scene = scene

    def _open(self):
        return None

    def _read(self, _):
        time.sleep(1.0 / self.fps)
        img = np.full((self.height, self.width, 3), 70, np.uint8)
        cv2.putText(img, self.name, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        if self.scene is not None:
            self.scene.paint(self.name, img)
        return img


def make_cameras(cfg: dict, scene=None) -> dict[str, Camera]:
    c = cfg["cameras"]
    w, h, fps = c["width"], c["height"], c["fps"]
    out: dict[str, Camera] = {}
    for name in ("wrist", "top", "inspect"):
        spec = c[name]
        kind = spec["kind"]
        if kind == "uvc":
            out[name] = UVCCamera(name, spec["source"], w, h, fps, spec.get("fourcc", "MJPG") or None)
        elif kind == "realsense":
            out[name] = RealSenseCamera(name, spec["source"], w, h, fps)
        elif kind == "fake":
            out[name] = FakeCamera(name, w, h, fps, scene)
        else:
            raise ValueError(f"camera {name}: unknown kind {kind!r}")
    return out


def wait_ready(cams: dict[str, Camera], timeout: float = 15.0) -> None:
    t0 = time.time()
    while any(c.frame is None for c in cams.values()):
        if time.time() - t0 > timeout:
            missing = {n: c.error for n, c in cams.items() if c.frame is None}
            raise RuntimeError(f"no frames from {missing}")
        time.sleep(0.05)

"""Find, test and assign the station's cameras (any models): `brick-sorter cameras`.

1. Lists every camera: RealSense devices by serial (via pyrealsense2) and every other USB/UVC camera by its
   stable /dev/v4l/by-id path (falls back to by-path, then /dev/videoN). RealSense UVC nodes are not listed twice.
2. Tests each one at the configured size/fps: which pixel formats work (MJPG / YUYV / default), measured fps,
   actual resolution, and saves a snapshot to runs/cameras/<n>.jpg so you can see what it points at.
3. Optional live preview page of all cameras (--preview) to tell them apart from another computer.
4. Writes the chosen wrist / top / inspect mapping into config/local.toml (a managed [cameras] block).
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

BLOCK_START = "# >>> cameras: written by `brick-sorter cameras` (re-run it to change)"
BLOCK_END = "# <<< cameras"


@dataclass
class Found:
    kind: str  # "uvc" | "realsense"
    source: str  # by-id path / serial
    name: str
    node: str = ""  # /dev/videoN for uvc
    usb: str = ""  # RealSense USB type ("3.2" / "2.1") or the uvc bus path
    formats: list[str] = field(default_factory=list)  # from v4l2-ctl, if available
    result: dict = field(default_factory=dict)  # filled by test()

    @property
    def label(self) -> str:
        return f"{self.name} [{self.kind}: {self.source}]"


# ------------------------------------------------------------------------------------------------ discovery
def _stable_path(node: str) -> str:
    """Prefer /dev/v4l/by-id (survives re-plugging into another port), then by-path, else the node itself."""
    for pattern in ("/dev/v4l/by-id/*", "/dev/v4l/by-path/*"):
        for link in sorted(glob.glob(pattern)):
            if os.path.realpath(link) == os.path.realpath(node):
                return link
    return node


def _v4l2_formats(node: str) -> list[str]:
    if not shutil.which("v4l2-ctl"):
        return []
    try:
        out = subprocess.run(["v4l2-ctl", "-d", node, "--list-formats"], capture_output=True, text=True,
                             timeout=5).stdout
    except Exception:  # noqa: BLE001
        return []
    return re.findall(r"'(\w{3,4})'", out)


def list_realsense() -> list[Found]:
    try:
        import pyrealsense2 as rs
    except ImportError:
        return []
    out = []
    for d in rs.context().devices:
        info = lambda k: d.get_info(k) if d.supports(k) else ""  # noqa: E731
        out.append(Found("realsense", info(rs.camera_info.serial_number), info(rs.camera_info.name),
                         usb=info(rs.camera_info.usb_type_descriptor)))
    return out


def list_uvc(skip_realsense: bool = True) -> list[Found]:
    out = []
    for sysdir in sorted(glob.glob("/sys/class/video4linux/video*"), key=lambda p: int(re.sub(r"\D", "", p) or 0)):
        node = "/dev/" + os.path.basename(sysdir)
        try:
            name = Path(sysdir, "name").read_text().strip()
            index = int(Path(sysdir, "index").read_text().strip() or 0)
        except OSError:
            continue
        if index != 0:  # index 1+ are metadata nodes of the same camera
            continue
        if skip_realsense and "realsense" in name.lower():
            continue
        formats = _v4l2_formats(node)
        if formats == [] and shutil.which("v4l2-ctl"):
            continue  # no capture formats: not a video capture node
        out.append(Found("uvc", _stable_path(node), name, node=node, formats=formats))
    return out


def list_cameras() -> list[Found]:
    """RealSense via its SDK (by serial) when pyrealsense2 is installed, every other camera as UVC."""
    try:
        import pyrealsense2  # noqa: F401

        have_sdk = True
    except ImportError:
        have_sdk = False
    return list_realsense() + list_uvc(skip_realsense=have_sdk)


# ------------------------------------------------------------------------------------------------ testing
def _measure(read, seconds: float) -> tuple[int, float, np.ndarray | None]:
    n, last, t0 = 0, None, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        f = read()
        if f is None:
            break
        n, last = n + 1, f
    return n, n / max(1e-6, time.perf_counter() - t0), last


def test_uvc(cam: Found, width: int, height: int, fps: int, seconds: float = 2.0) -> dict:
    tried = {}
    for fourcc in ("MJPG", "YUYV", None):
        cap = cv2.VideoCapture(cam.node or cam.source, cv2.CAP_V4L2)
        if not cap.isOpened():
            return {"ok": False, "error": "cannot open (busy? another program using it?)", "tried": tried}
        try:
            if fourcc:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            got = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode(errors="replace")

            def read():
                ok, f = cap.read()
                return f if ok else None

            n, measured, frame = _measure(read, seconds)
        finally:
            cap.release()
        key = fourcc or f"default({got})"
        tried[key] = round(measured, 1)
        if n >= 3 and frame is not None:
            return {"ok": True, "fourcc": fourcc or "", "format": got, "fps": round(measured, 1),
                    "size": f"{frame.shape[1]}x{frame.shape[0]}", "tried": tried, "frame": frame}
    return {"ok": False, "error": "no frames in any format", "tried": tried}


def test_realsense(cam: Found, width: int, height: int, fps: int, seconds: float = 2.0) -> dict:
    import pyrealsense2 as rs

    tried = {}
    for f in dict.fromkeys((fps, 30, 15)):
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(cam.source)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, f)
        try:
            pipe.start(cfg)
        except Exception as e:  # noqa: BLE001
            tried[f"{f}fps"] = str(e)[:60]
            continue
        try:
            def read():
                fs = pipe.wait_for_frames(3000)
                c = fs.get_color_frame()
                return np.asanyarray(c.get_data()).copy() if c else None

            n, measured, frame = _measure(read, seconds)
        except Exception as e:  # noqa: BLE001
            tried[f"{f}fps"] = str(e)[:60]
            continue
        finally:
            try:
                pipe.stop()
            except Exception:  # noqa: BLE001
                pass
        tried[f"{f}fps"] = round(measured, 1)
        if n >= 3:
            return {"ok": True, "fps": round(measured, 1), "requested_fps": f, "size": f"{frame.shape[1]}x{frame.shape[0]}",
                    "tried": tried, "frame": frame}
    return {"ok": False, "error": "no colour stream at this size", "tried": tried}


def test(cam: Found, width: int, height: int, fps: int, snap_dir: Path, n: int) -> dict:
    r = (test_realsense if cam.kind == "realsense" else test_uvc)(cam, width, height, fps)
    frame = r.pop("frame", None)
    if frame is not None:
        snap_dir.mkdir(parents=True, exist_ok=True)
        img = frame.copy()
        cv2.putText(img, f"#{n} {cam.name[:40]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        path = snap_dir / f"{n}.jpg"
        cv2.imwrite(str(path), img)
        r["snapshot"] = str(path)
        r["brightness"] = round(float(frame.mean()), 1)
    cam.result = r
    return r


def warnings_for(cam: Found, width: int, height: int, fps: int) -> list[str]:
    r, w = cam.result, []
    if not r.get("ok"):
        return [r.get("error", "failed")]
    if r.get("size") != f"{width}x{height}":
        w.append(f"delivers {r['size']}, not {width}x{height} (frames are resized; the policy expects what it was "
                 "trained on)")
    if r.get("fps", 0) < fps * 0.8:
        w.append(f"only {r['fps']} fps (wanted {fps}): try another USB port / fewer cameras on one hub")
    if cam.kind == "realsense" and cam.usb.startswith("2"):
        w.append(f"connected over USB {cam.usb}: use a USB 3 port/cable")
    if cam.kind == "uvc" and cam.source.startswith("/dev/video"):
        w.append("no stable /dev/v4l path: the number may change after re-plugging")
    if r.get("brightness", 100) < 15:
        w.append("image almost black (lens covered? exposure?)")
    return w


# ------------------------------------------------------------------------------------------------ config
def spec_for(cam: Found) -> dict:
    if cam.kind == "realsense":
        return {"kind": "realsense", "source": cam.source}
    return {"kind": "uvc", "source": cam.source, "fourcc": cam.result.get("fourcc", "MJPG") or ""}


def _toml_inline(d: dict) -> str:
    return "{ " + ", ".join(f'{k} = "{v}"' for k, v in d.items()) + " }"


def write_assignment(local_path: Path, mapping: dict[str, dict], extra: dict | None = None) -> None:
    """Replace the managed [cameras] block in local_path (appended at the end, so it can't swallow other keys)."""
    text = local_path.read_text() if local_path.exists() else ""
    text = re.sub(re.escape(BLOCK_START) + r".*?" + re.escape(BLOCK_END) + r"\n?", "", text, flags=re.S)
    if re.search(r"^\s*\[cameras\]", text, re.M):
        raise RuntimeError(f"{local_path} already has its own [cameras] section; remove it first")
    lines = [BLOCK_START, "[cameras]"]
    for k, v in (extra or {}).items():
        lines.append(f"{k} = {v}")
    for role in ("wrist", "top", "inspect"):
        lines.append(f"{role} = {_toml_inline(mapping[role])}")
    lines.append(BLOCK_END)
    text = text.rstrip() + ("\n\n" if text.strip() else "") + "\n".join(lines) + "\n"
    import tomllib

    tomllib.loads(text)  # never write a file we cannot read back
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(text)


# ------------------------------------------------------------------------------------------------ preview
def preview(cams: list[Found], width: int, height: int, fps: int, host: str, port: int) -> None:
    """Live grid of every working camera, to tell them apart (wave a hand in front of one)."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from brick_sorter.cameras import RealSenseCamera, UVCCamera

    streams = []
    for i, c in enumerate(cams, 1):
        if not c.result.get("ok"):
            continue
        s = (RealSenseCamera(f"#{i}", c.source, width, height, c.result.get("requested_fps", fps)) if c.kind == "realsense"
             else UVCCamera(f"#{i}", c.node or c.source, width, height, fps, c.result.get("fourcc") or None))
        streams.append((i, c, s.start()))
    tiles = "".join(f'<figure><img src="/s/{i}"><figcaption><b>#{i}</b> {c.name}<br><small>{c.kind}: {c.source}'
                    f'</small></figcaption></figure>' for i, c, _ in streams)
    page = ("<!doctype html><meta name=viewport content='width=device-width'><title>Cameras</title>"
            "<style>body{background:#111;color:#eee;font:14px system-ui;margin:16px}div{display:grid;"
            "grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}img{width:100%;background:#000}"
            f"small{{color:#999;word-break:break-all}}</style><h2>Cameras</h2><div>{tiles}</div>").encode()
    by_id = {i: s for i, _, s in streams}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):  # noqa: N802
            if self.path.startswith("/s/"):
                s = by_id.get(int(self.path[3:] or 0))
                if not s:
                    return self.send_error(404)
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
                self.end_headers()
                seq = -1
                try:
                    while True:
                        seq = s.wait_new(seq, 2.0)
                        j = s.jpeg()
                        if j:
                            self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + j + b"\r\n")
                        time.sleep(0.1)
                except (BrokenPipeError, ConnectionResetError):
                    return
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(page)

    srv = ThreadingHTTPServer((host, port), H)
    print(f"\nlive preview: http://{'localhost' if host == '0.0.0.0' else host}:{port}  (Ctrl+C when done)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        for _, _, s in streams:
            s.stop()
        time.sleep(0.6)  # let the threads release the devices before they are reopened

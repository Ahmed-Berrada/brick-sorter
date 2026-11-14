import tomllib

import pytest

from brick_sorter import camera_setup as cs
from brick_sorter import config as C


def test_write_assignment_creates_and_replaces_block(tmp_path):
    local = tmp_path / "local.toml"
    local.write_text('[arm]\nport = "/dev/so101-follower"\n')
    m = {"wrist": {"kind": "uvc", "source": "/dev/v4l/by-id/cam-a", "fourcc": "MJPG"},
         "top": {"kind": "realsense", "source": "111"},
         "inspect": {"kind": "uvc", "source": "/dev/v4l/by-id/cam-b", "fourcc": ""}}
    cs.write_assignment(local, m)
    m["top"] = {"kind": "realsense", "source": "222"}
    cs.write_assignment(local, m)  # re-run replaces, never duplicates
    text = local.read_text()
    assert text.count(cs.BLOCK_START) == 1
    d = tomllib.loads(text)
    assert d["arm"]["port"] == "/dev/so101-follower"
    assert d["cameras"]["top"]["source"] == "222"
    cfg = C.load(local)  # merges over station.toml: size/fps defaults survive
    assert cfg["cameras"]["width"] == 640 and cfg["cameras"]["wrist"]["source"] == "/dev/v4l/by-id/cam-a"


def test_write_assignment_refuses_hand_written_cameras_section(tmp_path):
    local = tmp_path / "local.toml"
    local.write_text('[cameras]\nfps = 15\n')
    with pytest.raises(RuntimeError, match="own \\[cameras\\]"):
        cs.write_assignment(local, {r: {"kind": "uvc", "source": r} for r in ("wrist", "top", "inspect")})


def test_warnings():
    c = cs.Found("realsense", "123", "D435", usb="2.1",
                 result={"ok": True, "size": "640x480", "fps": 14.0, "brightness": 80})
    w = cs.warnings_for(c, 640, 480, 30)
    assert any("USB 2.1" in x for x in w) and any("fps" in x for x in w)
    u = cs.Found("uvc", "/dev/video4", "webcam", result={"ok": True, "size": "1280x720", "fps": 30, "brightness": 5})
    w = cs.warnings_for(u, 640, 480, 30)
    assert any("stable" in x for x in w) and any("1280x720" in x for x in w) and any("black" in x for x in w)


def test_spec_for_keeps_working_format():
    u = cs.Found("uvc", "/dev/v4l/by-id/x", "cam", result={"ok": True, "fourcc": "YUYV"})
    assert cs.spec_for(u) == {"kind": "uvc", "source": "/dev/v4l/by-id/x", "fourcc": "YUYV"}

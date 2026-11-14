"""End-to-end loop with the simulated arm, cameras, policy and inspector, plus the web API."""

import json
import time
import urllib.request

import numpy as np
import pytest

from brick_sorter import config as C
from brick_sorter.arm import FakeArm, Poses
from brick_sorter.cameras import make_cameras, wait_ready
from brick_sorter.inspector import FakeInspector
from brick_sorter.orchestrator import Orchestrator
from brick_sorter.policy import FakePolicy
from brick_sorter.sim import Scene


def make(tmp_path, bricks, speed=8.0):
    cfg = C.load(None, C.fake_overrides())
    cfg["arm"]["calibration_dir"] = str(tmp_path)
    cfg["log"]["dir"] = str(tmp_path / "runs")
    cfg["inspect"]["settle_s"] = 0.0
    cfg["policy"]["pick_timeout"] = 10.0
    poses = Poses(tmp_path / "poses.json")
    poses.save("home", np.array([0, -100, 95, 5, 0, 1], np.float32))
    for i, b in enumerate(("yellow", "orange", "purple")):
        poses.save(b, np.array([-40 + 40 * i, -30, 50, 40, 0, 6], np.float32))
    scene = Scene(bricks, cfg["rules"])
    cams = {n: c.start() for n, c in make_cameras(cfg, scene).items()}
    wait_ready(cams)
    arm = FakeArm(scene, 30, cfg["arm"]["max_speed"], cfg["arm"]["move_speed"], speed)
    return Orchestrator(cfg, arm, cams, FakePolicy(latency_ms=1), FakeInspector(scene), scene), scene, cams


def test_full_loop_sorts_every_brick(tmp_path):
    orch, scene, cams = make(tmp_path, ["plain", "circle", "cross", "faint", "unclear"])
    orch.start(max_bricks=5)
    orch.join(120)
    for c in cams.values():
        c.stop()
    assert orch.error is None, orch.error
    assert scene.placed == [("plain", "yellow"), ("circle", "orange"), ("cross", "purple"),
                            ("faint", "orange"), ("unclear", "purple")]
    assert orch.counts == {"yellow": 1, "orange": 2, "purple": 2}
    rows = [json.loads(line) for line in (orch.run_dir / "bricks.jsonl").read_text().splitlines()]
    assert [r["attempts"] for r in rows] == [1, 1, 1, 2, 2]
    assert rows[-1]["decision"]["forced"] is True
    assert (orch.run_dir / "summary.json").exists()


def test_refuses_to_start_without_taught_poses(tmp_path):
    orch, _, cams = make(tmp_path, ["plain"])
    orch.poses.data.pop("orange")
    with pytest.raises(RuntimeError, match="teach"):
        orch.start()
    for c in cams.values():
        c.stop()


def test_stop_interrupts_and_holds(tmp_path):
    orch, scene, cams = make(tmp_path, ["plain"] * 10, speed=1.0)
    orch.start()
    time.sleep(1.0)
    orch.stop()
    orch.join(10)
    for c in cams.values():
        c.stop()
    assert orch.state == "stopped"
    assert len(scene.placed) < 10


def test_web_api(tmp_path):
    from brick_sorter.web import serve

    orch, _, cams = make(tmp_path, ["cross"])
    orch.cfg["policy"]["pick_timeout"] = 2.0  # table empty after one brick: two misses end the loop
    srv = serve(orch, "127.0.0.1", 0)
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        assert b"Brick Sorter" in urllib.request.urlopen(base + "/").read()
        urllib.request.urlopen(urllib.request.Request(base + "/api/start", method="POST")).read()
        orch.join(60)
        s = json.loads(urllib.request.urlopen(base + "/api/status").read())
        assert s["counts"] == {"purple": 1} and s["inspection"]["outcome"] == "cross"
        with urllib.request.urlopen(base + "/stream/inspect") as r:
            assert r.read(7) == b"--frame"
    finally:
        srv.shutdown()
        for c in cams.values():
            c.stop()


def test_pick_only_policy_is_carried_to_the_inspect_pose(tmp_path):
    orch, scene, cams = make(tmp_path, ["circle"])
    orch.cfg["policy"]["present_pose"] = "inspect"
    assert orch.missing_poses() == ["inspect"]
    orch.poses.save("inspect", np.array([-30, -60, 60, 20, 0, 6], np.float32))
    orch.start(max_bricks=1)
    orch.join(60)
    for c in cams.values():
        c.stop()
    msgs = [e["msg"] for e in orch.events]
    assert orch.error is None and scene.placed == [("circle", "orange")]
    i_carry = next(i for i, m in enumerate(msgs) if "carried to the 'inspect' pose" in m)
    i_decide = next(i for i, m in enumerate(msgs) if m.startswith("circle"))
    assert i_carry < i_decide

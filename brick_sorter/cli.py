"""brick-sorter command line.

  brick-sorter sim                  full loop with simulated arm, cameras, policy and inspector (any laptop)
  brick-sorter doctor               check drivers, devices, models, ports, cameras, calibration
  brick-sorter calib                compare the calibration file with the motors and the policy's training calibration
  brick-sorter teach home yellow orange purple    hand-guide the arm and save named poses
  brick-sorter bench                OPTIMIZE table: policy + inspection latency per OpenVINO device (+ PyTorch baseline)
  brick-sorter run                  the real station, control page on http://<host>:8080
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

from brick_sorter import CAMERAS, JOINTS, ROOT
from brick_sorter import config as C

log = logging.getLogger("brick_sorter")


# ---------------------------------------------------------------------------------------------- run / sim
def _serve_until_signal(orch, cfg, autostart: bool, max_bricks: int | None, headless: bool) -> int:
    from brick_sorter.web import serve

    srv = None if headless else serve(orch, cfg["web"]["host"], cfg["web"]["port"])
    done = {"sig": False}

    def on_sig(*_):
        if done["sig"]:
            os._exit(1)
        done["sig"] = True
        orch.stop()

    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)
    if autostart:
        orch.start(max_bricks)
    try:
        while not done["sig"]:
            time.sleep(0.2)
            if headless and orch._thread is not None and not orch._thread.is_alive():
                break
    finally:
        orch.stop()
        orch.join(5)
        if srv:
            srv.shutdown()
    s = orch.snapshot()
    print(json.dumps({"state": s["state"], "counts": s["counts"], "latency": s["latency"], "run_dir": s["run_dir"],
                      "error": s["error"]}, indent=2))
    return 1 if s["error"] else 0


def cmd_sim(a) -> int:
    from brick_sorter.arm import FakeArm, Poses
    from brick_sorter.cameras import make_cameras, wait_ready
    from brick_sorter.inspector import FakeInspector
    from brick_sorter.orchestrator import Orchestrator
    from brick_sorter.policy import FakePolicy
    from brick_sorter.sim import Scene

    import tempfile

    cfg = C.load(a.config, C.fake_overrides())
    sim_dir = Path(tempfile.mkdtemp(prefix="brick-sorter-sim-"))
    cfg["arm"]["calibration_dir"] = str(sim_dir)
    cfg["policy"]["pick_timeout"] = 15.0
    cfg["inspect"]["settle_s"] = 0.1 / a.speed
    poses = Poses(sim_dir / "poses.json")  # simulated poses, never the real arm's
    poses.save("home", np.array([0, -100, 95, 5, 0, 1], np.float32))
    for i, bowl in enumerate(("yellow", "orange", "purple")):
        poses.save(bowl, np.array([-40 + 40 * i, -30, 50, 40, 0, 6], np.float32))
    scene = Scene(a.bricks.split(","), cfg["rules"])
    cams = {n: c.start() for n, c in make_cameras(cfg, scene).items()}
    wait_ready(cams)
    arm = FakeArm(scene, cfg["arm"]["fps"], cfg["arm"]["max_speed"], cfg["arm"]["move_speed"], a.speed)
    orch = Orchestrator(cfg, arm, cams, FakePolicy(), FakeInspector(scene), scene)
    orch.step_mode = a.step
    if a.port:
        cfg["web"]["port"] = a.port
    rc = _serve_until_signal(orch, cfg, autostart=not a.no_start, max_bricks=len(scene.bricks),
                             headless=a.headless)
    print("placed:", scene.placed)
    return rc


def cmd_run(a) -> int:
    from brick_sorter.arm import make_arm
    from brick_sorter.cameras import make_cameras, wait_ready
    from brick_sorter.inspector import make_inspector
    from brick_sorter.orchestrator import Orchestrator
    from brick_sorter.policy import make_policy

    cfg = C.load(a.config)
    if a.port:
        cfg["web"]["port"] = a.port
    log.info("loading the policy (%s on %s)", cfg["policy"]["backend"], cfg["policy"].get("device"))
    policy = make_policy(cfg)
    policy.warmup()
    log.info("loading the inspector")
    inspector = make_inspector(cfg)
    cams = {n: c.start() for n, c in make_cameras(cfg).items()}
    wait_ready(cams)
    arm = make_arm(cfg, write_calibration=False)
    arm.connect()
    try:
        orch = Orchestrator(cfg, arm, cams, policy, inspector)
        orch.step_mode = a.step
        missing = orch.missing_poses()
        if missing:
            log.warning("teach these poses before starting: brick-sorter teach %s", " ".join(missing))
        return _serve_until_signal(orch, cfg, autostart=a.start and not missing, max_bricks=a.max_bricks,
                                   headless=False)
    finally:
        for c in cams.values():
            c.stop()
        arm.disconnect()


# ---------------------------------------------------------------------------------------------- teach
def cmd_teach(a) -> int:
    from brick_sorter.arm import Poses, make_arm

    cfg = C.load(a.config)
    arm = make_arm(cfg)
    arm.connect()
    poses = Poses(Path(cfg["arm"]["calibration_dir"]) / "poses.json")
    hints = {"home": "fold the arm into its rest pose, gripper closed",
             "inspect": "hold a brick in front of the inspect camera, as the policy should present it"}
    try:
        for name in a.names:
            hint = hints.get(name, f"gripper just above the {name} bowl, closed, at brick-holding height")
            input(f"\n[{name}] Support the arm with one hand. Torque goes OFF when you press Enter...")
            arm.torque(False)
            input(f"[{name}] Move it: {hint}. Press Enter to save...")
            pose = arm.read_state()
            poses.save(name, pose)
            print(f"saved {name}: {dict(zip(JOINTS, np.round(pose, 1).tolist()))}")
            arm.torque(True)  # goal = present first, so it holds where you left it
    finally:
        arm.disconnect()
    print(f"poses in {poses.path}: {sorted(poses.data)}")
    return 0


# ---------------------------------------------------------------------------------------------- calib
def cmd_calib(a) -> int:
    from brick_sorter import calib

    cfg = C.load(a.config)
    cal_dir = Path(cfg["arm"]["calibration_dir"])
    file_path = cal_dir / f"{cfg['arm']['id']}.json"
    cur = calib.load(file_path)
    print(f"calibration file: {file_path}")
    for p in calib.problems(cur):
        print("  ! " + p)
    ref_path = Path(a.against) if a.against else ROOT / "calibration" / "reference" / "act_training_follower.json"
    if ref_path.exists():
        ref = calib.load(ref_path)
        off = calib.offset_deg(cur, ref)
        print(f"\nreading under this file minus reading under {ref_path.name} (same pose, deg):")
        for j, v in off.items():
            flag = "" if abs(v) < 10 else "   <- large: horn re-mounted or a different arm; an offset will not fix this"
            print(f"  {j:14s} {v:8.2f}{flag}")
        print("  gripper        (range-normalised; compare open/closed readings instead)")
        if max(abs(v) for v in off.values()) < 10:
            print("\nsmall offsets: put them in [arm] joint_offset in your config to reuse the policy as-is:")
            print("  joint_offset = { " + ", ".join(f"{k} = {v}" for k, v in off.items()) + ", gripper = 0.0 }")
        else:
            print("\nlarge offsets: the policy was trained on a differently calibrated/assembled arm. Re-record demos"
                  " with this calibration and fine-tune, rather than reusing it with an offset.")
    if a.offline:
        return 0

    # ---- motors (read-only unless --write)
    from brick_sorter.arm import SO101, find_follower

    port = cfg["arm"]["port"]
    if port == "auto":
        port, volts = find_follower()
        if not port:
            print(f"\nno follower found to compare with the motors (servo voltages {volts}); use --offline")
            return 2
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.so_follower import SO101FollowerConfig

    robot = make_robot_from_config(SO101FollowerConfig(port=port, id=cfg["arm"]["id"], calibration_dir=cal_dir,
                                                       disable_torque_on_disconnect=False))
    robot.bus.connect()
    try:
        motors = calib.read_motor_calibration(robot.bus)
        diffs = calib.compare(cur, motors)
        print(f"\nmotors on {port}: " + ("MATCH the file" if not diffs else f"{len(diffs)} differences from the file"))
        for d in diffs:
            print("  " + d)
        if ref_path.exists():
            print(f"motors vs {ref_path.name}: "
                  + ("match" if not calib.compare(ref, motors) else "differ"))
        out = cal_dir / "motors_read.json"
        out.write_text(json.dumps(motors, indent=2) + "\n")
        print(f"(what the motors hold was saved to {out})")
        if diffs and a.write:
            ans = input(f"Write {file_path.name} to the motors? The arm goes limp briefly. Support it. [y/N] ")
            if ans.strip().lower() == "y":
                robot.bus.disable_torque()
                robot.bus.write_calibration(robot.calibration)
                print("written; re-run `brick-sorter calib` to confirm")
        return 0 if not diffs else 3
    finally:
        robot.bus.disconnect(disable_torque=False)


# ---------------------------------------------------------------------------------------------- cameras
def cmd_cameras(a) -> int:
    from brick_sorter import camera_setup as cs

    cfg = C.load(a.config)
    w, h, fps = cfg["cameras"]["width"], cfg["cameras"]["height"], cfg["cameras"]["fps"]
    cams = cs.list_cameras()
    if not cams:
        print("no cameras found. Check USB, `ls /dev/video*`, and that you are in the video group.")
        return 1
    snap_dir = ROOT / "runs" / "cameras"
    print(f"found {len(cams)} camera(s); testing each at {w}x{h}@{fps} (close Studio / other camera apps first)\n")
    for i, c in enumerate(cams, 1):
        r = cs.test(c, w, h, fps, snap_dir, i)
        status = (f"OK   {r.get('size')} {r.get('fps')} fps" + (f" {r['format']}" if r.get("format") else "")
                  if r.get("ok") else "FAIL")
        print(f"#{i}  {status:28s} {c.name}")
        print(f"      {c.kind}: {c.source}" + (f"   (USB {c.usb})" if c.usb else ""))
        if c.formats:
            print(f"      formats: {', '.join(c.formats)}")
        print(f"      tried: {r.get('tried')}")
        for warn in cs.warnings_for(c, w, h, fps):
            print(f"      ! {warn}")
        if r.get("snapshot"):
            print(f"      snapshot: {r['snapshot']}")
    current = {role: cfg["cameras"][role] for role in CAMERAS}
    print("\ncurrent config:")
    for role, spec in current.items():
        hit = [i for i, c in enumerate(cams, 1) if str(spec.get("source")) in (c.source, c.node)]
        print(f"  {role:8s} {spec.get('kind')}: {spec.get('source')}  -> {'#' + str(hit[0]) if hit else 'NOT CONNECTED'}")

    if a.preview:
        cs.preview(cams, w, h, fps, cfg["web"]["host"], a.preview_port)

    if a.assign:
        pairs = dict(kv.split("=", 1) for kv in a.assign.split(","))
    elif sys.stdin.isatty() and not a.no_assign:
        print("\nWhich camera is which? Look at the snapshots (or use --preview). Enter a number, blank = keep.")
        pairs = {}
        for role in CAMERAS:
            ans = input(f"  {role:8s} #").strip()
            if ans:
                pairs[role] = ans
    else:
        return 0
    if not pairs:
        print("nothing changed")
        return 0
    mapping = {role: dict(current[role]) for role in CAMERAS}
    for role, num in pairs.items():
        if role not in CAMERAS:
            print(f"unknown role {role!r} (use wrist, top, inspect)")
            return 2
        idx = int(num.lstrip("#")) - 1
        if not 0 <= idx < len(cams) or not cams[idx].result.get("ok"):
            print(f"#{num} is not a working camera")
            return 2
        mapping[role] = cs.spec_for(cams[idx])
    used = [str(m["source"]) for m in mapping.values()]
    if len(set(used)) != len(used):
        print(f"the same camera is assigned twice: {mapping}")
        return 2
    local = Path(a.config) if a.config else ROOT / "config" / "local.toml"
    local = local if local.is_absolute() else ROOT / local
    cs.write_assignment(local, mapping)
    print(f"\nwrote to {local}:")
    for role in CAMERAS:
        print(f"  {role:8s} {mapping[role]}")
    if not a.config:
        print("run with it: scripts/run.sh (uses config/local.toml), or brick-sorter -c config/local.toml ...")
    return 0


# ---------------------------------------------------------------------------------------------- doctor
def cmd_doctor(a) -> int:
    import glob
    import grp
    import platform

    cfg = C.load(a.config)
    bad = 0

    def line(ok, what, note=""):
        nonlocal bad
        bad += 0 if ok in (True, None) else 1
        mark = {True: "OK  ", False: "FAIL", None: "--  "}[ok]
        print(f"[{mark}] {what}" + (f"  ({note})" if note else ""))

    print(f"python {platform.python_version()} on {platform.platform()}")
    try:
        import openvino as ov

        core = ov.Core()
        devs = core.available_devices
        line(True, f"OpenVINO {ov.__version__}", ", ".join(f"{d}: {core.get_property(d, 'FULL_DEVICE_NAME')}" for d in devs))
        for need in ("GPU", "NPU"):
            line(any(d.startswith(need) for d in devs) or None, f"OpenVINO {need}",
                 "found" if any(d.startswith(need) for d in devs) else "not found: scripts/setup_intel.sh installs drivers")
    except Exception as e:  # noqa: BLE001
        line(False, "OpenVINO", str(e))
    line(bool(glob.glob("/dev/dri/renderD*")) or None, "Intel GPU render node /dev/dri/renderD*")
    line(bool(glob.glob("/dev/accel/accel*")) or None, "NPU device /dev/accel/accel*")
    try:
        groups = {grp.getgrgid(g).gr_name for g in os.getgroups()}
        for g in ("dialout", "video", "render"):
            line(g in groups or None, f"user in group {g}", "" if g in groups else f"sudo usermod -aG {g} $USER; log out/in")
    except Exception:  # noqa: BLE001
        pass
    for key, files in (("policy", ["act.xml"] if cfg["policy"]["backend"] == "openvino" else ["config.json"]),
                       ("inspect.patchcore", ["model.xml"]), ("inspect.classifier", ["best.xml", "classifier.json"])):
        sec, _, k = key.partition(".")
        d = Path(cfg[sec][k or "path"]) if cfg[sec].get(k or "path") else None
        if d is None:
            line(None, f"model {key}", "disabled")
            continue
        missing = [f for f in files if not (d / f).exists()]
        line(not missing, f"model {key}: {d}", f"missing {missing}: scripts/download_models.sh" if missing else "")
    from brick_sorter import calib

    cal_file = Path(cfg["arm"]["calibration_dir"]) / f"{cfg['arm']['id']}.json"
    if cal_file.exists():
        probs = calib.problems(calib.load(cal_file))
        line(not probs, f"calibration {cal_file}", "; ".join(probs))
    else:
        line(False, f"calibration {cal_file}", "missing")
    from brick_sorter.arm import Poses
    from brick_sorter.reasoner import required_poses

    poses = Poses(Path(cfg["arm"]["calibration_dir"]) / "poses.json")
    need = required_poses(cfg["rules"]) + ([cfg["policy"]["present_pose"]] if cfg["policy"].get("present_pose") else [])
    miss = [p for p in need if not poses.has(p)]
    line(not miss or None, "taught poses", f"teach: brick-sorter teach {' '.join(miss)}" if miss else ", ".join(sorted(poses.data)))
    if not a.no_hardware:
        from brick_sorter.arm import serial_ports, supply_volts

        ports = serial_ports()
        volts = {p: supply_volts(p) for p in ports}
        line(any(v and v >= 9 for v in volts.values()), "follower arm (servos at 12 V)", str(volts) or "no serial ports")
        try:
            import pyrealsense2 as rs

            serials = [d.get_info(rs.camera_info.serial_number) for d in rs.context().devices]
        except Exception as e:  # noqa: BLE001
            serials = []
            line(False, "pyrealsense2", str(e))
        for cam in CAMERAS:
            spec = cfg["cameras"][cam]
            fix = "not connected: run `brick-sorter cameras` to find and assign your cameras"
            if spec["kind"] == "realsense":
                ok = spec["source"] in serials
                line(ok, f"camera {cam}: RealSense {spec['source']}", "" if ok else f"{fix} (RealSense here: {serials})")
            elif spec["kind"] == "uvc":
                src = spec["source"]
                ok = Path(src).exists() if not str(src).isdigit() else Path(f"/dev/video{src}").exists()
                line(ok, f"camera {cam}: {src}", "" if ok else fix)
    print("\nall good" if not bad else f"\n{bad} problem(s)")
    return 0 if not bad else 1


# ---------------------------------------------------------------------------------------------- bench
def cmd_bench(a) -> int:
    import openvino as ov

    from brick_sorter.inspector import Inspector
    from brick_sorter.policy import OpenVINOACT, TorchACT

    cfg = C.load(a.config)
    rng = np.random.default_rng(0)
    frames = {c: rng.integers(0, 255, (480, 640, 3), dtype=np.uint8) for c in CAMERAS}
    state = np.array([0, -100, 95, 5, 0, 1], np.float32)
    devs = [d for d in (a.devices.split(",") if a.devices else ["CPU", "GPU", "NPU"])
            if any(x.startswith(d) for x in ov.Core().available_devices)]
    rows = []

    def timeit(fn, n):
        fn(); fn()
        ts = []
        for _ in range(n):
            t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1e3)
        return round(float(np.median(ts)), 2), round(float(np.percentile(ts, 99)), 2)

    if a.torch and Path(ROOT / "models" / "act").exists():
        try:
            p = TorchACT(ROOT / "models" / "act", a.torch)
            rows.append(("policy ACT chunk", f"PyTorch {a.torch} (baseline)", *timeit(lambda: p.predict_chunk(state, frames), a.n)))
        except Exception as e:  # noqa: BLE001
            print("torch baseline skipped:", e)
    for d in devs:
        try:
            p = OpenVINOACT(cfg["policy"]["path"], d)
            rows.append(("policy ACT chunk", f"OpenVINO {p.device}", *timeit(lambda: p.predict_chunk(state, frames), a.n)))
        except Exception as e:  # noqa: BLE001
            rows.append(("policy ACT chunk", f"OpenVINO {d}", "failed", str(e)[:60]))
        try:
            c = dict(cfg)
            c["inspect"] = {**cfg["inspect"], "device": d}
            ins = Inspector(c)
            imgs = [frames["inspect"]] * cfg["inspect"]["frames"]
            med, p99 = timeit(lambda: ins.inspect(imgs), a.n)
            n = cfg["inspect"]["frames"]
            rows.append((f"inspection ({n} frames: PatchCore+YOLO)", f"OpenVINO {ins.device}", med, p99))
        except Exception as e:  # noqa: BLE001
            rows.append(("inspection", f"OpenVINO {d}", "failed", str(e)[:60]))
    print(f"\n{'stage':42s} {'backend':28s} {'median ms':>10s} {'p99 ms':>10s}")
    for r in rows:
        print(f"{r[0]:42s} {r[1]:28s} {str(r[2]):>10s} {str(r[3]):>10s}")
    out = ROOT / "runs" / f"bench-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps([dict(zip(["stage", "backend", "median_ms", "p99_ms"], r)) for r in rows], indent=2))
    print(f"\nsaved {out}")
    return 0


# ---------------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="brick-sorter", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", "--config", default=os.environ.get("BRICK_SORTER_CONFIG"), help="TOML overlay on config/station.toml")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sim", help="dry run with everything simulated")
    s.add_argument("--bricks", default="plain,circle,cross,faint,unclear,plain")
    s.add_argument("--speed", type=float, default=3.0, help="simulated arm speed-up")
    s.add_argument("--step", action="store_true", help="wait for NEXT between bricks")
    s.add_argument("--no-start", action="store_true", help="wait for START on the page")
    s.add_argument("--headless", action="store_true", help="no web page; exit when done")
    s.add_argument("--port", type=int)
    s.set_defaults(fn=cmd_sim)

    s = sub.add_parser("run", help="the real station")
    s.add_argument("--start", action="store_true", help="start the loop immediately (default: wait for START)")
    s.add_argument("--step", action="store_true", help="wait for NEXT between bricks")
    s.add_argument("--max-bricks", type=int)
    s.add_argument("--port", type=int)
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("teach", help="hand-guide the arm and save named poses")
    s.add_argument("names", nargs="+", help="e.g. home yellow orange purple (optional: <bowl>_approach)")
    s.set_defaults(fn=cmd_teach)

    s = sub.add_parser("calib", help="check the calibration against the motors and the policy's training data")
    s.add_argument("--against", help="reference calibration (default: calibration/reference/act_training_follower.json)")
    s.add_argument("--offline", action="store_true", help="files only, no arm")
    s.add_argument("--write", action="store_true", help="offer to write the file to the motors if they differ")
    s.set_defaults(fn=cmd_calib)

    s = sub.add_parser("cameras", help="find and test every camera, then assign wrist / top / inspect")
    s.add_argument("--assign", help="non-interactive, e.g. wrist=2,top=1,inspect=3 (numbers from the listing)")
    s.add_argument("--no-assign", action="store_true", help="only list and test")
    s.add_argument("--preview", action="store_true", help="live page of all cameras to tell them apart")
    s.add_argument("--preview-port", type=int, default=8081)
    s.set_defaults(fn=cmd_cameras)

    s = sub.add_parser("doctor", help="check the machine")
    s.add_argument("--no-hardware", action="store_true")
    s.set_defaults(fn=cmd_doctor)

    s = sub.add_parser("bench", help="latency per device (OPTIMIZE table)")
    s.add_argument("--devices", help="comma list, default CPU,GPU,NPU (those present)")
    s.add_argument("--torch", default=None, help="also time the PyTorch baseline on this device (cpu | xpu)")
    s.add_argument("-n", type=int, default=20)
    s.set_defaults(fn=cmd_bench)

    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

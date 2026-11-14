# brick-sorter

An SO-101 arm picks up a brick and shows it to an inspection camera. Anomalib and a classifier decide whether the
brick is plain (good), has a circle drawn on it (minor defect) or has a cross (defective). A rule table then picks
the bowl, and the arm places the brick there. Built for Intel Core Ultra with OpenVINO on CPU, GPU and NPU.
Standalone; no other repository needed.

```
 HOME ─► PICK ──────────────► INSPECT ───────────────────► DECIDE ─────────────► PLACE ─► HOME ─► next brick
         ACT policy (GPU)      PatchCore + YOLO (NPU)       rules (CPU)            taught bowl pose
         "brick held still     Anomalib: marked or not       plain  → yellow
          at the camera"       YOLO: plain/circle/cross      circle → orange
                               both must agree, otherwise    cross  → purple
                               uncertain → turn and look     uncertain ×2 → purple (safe default)
                               again
```

The policy only does the hard, learned part: picking the brick and presenting it to the camera. Placing uses
poses you teach by hand, so sorting stays deterministic.

## Quick start: no hardware needed

```bash
uv sync
uv run brick-sorter sim                 # open http://localhost:8080
uv run brick-sorter sim --headless      # terminal only; exits when all bricks are sorted
uv run pytest -q                        # 24 tests (the real-model tests need models/)
```

The simulation sorts `plain, circle, cross, faint, unclear, plain`. "faint" is read on the second look. "unclear"
is never readable, so it ends in the purple bowl.

## Deploy on the Intel machine

```bash
git clone <this repo> ~/brick-sorter && cd ~/brick-sorter
scripts/setup_intel.sh                  # drivers, groups, udev rules, Python env, models, then `doctor`
# log out and back in if it added you to groups
```

`models/act_openvino`, the OpenVINO export of the policy, can't be downloaded directly. `scripts/download_models.sh` fetches all four models (ACT checkpoint, its OpenVINO
export, PatchCore, YOLO) from this repo's `models-v1` release and verifies their checksums.

Then, in this order:

| Step | Command | What it checks or does |
|---|---|---|
| 1 | `uv run brick-sorter doctor` | OpenVINO GPU/NPU, groups, models, arm at 12 V, camera serials |
| 2 | `uv run brick-sorter calib` | **reads the motors (read-only)** and compares them with `calibration/follower/follower.json` |
| 3 | `uv run brick-sorter teach home yellow orange purple` | torque off, hand-guide the arm to each pose, press Enter |
| 4 | `uv run brick-sorter bench --torch cpu` | OPTIMIZE table: PyTorch baseline vs OpenVINO CPU/GPU/NPU, median and p99 |
| 5 | `scripts/run.sh --step` | loads everything, serves `http://<host>:8080`; press **Start**, then **Next brick** for each brick |
| 6 | `scripts/run.sh` | continuous loop; `scripts/install_service.sh` starts it at login |

Put machine-specific settings in `config/local.toml` (ports, camera serials, devices, crop). It's layered on top
of `config/station.toml`. The setup script creates it for you.

## Safety

- **Calibration lock.** The arm won't connect unless the calibration stored in the motors matches
  `calibration/follower/follower.json`. LeRobot doesn't check this, and a mismatch makes every joint angle wrong.
  `brick-sorter calib --write` writes the file to the motors, but only after you confirm.
- **No lunge on startup.** Each motor's goal position is set to where it already is before torque goes on.
- **Speed limits.** Every command is rate-limited per joint (`arm.max_speed` and `move_speed`, in degrees/s).
- **Stopping.** The page's **E-STOP**, **Stop** and Ctrl+C all abort the current move and hold position.
  The arm is never dropped, because torque stays on.
- **No surprise moves.** The loop never starts on its own. It needs **Start**, or `--start`.

## Calibration: what we found

`calibration/follower/follower.json` is your real follower calibration, the same one Studio uses.
`calibration/reference/act_training_follower.json` is the calibration the downloaded ACT policy was trained with.
`brick-sorter calib --offline` compares the two:

```
shoulder_pan 21.0°   shoulder_lift -116.4°   elbow_flex -26.3°   wrist_flex -95.2°   wrist_roll -145.5°
```

These differences are far too large for a calibration drift. The servo horns were re-mounted, or it's a different
arm, so a constant joint offset can't make the existing policy work. **Train on your colleagues' demos recorded
with this calibration** (below). If the gap is ever small (under 10°), put the printed numbers in
`[arm] joint_offset` and the policy can be reused.

## Training your own pick policy

1. Record in the exact deployment setup. Name the cameras `wrist`, `top` and `inspect`, record at 640×480 and
   30 fps, lock auto-exposure and white balance, and never move `top` or `inspect` afterwards. Each demo is one
   skill: pick the brick from the table and hold it still in front of the inspect camera. Start with 50 or more
   demos, with the brick in 5 or more positions and around 10 demos at each position, using the same grasp
   strategy every time.
2. Train ACT. You can use Physical AI Studio (local or remote GPU), or LeRobot:
   `lerobot-train --policy.type=act --dataset.repo_id=<you>/<dataset> ...`
3. Export to OpenVINO with Studio, and copy the export folder (`act.xml`, `act.bin`, `manifest.json`) to
   `models/act_openvino`. The loader checks the input names and applies the manifest's resize. For a raw LeRobot
   checkpoint, set `[policy] backend = "torch"` and `path = "models/act"`.

## Layout

```
brick_sorter/
  orchestrator.py  state machine, per-stage latency, logs in runs/<time>/{events,bricks}.jsonl + summary.json
  policy.py        OpenVINO ACT (matches PyTorch to within 6e-5) | PyTorch ACT (baseline) | simulated
  inspector.py     PatchCore + YOLO on OpenVINO (fixed input shapes so they can run on the NPU), agreement rule, heatmap
  reasoner.py      outcome → bowl, re-inspect, fallback
  arm.py           SO-101 with the safety checks above | simulated arm
  cameras.py       UVC (MJPG) and RealSense threads that reconnect after failures | simulated
  calib.py         calibration comparison and offset formula (checked against a known recalibration)
  web.py           control page: live cameras, state, heatmap, bowl counts, latency, controls
  cli.py           sim | run | teach | calib | doctor | bench
config/station.toml   all settings, with comments
scripts/              setup_intel.sh, download_models.sh, run.sh, install_service.sh
```

## Measured so far (on an AMD laptop CPU, for reference; run `bench` on the Intel machine)

| Stage | Backend | Median | p99 |
|---|---|---|---|
| ACT, one 100-step chunk | PyTorch CPU (baseline) | 482 ms | 491 ms |
| ACT, one 100-step chunk | OpenVINO CPU | 284 ms | 306 ms |
| Inspection (3 frames, PatchCore + YOLO) | OpenVINO CPU | 424 ms | 464 ms |

One chunk drives about 3 seconds of motion at 30 Hz, so the policy runs once every 100 control steps.

## Troubleshooting

- **The arm or cameras vanish on WSL.** Windows took the USB devices back. Run `usbipd list`, then
  `usbipd attach --wsl --busid <id>`. To make this automatic, keep `--auto-attach` running for each device, and
  turn off USB selective suspend.
- **The wrist camera gives no frames.** Keep `fourcc = "MJPG"`. Uncompressed YUYV times out on shared or
  virtual USB.
- **The arm refuses to connect because of a calibration mismatch.** Run `brick-sorter calib` to see the
  differences. Write the file to the motors (`--write`) only if the file is right for this arm.
- **The inspection crop is wrong.** Watch the inspect camera on the page while the arm presents a brick, then set
  `[inspect] crop = [x, y, w, h]` around the brick face. PatchCore was trained on 320×320 crops.

import json
from pathlib import Path

import numpy as np
import pytest

from brick_sorter import ROOT, calib
from brick_sorter import config as C
from brick_sorter.arm import step_toward
from brick_sorter.inspector import decide as inspect_decide
from brick_sorter.reasoner import decide, required_poses

RULES = {"plain": "yellow", "circle": "orange", "cross": "purple", "max_uncertain": 2, "fallback": "purple"}


@pytest.mark.parametrize("outcome,bowl", [("plain", "yellow"), ("circle", "orange"), ("cross", "purple")])
def test_rules_map_outcomes_to_bowls(outcome, bowl):
    d = decide(outcome, 1, RULES)
    assert (d.action, d.bowl, d.forced) == ("place", bowl, False)


def test_uncertain_reinspects_then_falls_back():
    assert decide("uncertain", 1, RULES).action == "reinspect"
    d = decide("uncertain", 2, RULES)
    assert (d.action, d.bowl, d.forced) == ("place", "purple", True)


def test_required_poses():
    assert required_poses(RULES) == ["home", "orange", "purple", "yellow"]


P = lambda plain, circle, cross: {"plain": plain, "circle": circle, "cross": cross}  # noqa: E731


@pytest.mark.parametrize("anomalous,probs,expected", [
    (True, P(.05, .05, .9), "cross"),
    (True, P(.05, .9, .05), "circle"),
    (False, P(.9, .05, .05), "plain"),
    (True, P(.9, .05, .05), "uncertain"),   # models disagree
    (False, P(.05, .05, .9), "uncertain"),  # models disagree
    (True, P(.1, .45, .45), "uncertain"),   # classifier unsure
    (True, None, "cross"),                  # no classifier: marked -> defective
    (False, None, "plain"),
])
def test_inspection_agreement_rule(anomalous, probs, expected):
    assert inspect_decide(anomalous, probs, 0.8, 0.2)[0] == expected


def test_offset_formula_reproduces_known_recalibration():
    """A recorded recalibration (2026-11-14), joints whose horns were not re-mounted."""
    c = lambda h, lo, hi: {"homing_offset": h, "range_min": lo, "range_max": hi}  # noqa: E731
    old = {"shoulder_pan": c(2036, 818, 3276), "shoulder_lift": c(1276, 952, 3143),
           "elbow_flex": c(1745, 930, 3165), "wrist_flex": c(1627, 854, 3241), "wrist_roll": c(0, 0, 4095)}
    new = {"shoulder_pan": c(1804, 1074, 3021), "shoulder_lift": c(1367, 857, 3237),
           "elbow_flex": c(1743, 940, 3155), "wrist_flex": c(1627, 857, 3237), "wrist_roll": c(0, 0, 4095)}
    off = calib.offset_deg(new, old)
    expected = {"shoulder_pan": 20.35, "shoulder_lift": -7.96, "elbow_flex": 0.18, "wrist_flex": 0.04}
    for j, v in expected.items():
        assert off[j] == pytest.approx(v, abs=0.01)
    assert calib.wrap180(370) == pytest.approx(10)


def test_offset_between_shipped_calibrations_is_flagged_large():
    cur = calib.load(ROOT / "calibration/follower/follower.json")
    ref = calib.load(ROOT / "calibration/reference/act_training_follower.json")
    off = calib.offset_deg(cur, ref)
    assert max(abs(v) for v in off.values()) > 10  # different assembly: offsets cannot fix it
    assert calib.compare(cur, cur) == []
    assert calib.compare(cur, ref)


def test_step_toward_rate_limits_each_joint():
    out = step_toward(np.zeros(6), np.array([10, -10, 0.5, 0, 0, 3.0]), 1.0)
    assert out.tolist() == [1, -1, 0.5, 0, 0, 1]


def test_config_loads_and_resolves_paths():
    cfg = C.load()
    assert Path(cfg["policy"]["path"]).is_absolute()
    assert cfg["rules"]["plain"] == "yellow"
    fake = C.load(None, C.fake_overrides())
    assert fake["arm"]["port"] == "fake" and fake["cameras"]["top"]["kind"] == "fake"

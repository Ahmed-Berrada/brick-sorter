"""Load config/station.toml into a plain nested dict with paths resolved against the project root."""

from __future__ import annotations

import copy
import tomllib
from pathlib import Path

from brick_sorter import ROOT

DEFAULT = ROOT / "config" / "station.toml"
PATH_KEYS = {("arm", "calibration_dir"), ("policy", "path"), ("inspect", "patchcore"), ("inspect", "classifier"),
             ("log", "dir")}


def resolve(p: str | Path) -> Path:
    p = Path(p).expanduser()
    return p if p.is_absolute() else ROOT / p


def load(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """The default config, overlaid with `path` (if it is not the default) and then `overrides`."""
    cfg = tomllib.loads(DEFAULT.read_text())
    if path and resolve(path) != DEFAULT:
        merge(cfg, tomllib.loads(resolve(path).read_text()))
    if overrides:
        merge(cfg, overrides)
    for section, key in PATH_KEYS:
        v = cfg.get(section, {}).get(key)
        if v:
            cfg[section][key] = str(resolve(v))
    return cfg


def merge(base: dict, extra: dict) -> dict:
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            merge(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def fake_overrides() -> dict:
    """Everything simulated: runs on any laptop with no arm, camera or model."""
    return {"arm": {"port": "fake"}, "policy": {"backend": "fake"}, "inspect": {"backend": "fake"},
            "cameras": {n: {"kind": "fake", "source": n} for n in ("wrist", "top", "inspect")}}

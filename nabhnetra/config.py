"""Load the YAML configuration, optionally with command-line overrides."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

from .common import ROOT

DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None, overrides=None) -> dict:
    p = Path(path) if path else DEFAULT_CONFIG
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return apply_overrides(cfg, overrides or [])


def get_key(cfg: dict, dotted: str):
    node = cfg
    for part in dotted.split("."):
        node = node[part]
    return node


def set_key(cfg: dict, dotted: str, value) -> dict:
    """Return a copy of cfg with one nested key replaced, e.g. set_key(cfg, 'tracker.q', 3.0)."""
    cfg = copy.deepcopy(cfg)
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        if part not in node:
            raise KeyError(f"no config key '{dotted}'")
        node = node[part]
    if parts[-1] not in node:
        raise KeyError(f"no config key '{dotted}'")
    node[parts[-1]] = value
    return cfg


def apply_overrides(cfg: dict, overrides) -> dict:
    """Apply 'a.b.c=value' strings; values are parsed as YAML (numbers, lists, true/false)."""
    for item in overrides:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"override '{item}' must look like key=value")
        cfg = set_key(cfg, key.strip(), yaml.safe_load(raw))
    return cfg

"""Load the YAML configuration."""
from __future__ import annotations

from pathlib import Path

import yaml

from .common import ROOT

DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path else DEFAULT_CONFIG
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)

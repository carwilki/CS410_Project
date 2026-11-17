"""Load config.yaml (project root) into a nested dict with attribute access."""
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


class Cfg(dict):
    def __getattr__(self, k) -> Any:
        v = self[k]
        return Cfg(v) if isinstance(v, dict) else v


def load(path: Path = ROOT / "config.yaml") -> Cfg:
    return Cfg(yaml.safe_load(path.read_text()))


def path(rel: str) -> Path:
    return ROOT / rel

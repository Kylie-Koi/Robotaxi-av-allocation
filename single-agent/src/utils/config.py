"""Load the project's YAML config into a plain dict, resolved against the repo root."""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


def load_config(path: Path | str | None = None) -> dict:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_repo_root"] = str(REPO_ROOT)
    return cfg


def resolve_path(cfg: dict, relative: str) -> Path:
    return Path(cfg["_repo_root"]) / relative

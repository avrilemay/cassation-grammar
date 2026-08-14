"""Loads config.yaml, the single source of truth for every path and setting the pipeline uses.

It looks first for an explicit path given by the caller, then an
environment variable that lets a user redirect it to their own copy (for
example, pointing the data folder at an external drive) without touching
the tracked file, and falls back to the version committed in the
repository. A small helper also turns a relative path from the config into
an absolute one, based at the repository root.
"""
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"
CONFIG_PATH_ENV_VAR = "CASSATION_GRAMMAR_CONFIG"


def load_config(path=None):
    """Load config.yaml (or an alternate path) as a plain dict.

    See the module docstring for the resolution order.
    """
    if path is None:
        path = os.environ.get(CONFIG_PATH_ENV_VAR) or DEFAULT_CONFIG_PATH
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path_str):
    """Resolve a config path relative to the repository root.

    Absolute paths are returned unchanged, so a user can point config.yaml at
    an external data location (e.g. a Zenodo download directory) without any
    code change.
    """
    p = Path(path_str)
    return p if p.is_absolute() else (REPO_ROOT / p)

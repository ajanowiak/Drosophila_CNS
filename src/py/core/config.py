# config.py

"""
Loads config/config.yml for scripts that need a config value not already
supplied as a CLI argument (currently: plot colors).

Pipeline context: importable from any stage directory via PYTHONPATH=src/py.

Inputs: config/config.yml
Outputs: none (returns the parsed config as a dict)
"""

from pathlib import Path

import yaml

CONFIG_PATH = Path("config/config.yml")


def load_config() -> dict:
    """Load and return config/config.yml as a plain dict."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

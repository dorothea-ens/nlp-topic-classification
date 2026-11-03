"""Load numeric-named research scripts without changing their file locations."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(filename: str):
    path = ROOT / filename
    spec = importlib.util.spec_from_file_location("script_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def scripts():
    return load_script

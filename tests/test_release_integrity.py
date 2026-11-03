from pathlib import Path
import importlib.util
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("release_audit", ROOT/"17_release_audit.py")
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def test_release_consistent():
    failures, warnings=module.audit(ROOT)
    assert not failures, failures

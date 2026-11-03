from pathlib import Path


def test_project_source_fingerprints_exist(scripts):
    m = scripts("13_verify_project.py")
    info = m.audit(Path(__file__).resolve().parents[1])
    assert info["all_sources_present"]
    assert len(info["sources"]) == len(m.SOURCES)
    assert all(len(row["sha256"]) == 64 for row in info["sources"].values())
    assert isinstance(info["all_three_models_ready"], bool)


def test_missing_artifacts_do_not_look_ready(tmp_path, scripts):
    m = scripts("13_verify_project.py")
    result = m.audit(tmp_path)
    assert not result["all_three_models_ready"]
    assert not result["all_sources_present"]

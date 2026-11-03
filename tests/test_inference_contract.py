from pathlib import Path

import numpy as np
import pytest


def test_inference_model_validation_and_ranking(scripts, tmp_path):
    engine = scripts("nlp_inference.py")
    names = ["a", "b", "c"]
    order = engine._ranking(np.array([0.3, 0.8, 0.2]), names, kind="score")
    assert [item["topic"] for item in order] == ["b", "a", "c"]
    engine._check_label_order(names, names, "test")
    with pytest.raises(ValueError, match="class order"):
        engine._check_label_order(names, ["a", "c", "b"], "test")
    assert engine.locate_artifacts(tmp_path) == {}


def test_embeddings_representation_uses_validation_only(scripts):
    engine = scripts("nlp_inference.py")
    summary = {"selected": {
        "head_only": {"macro_f1": .64, "accuracy": .70},
        "head_tail_mean": {"macro_f1": .66, "accuracy": .68},
    }}
    models = {"head_only": {}, "head_tail_mean": {}}
    assert engine.selected_representation(summary, models) == "head_tail_mean"
    assert np.allclose(engine._normalize([[3, 4]]), [[.6, .8]])


def test_full_comparison_requires_models(scripts, tmp_path):
    compare = scripts("09_compare_models.py")
    with pytest.raises(FileNotFoundError, match="missing"):
        compare.find_inputs(tmp_path)


def test_quick_mode_selected_model_is_detected(scripts, tmp_path):
    import json
    engine = scripts("nlp_inference.py")
    model = tmp_path / "outputs" / "08_quick_EXPLORATORY" / "models" / "p100_len256_lr5e-05"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"stub")
    manifest = model.parent.parent / "08_selected_configuration.json"
    manifest.write_text(json.dumps({
        "research_mode": "08_quick_EXPLORATORY",
        "selected_full_model": r"C:\archive\p100_len256_lr5e-05",
    }), encoding="utf-8")
    found = engine.locate_artifacts(tmp_path)
    assert engine.TRANSFORMER_NAME in found
    assert found[engine.TRANSFORMER_NAME].path == model

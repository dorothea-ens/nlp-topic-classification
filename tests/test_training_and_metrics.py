from __future__ import annotations

import numpy as np
import pytest


def test_svm_pipeline_on_synthetic_texts(scripts):
    svm = scripts("04_optimize_svm.py")
    texts = [
        "galaxy telescope planetary star mission", "galaxy telescope asteroid orbit mission",
        "planet satellite telescope galaxy mission", "space galaxy planet telescope",
        "hockey goalkeeper puck match ice", "hockey goalkeeper puck rink ice",
        "goalkeeper hockey puck match rink", "hockey team puck goal match",
    ]
    labels = np.array([0] * 4 + [1] * 4)
    pipeline = svm.build_pipeline("unigram_bigram_no_stopwords", 1.0)
    pipeline.fit(texts, labels)
    pred = pipeline.predict(["galaxy satellite telescope", "hockey puck goalkeeper"])
    assert pred.tolist() == [0, 1]
    s = svm.metrics(np.array([0, 1]), pred, [0, 1])
    assert s["accuracy"] == pytest.approx(1.0)
    assert s["macro_f1"] == pytest.approx(1.0)


def test_nested_subsets_are_nested_and_repeatable(scripts):
    svm = scripts("04_optimize_svm.py")
    y = np.repeat(np.arange(5), 40)
    a = svm.nested_stratified_indices(y)
    b = svm.nested_stratified_indices(y)
    assert len(a[1.0]) == len(y)
    for f in svm.TRAIN_FRACTIONS:
        np.testing.assert_array_equal(a[f], b[f])
        assert set(np.unique(y[a[f]])) == set(range(5))
    for small, large in zip(svm.TRAIN_FRACTIONS, svm.TRAIN_FRACTIONS[1:]):
        assert set(a[small]).issubset(set(a[large]))


def test_bootstrap_uses_same_case_indices_for_models(scripts):
    compare = scripts("09_compare_models.py")
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    predictions = {
        compare.MODEL_NAMES[0]: y.copy(),
        compare.MODEL_NAMES[1]: np.array([0, 1, 2, 0, 1, 0, 0, 2, 2]),
        compare.MODEL_NAMES[2]: np.array([0, 1, 2, 0, 1, 2, 0, 2, 2]),
    }
    output_a = compare.paired_bootstrap_macro_f1(y, predictions, repetitions=40, seed=42)
    output_b = compare.paired_bootstrap_macro_f1(y, predictions, repetitions=40, seed=42)
    assert output_a == output_b

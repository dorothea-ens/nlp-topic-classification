"""Installment 11: Auditable post-hoc NLP error and uncertainty analysis.

Prerequisites:
    python 04_optimize_svm.py
    python 06_optimize_embeddings.py
    python 08_optimize_distilbert.py
    python 09_compare_models.py

Install dependencies (same as Installment 9, plus scipy):
    python -m pip install scikit-learn numpy scipy matplotlib joblib sentence-transformers torch transformers

Offline self-test (does NOT download models or load the held-out test data):
    python 11_error_analysis.py --self-test

Analyze fixed, preselected classifiers (does NOT retrain or optimize anything):
    python 11_error_analysis.py

This analysis uses the official 20 Newsgroups TEST data ONLY after the final
one-time comparison from Installment 9 has already completed. Because that
script did not save per-document predictions, this script reproduces inference
once using its exact functions, confirms that results match the prior scores
and confusion matrices, and caches a detailed analysis. Never use findings to
retune models against the held-out test set.

Main outputs in outputs/11_error_analysis/:
    11_pairwise_statistics.csv
    11_error_by_length.csv
    11_top_confusions.csv
    11_model_disagreements.csv
    11_review_queue.csv
    11_error_by_length.png
    11_bootstrap_differences.png
    11_analysis_manifest.json

Review queue contains excerpts from third-party source data for PERSONAL
manual inspection. Check dataset licensing before publishing any examples.

CAUTION: Bootstrap intervals, McNemar tests and Holm-adjusted p-values are
exploratory post-hoc diagnostics, not proof of practical superiority. Treat
the same test set as a fixed final assessment; do not iterate on its labels.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from itertools import combinations
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parent
IN_DIR = ROOT / "outputs" / "09_final_comparison"
OUT_DIR = ROOT / "outputs" / "11_error_analysis"
SEED = 42
BOOTSTRAP_REPEATS = 1000


def load_previous_module():
    """Import functions from the exact published Installment 9 file."""
    script = ROOT / "09_compare_models.py"
    if not script.is_file():
        raise FileNotFoundError(f"Required previous script missing: {script}")
    spec = importlib.util.spec_from_file_location("nlp_installment_09", script)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot import Installment 9 script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def csv_write(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def text_length_bucket(text):
    n_words = len(text.split())
    if n_words <= 50:
        return "short_0_50"
    if n_words <= 200:
        return "medium_51_200"
    return "long_201_plus"


def assert_valid_predictions(y_true, predictions, label_names):
    true = np.asarray(y_true, dtype=np.int64)
    if true.ndim != 1 or len(true) == 0:
        raise ValueError("Expected a nonempty one-dimensional truth array")
    k = len(label_names)
    if k < 2 or not np.isin(true, np.arange(k)).all():
        raise ValueError("True labels invalid for the declared categories")
    if len(predictions) != 3:
        raise ValueError("Exactly three named model predictions required")
    for model, arr in predictions.items():
        pred = np.asarray(arr)
        if pred.shape != true.shape or not np.isin(pred, np.arange(k)).all():
            raise ValueError(f"Invalid prediction indices or length: {model}")


def score_against_prior(y_true, preds, classes, prior, comparison):
    """Prevent silently analyzing a changed checkpoint or a mismatched test set."""
    names = comparison.MODEL_NAMES
    if list(prior.get("classes", [])) != list(classes):
        raise ValueError("Category order differs from Installment 9")
    if prior.get("dataset") != "20 Newsgroups; official test partition":
        raise ValueError("Unexpected prior dataset manifest")
    score_map = {row["model"]: row for row in prior["models"]}
    if set(score_map) != set(names):
        raise ValueError("Not the expected three comparison models")
    for name, slug in zip(names, ("svm", "minilm", "distilbert")):
        observed = comparison.score_predictions(y_true, preds[name], classes, 1.0)
        old = score_map[name]
        if int(old["test_documents"]) != len(y_true):
            raise ValueError(f"{name}: test document count differs")
        for metric in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"):
            if not np.isclose(observed[metric], old[metric], atol=1e-10, rtol=0):
                raise ValueError(
                    f"{name}: {metric} mismatch ({observed[metric]} vs {old[metric]}). "
                    "Check checkpoint, tokenizer, and dataset versions."
                )
        cm_path = IN_DIR / f"09_{slug}_confusion_matrix.csv"
        if not cm_path.exists():
            raise FileNotFoundError(f"Expected previous confusion matrix: {cm_path}")
        with cm_path.open(encoding="utf-8", newline="") as fh:
            saved = list(csv.DictReader(fh))
        old_cm = np.array([[int(saved[i][label]) for label in classes]
                           for i in range(len(classes))], dtype=int)
        if any(saved[i]["actual_category"] != classes[i] for i in range(len(classes))):
            raise ValueError(f"{name}: class order mismatch in confusion matrix")
        if not np.array_equal(old_cm, confusion_matrix(
                y_true, preds[name], labels=np.arange(len(classes)))):
            raise ValueError(f"{name}: predictions do not reproduce previous confusion matrix")


def paired_statistics(y_true, predictions, classes, *, repeats=1000, seed=SEED):
    """Fixed-model test comparisons. Same bootstrap sample for all pairs."""
    assert_valid_predictions(y_true, predictions, classes)
    ys = np.asarray(y_true, dtype=np.int64)
    labels = np.arange(len(classes))
    rng = np.random.default_rng(seed)
    n = len(ys)
    pairs = list(combinations(predictions.keys(), 2))
    diffs = {pair: np.empty(repeats, dtype=float) for pair in pairs}
    for i in range(repeats):
        idx = rng.integers(0, n, n)
        boot = {name: f1_score(ys[idx], np.asarray(p)[idx], labels=labels,
                               average="macro", zero_division=0)
                for name, p in predictions.items()}
        for first, second in pairs:
            diffs[first, second][i] = boot[second] - boot[first]
    results = []
    for first, second in pairs:
        a = np.asarray(predictions[first]) == ys
        b = np.asarray(predictions[second]) == ys
        a_only = int(np.sum(a & ~b))
        b_only = int(np.sum(~a & b))
        disagreeing_correctness = a_only + b_only
        # Exact two-sided McNemar via binomial distribution (conditional test).
        p = float(binomtest(min(a_only, b_only), disagreeing_correctness,
                            p=0.5, alternative="two-sided").pvalue) \
            if disagreeing_correctness else 1.0
        a_f1 = f1_score(ys, predictions[first], labels=labels,
                        average="macro", zero_division=0)
        b_f1 = f1_score(ys, predictions[second], labels=labels,
                        average="macro", zero_division=0)
        results.append({
            "model_a": first, "model_b": second,
            "delta_macro_f1_b_minus_a": float(b_f1 - a_f1),
            "bootstrap_95pct_low": float(np.quantile(diffs[first, second], .025)),
            "bootstrap_95pct_high": float(np.quantile(diffs[first, second], .975)),
            "a_correct_b_wrong": a_only,
            "a_wrong_b_correct": b_only,
            "mcnemar_exact_p": p,
            "bootstrap_repetitions": repeats,
        })
    # Holm familywise adjustment for the three exploratory paired comparisons.
    order = sorted(range(len(results)), key=lambda i: results[i]["mcnemar_exact_p"])
    running = 0.0
    for pos, index in enumerate(order):
        candidate = (len(results) - pos) * results[index]["mcnemar_exact_p"]
        running = max(running, candidate)
        results[index]["mcnemar_holm_p"] = min(1.0, running)
    return results


def category_confusions(ys, preds, classes, top_n=12):
    output = []
    for model, p in preds.items():
        matrix = confusion_matrix(ys, p, labels=np.arange(len(classes)))
        pairs = [(int(matrix[i, j]), i, j)
                 for i in range(len(classes)) for j in range(len(classes))
                 if i != j and matrix[i, j] > 0]
        for rank, (count, i, j) in enumerate(sorted(pairs, reverse=True)[:top_n], 1):
            output.append({"model": model, "rank": rank,
                           "true_category": classes[i], "predicted_category": classes[j],
                           "misclassified_documents": count,
                           "true_category_support": int(np.sum(np.asarray(ys) == i))})
    return output


def length_breakdown(texts, y_true, preds, classes):
    ys = np.asarray(y_true)
    lengths = np.array([text_length_bucket(t) for t in texts], dtype=object)
    output = []
    for name, p in preds.items():
        p = np.asarray(p)
        for label in ("short_0_50", "medium_51_200", "long_201_plus"):
            mask = lengths == label
            if not np.any(mask):
                continue
            output.append({
                "model": name, "length_group": label,
                "documents": int(mask.sum()),
                "accuracy": float(accuracy_score(ys[mask], p[mask])),
                "macro_f1_20_categories": float(f1_score(
                    ys[mask], p[mask], labels=np.arange(len(classes)),
                    average="macro", zero_division=0)),
                "errors": int(np.sum(ys[mask] != p[mask])),
            })
    return output


def document_diagnostics(texts, y_true, predictions, classes):
    names = list(predictions)
    ys = np.asarray(y_true)
    pred_array = [np.asarray(predictions[n]) for n in names]
    rows = []
    for i, (txt, label) in enumerate(zip(texts, ys)):
        chosen = [int(p[i]) for p in pred_array]
        different = len(set(chosen)) > 1
        errors = sum(p != label for p in chosen)
        if not different and errors == 0:
            continue
        row = {
            "row_index": i,
            "true_category": classes[label],
            "word_count": len(txt.split()),
            "length_group": text_length_bucket(txt),
            "models_wrong": int(errors),
            "models_disagree": int(different),
            "text_excerpt": " ".join(txt.strip().split())[:220],
        }
        for slug, p in zip(("svm", "minilm", "distilbert"), chosen):
            row[f"{slug}_prediction"] = classes[p]
            row[f"{slug}_correct"] = int(p == label)
        rows.append(row)
    return rows


def save_figures(pairwise, length_rows, out):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(pairwise))
    y = np.asarray([row["delta_macro_f1_b_minus_a"] for row in pairwise])
    lo = np.asarray([row["bootstrap_95pct_low"] for row in pairwise])
    hi = np.asarray([row["bootstrap_95pct_high"] for row in pairwise])
    ax.errorbar(x, y, yerr=[np.maximum(0, y - lo), np.maximum(0, hi - y)],
                fmt="o", capsize=4)
    ax.axhline(0, linestyle="--", linewidth=1, color="gray")
    ax.set_xticks(x, [f"{r['model_b']}\nminus {r['model_a']}" for r in pairwise],
                  fontsize=8)
    ax.set_ylabel("Difference in held-out macro-F1")
    ax.set_title("Exploratory paired bootstrap 95% intervals")
    fig.tight_layout()
    fig.savefig(out / "11_bootstrap_differences.png", dpi=180)
    plt.close(fig)

    models = list(dict.fromkeys(row["model"] for row in length_rows))
    groups = ["short_0_50", "medium_51_200", "long_201_plus"]
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    barwidth = .8 / len(models)
    for i, model in enumerate(models):
        lookup = {(r["model"], r["length_group"]): r for r in length_rows}
        vals = [lookup.get((model, group), {}).get("accuracy", np.nan)
                for group in groups]
        ax.bar(x - .4 + barwidth * (i + .5), vals, width=barwidth, label=model)
    ax.set_xticks(x, ["0–50 words", "51–200 words", "201+ words"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Test accuracy")
    ax.set_title("Post-hoc error patterns by document length")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "11_error_by_length.png", dpi=180)
    plt.close(fig)


def generate_analysis(texts, y_true, predictions, classes, out, *, repeats=1000):
    """Pure post-hoc computation; no pretrained models required."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    assert_valid_predictions(y_true, predictions, classes)
    if len(texts) != len(y_true):
        raise ValueError("Document text and gold labels have different lengths")
    pairwise = paired_statistics(y_true, predictions, classes, repeats=repeats)
    confusions = category_confusions(y_true, predictions, classes)
    lengths = length_breakdown(texts, y_true, predictions, classes)
    details = document_diagnostics(texts, y_true, predictions, classes)
    sorted_queue = sorted(details, key=lambda r: (
        -r["models_wrong"], -r["models_disagree"], r["row_index"]))[:60]
    cols_pair = ["model_a", "model_b", "delta_macro_f1_b_minus_a",
                 "bootstrap_95pct_low", "bootstrap_95pct_high",
                 "a_correct_b_wrong", "a_wrong_b_correct",
                 "mcnemar_exact_p", "mcnemar_holm_p", "bootstrap_repetitions"]
    cols_length = ["model", "length_group", "documents", "accuracy",
                   "macro_f1_20_categories", "errors"]
    cols_conf = ["model", "rank", "true_category", "predicted_category",
                 "misclassified_documents", "true_category_support"]
    cols_rows = ["row_index", "true_category", "word_count", "length_group",
                 "models_wrong", "models_disagree", "svm_prediction", "svm_correct",
                 "minilm_prediction", "minilm_correct", "distilbert_prediction",
                 "distilbert_correct", "text_excerpt"]
    csv_write(out / "11_pairwise_statistics.csv", pairwise, cols_pair)
    csv_write(out / "11_error_by_length.csv", lengths, cols_length)
    csv_write(out / "11_top_confusions.csv", confusions, cols_conf)
    csv_write(out / "11_model_disagreements.csv", details, cols_rows)
    csv_write(out / "11_review_queue.csv", sorted_queue, cols_rows)
    save_figures(pairwise, lengths, out)
    manifest = {
        "analysis": "Post-hoc, fixed-model error analysis of 20 Newsgroups official test split",
        "data": "Non-empty official test documents with headers/footers/quotes removed",
        "test_documents": len(y_true), "num_categories": len(classes),
        "model_names": list(predictions), "seed": SEED,
        "paired_bootstrap_repeats": repeats,
        "posthoc_tests": "Exploratory exact McNemar on paired correctness with Holm correction",
        "review_queue_examples": len(sorted_queue),
        "disagreements_or_errors": len(details),
        "accuracy_by_length_is_descriptive": True,
        "do_not_retrain_using_test_findings": True,
        "licensing_note": "Inspect source dataset rights before publishing quoted documents",
    }
    # Completion marker last; partial failed output should not be considered final.
    (out / "11_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest, pairwise


def offline_self_test():
    # Three synthetic classes, all represented; model errors are deliberately distinct.
    truth = np.array([0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2])
    classes = ["technology", "space", "sports"]
    models = ("TF-IDF + Linear SVM", "MiniLM + Logistic Regression", "Fine-tuned DistilBERT")
    preds = {
        models[0]: np.array([0,1,2,0,0,2,0,1,0,0,1,2,0,1,2,0,1,2]),
        models[1]: np.array([0,1,2,0,1,2,0,1,2,0,1,2,1,1,2,0,1,2]),
        models[2]: np.array([0,1,2,0,1,2,0,2,2,0,1,2,0,1,1,0,1,2]),
    }
    docs = ["word " * (i * 15 + 3) for i in range(len(truth))]
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory)
        manifest, pairs = generate_analysis(docs, truth, preds, classes, out, repeats=50)
        assert len(pairs) == 3
        assert manifest["test_documents"] == len(truth)
        assert (out / "11_pairwise_statistics.csv").is_file()
        assert (out / "11_top_confusions.csv").is_file()
        assert (out / "11_review_queue.csv").is_file()
        assert (out / "11_bootstrap_differences.png").is_file()
        assert (out / "11_error_by_length.png").is_file()
        with (out / "11_pairwise_statistics.csv").open(encoding="utf-8") as f:
            r = list(csv.DictReader(f))
            assert 0 <= float(r[0]["mcnemar_holm_p"]) <= 1
        assert text_length_bucket("a " * 51) == "medium_51_200"
        assert text_length_bucket("a " * 201) == "long_201_plus"
        # Deliberate checksum mismatch must be caught.
        comp = load_previous_module()
        prior = {
            "classes": classes,
            "dataset": "20 Newsgroups; official test partition",
            "models": [{"model": n, "test_documents": len(truth),
                        "accuracy": 0.0, "macro_f1": 0.0,
                        "macro_precision": 0.0, "macro_recall": 0.0, "weighted_f1": 0.0}
                       for n in models],
        }
        try:
            score_against_prior(truth, preds, classes, prior, comp)
        except ValueError:
            pass
        else:
            raise AssertionError("Mismatch with old test results went undetected")
    print("SELF-TEST PASSED: error groups, model comparisons, exact tests, "
          "Holm correction, plots, CSVs, and mismatch protection.")
    print("No real pretrained model, official test labels or downloads were used.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Run synthetic offline self-test")
    parser.add_argument("--bootstrap-repeats", type=int, default=BOOTSTRAP_REPEATS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.self_test:
        offline_self_test()
        return
    if min(args.bootstrap_repeats, args.batch_size, args.embedding_batch_size) < 1:
        parser.error("Batch sizes and bootstrap repetitions must be positive")
    if (OUT_DIR / "11_analysis_manifest.json").is_file():
        raise FileExistsError(
            "Installment 11 analysis already exists. Reuse it instead of "
            "repeatedly inspecting the official test set."
        )
    comp = load_previous_module()
    prior_path = IN_DIR / "09_test_scores.json"
    if not prior_path.is_file():
        raise FileNotFoundError(
            "Run the completed final comparison first: "
            "python 09_compare_models.py\n" + str(prior_path)
        )
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    texts, ys, labels = comp.load_test_data()
    saved_predictions = IN_DIR / "09_test_predictions.csv"
    if saved_predictions.is_file():
        print("Using saved row-level final predictions; no repeated model inference.")
        with saved_predictions.open(encoding="utf-8", newline="") as fh:
            prediction_rows = list(csv.DictReader(fh))
        if len(prediction_rows) != len(ys):
            raise ValueError("Predictions CSV and official test split have different lengths")
        if any(int(r["row_index"]) != i or int(r["actual_id"]) != int(ys[i])
               for i, r in enumerate(prediction_rows)):
            raise ValueError("Predictions CSV labels/order disagree with official test split")
        predictions = {
            name: np.asarray([int(r[field]) for r in prediction_rows], dtype=np.int64)
            for name, field in zip(comp.MODEL_NAMES,
                ("svm_predicted_id", "minilm_predicted_id", "distilbert_predicted_id"))
        }
    else:
        # Legacy outputs from earlier versions: reproduce fixed inference once.
        print("Earlier comparison has no prediction cache; reproducing fixed inference.")
        paths = comp.find_inputs()
        svm, _, _ = comp.predict_svm(texts, labels, paths["svm"])
        mini, _, _ = comp.predict_embeddings(
            texts, labels, paths["embedding"], paths["embedding_manifest"],
            args.embedding_batch_size)
        dist, _, _ = comp.predict_transformer(
            texts, labels, paths["transformer_model"],
            paths["transformer_metadata"], args.batch_size)
        predictions = dict(zip(comp.MODEL_NAMES, (svm, mini, dist)))
    # Verify BOTH aggregate scores and all 3 complete confusion matrices.
    score_against_prior(ys, predictions, labels, prior, comp)
    print("Verified: scores and confusion matrices exactly match Installment 9.")
    manifest, comparisons = generate_analysis(
        texts, ys, predictions, labels, OUT_DIR,
        repeats=args.bootstrap_repeats)
    print(f"Analyzed {manifest['test_documents']} documents; "
          f"{manifest['disagreements_or_errors']} with errors/disagreement.")
    for comparison in comparisons:
        print(f"{comparison['model_b']} - {comparison['model_a']}: "
              f"delta macro-F1={comparison['delta_macro_f1_b_minus_a']:+.4f}; "
              f"bootstrap 95% CI ["
              f"{comparison['bootstrap_95pct_low']:+.4f}, "
              f"{comparison['bootstrap_95pct_high']:+.4f}]")
    print("Saved reports:", OUT_DIR)
    print("Do not use these test-set error analyses for model tuning.")


if __name__ == "__main__":
    main()

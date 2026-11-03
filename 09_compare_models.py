"""Installment 9 — final, one-time held-out evaluation of the three NLP models.

Prerequisites (run FULL, not quick/smoke modes):
    python 04_optimize_svm.py
    python 06_optimize_embeddings.py
    python 08_optimize_distilbert.py

Run:
    python -m pip install scikit-learn numpy matplotlib joblib sentence-transformers torch transformers
    python 09_compare_models.py

Test the evaluation/reporting logic offline (no downloads / model checkpoints):
    python 09_compare_models.py --self-test

Design:
* All configurations were selected on VALIDATION data in earlier installments.
* Only now is the official TEST partition (previously used only for descriptive counts) loaded.
* No tuning or retraining with the TEST data.
* Aborts before loading test examples if any required checkpoint is absent.
* Uses a validation-selected 100%-data quick-mode DistilBERT checkpoint only
  when the complete-grid checkpoint is missing; reports the abbreviated search.
* Refuses an accidental second official test run unless --overwrite is used.
* Saves measured numbers, confusion matrices, and paired bootstrap intervals.

WARNING: Only load .joblib/.pt model files created by you or trusted sources.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score,
)

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs" / "09_final_comparison"
SEED = 42
REMOVE_PARTS = ("headers", "footers", "quotes")
MODEL_NAMES = ("TF-IDF + Linear SVM", "MiniLM + Logistic Regression", "Fine-tuned DistilBERT")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def find_inputs(root=ROOT):
    """Require artifacts from the actual FULL experiment, not a smoke/quick run."""
    root = Path(root)
    paths = {
        "svm": root / "models" / "04_selected_svm.joblib",
        "embedding": root / "models" / "06_selected_embedding_classifiers.joblib",
        "embedding_manifest": root / "outputs" / "06_best_configuration.json",
    }
    absent = [str(p) for p in paths.values() if not p.is_file()]
    if absent:
        raise FileNotFoundError(
            "Selected classifier artifacts missing (do not retrain automatically):\n  "
            + "\n  ".join(absent)
        )

    # Either result is valid for ONE final held-out test if it was preselected
    # with validation data. Prefer the complete grid if available. The quick
    # experiment DID run a 100%-data 3-epoch checkpoint; its search grid was
    # only smaller, and that limitation must be disclosed.
    missing = []
    for mode in ("08_full", "08_quick_EXPLORATORY"):
        manifest_path = root / "outputs" / mode / "08_selected_configuration.json"
        if not manifest_path.is_file():
            missing.append(str(manifest_path))
            continue
        manifest = read_json(manifest_path)
        if manifest.get("research_mode") != mode:
            raise ValueError(f"Transformer manifest mode mismatch: {manifest_path}")
        from pathlib import PureWindowsPath
        saved_location = str(manifest["selected_full_model"])
        model_tag = PureWindowsPath(saved_location).name
        fallback_model = root / "outputs" / mode / "models" / model_tag
        configured = Path(saved_location)
        model_path = configured if configured.is_dir() else fallback_model
        if not (model_path / "config.json").is_file() or not (
            (model_path / "model.safetensors").is_file() or
            (model_path / "pytorch_model.bin").is_file()
        ):
            missing.append(str(model_path))
            continue
        paths["transformer_manifest"] = manifest_path
        paths["transformer_model"] = model_path
        paths["transformer_metadata"] = manifest
        paths["transformer_protocol"] = mode
        return paths
    raise FileNotFoundError(
        "Neither the complete nor the exploratory 100%-data DistilBERT "
        "checkpoint was found:\n  " + "\n  ".join(missing)
    )


def load_test_data():
    """Exactly the filtered official TEST partition, no reshuffling or relabeling."""
    raw = fetch_20newsgroups(
        subset="test", remove=REMOVE_PARTS, random_state=SEED
    )
    pairs = [(text, int(label)) for text, label in zip(raw.data, raw.target)
             if text.strip()]
    if not pairs:
        raise ValueError("No nonempty test documents.")
    xs, ys = zip(*pairs)
    return list(xs), np.asarray(ys, dtype=np.int64), list(raw.target_names)


def check_class_names(model_names, test_names, which):
    if list(model_names) != list(test_names):
        raise ValueError(
            f"{which}: label order disagrees with official test-set categories. "
            "Refusing to produce invalid test scores."
        )


def predict_svm(texts, classes, path):
    artifact = joblib.load(path)
    check_class_names(artifact["class_names"], classes, "SVM")
    begin = perf_counter()
    pred = np.asarray(artifact["pipeline"].predict(texts), dtype=np.int64)
    seconds = perf_counter() - begin
    return pred, seconds, {"representation": "TF-IDF vectorizer + optimized LinearSVC"}


def choose_embedding_method(summary, models):
    """Select head/head+tail based ONLY on earlier validation metrics."""
    scores = summary.get("selected", {})
    candidates = [rep for rep in ("head_only", "head_tail_mean")
                  if rep in scores and rep in models]
    if not candidates:
        raise ValueError("No matching embedding method in manifest and model artifact.")
    # First candidate wins an exact tie, so head-only is preferred on ties.
    rep = max(candidates, key=lambda key: (
        float(scores[key]["macro_f1"]), float(scores[key]["accuracy"])
    ))
    return rep


def compute_last_windows(texts, encoder):
    """Replicate the last-token-window extraction from Installment 6.

    The final window is independently encoded, using precisely the same
    tokenization and decoding rule used to train head_tail_mean.
    """
    limit = int(encoder.max_seq_length)
    tokenizer = encoder.tokenizer
    usable = limit - int(tokenizer.num_special_tokens_to_add(pair=False))
    if usable <= 0:
        raise ValueError("Invalid encoder token budget.")
    locations, last_windows = [], []
    for index, txt in enumerate(texts):
        ids = tokenizer.encode(txt, add_special_tokens=False, truncation=False)
        if len(ids) > usable:
            locations.append(index)
            last_windows.append(tokenizer.decode(
                ids[-usable:], skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            ))
    return np.asarray(locations, dtype=np.int64), last_windows


def normalize_vectors(arr):
    arr = np.asarray(arr, dtype=np.float32)
    return arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)


def predict_embeddings(texts, class_names, artifact_path, summary_path, batch_size):
    artifact = joblib.load(artifact_path)
    summary = read_json(summary_path)
    check_class_names(artifact["class_names"], class_names, "MiniLM")
    rep = choose_embedding_method(summary, artifact["models"])
    info = artifact["models"][rep]
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError("Install: python -m pip install sentence-transformers") from error

    encoder = SentenceTransformer(artifact["encoder_name"])
    if (summary.get("max_seq_length") is not None and
            int(summary["max_seq_length"]) != int(encoder.max_seq_length)):
        raise RuntimeError("Encoder maximum sequence length differs from previous runs.")

    begin = perf_counter()  # Includes ALL document embedding generation.
    first = encoder.encode(
        texts, batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True,
    ).astype(np.float32)
    if rep == "head_tail_mean":
        indices, tails = compute_last_windows(texts, encoder)
        if len(indices):
            tail_emb = encoder.encode(
                tails, batch_size=batch_size, convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=True,
            ).astype(np.float32)
            if tail_emb.shape != first[indices].shape:
                raise ValueError("Unexpected tail embedding dimensions.")
            first[indices] = normalize_vectors(first[indices] + tail_emb)
    pred = np.asarray(info["classifier"].predict(first), dtype=np.int64)
    seconds = perf_counter() - begin
    return pred, seconds, {
        "representation": rep,
        "encoder": artifact["encoder_name"],
        "classifier_C": float(info["C"]),
        "token_max_length": int(encoder.max_seq_length),
    }


def predict_transformer(texts, class_names, path, manifest, batch_size):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError as error:
        raise RuntimeError("Install: python -m pip install torch transformers") from error

    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(path), local_files_only=True
    )
    if model.config.num_labels != len(class_names):
        raise ValueError("Transformer number of classes does not match the test set.")
    actual_labels = [model.config.id2label.get(i) for i in range(len(class_names))]
    check_class_names(actual_labels, class_names, "DistilBERT")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    max_length = int(manifest["selected_max_length"])

    results = []
    start = perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(texts), batch_size):
            items = tokenizer(
                texts[offset:offset + batch_size], padding=True,
                truncation=True, max_length=max_length, return_tensors="pt"
            )
            batch = {key: tensor.to(device) for key, tensor in items.items()}
            logits = model(**batch).logits
            results.extend(logits.argmax(dim=-1).cpu().tolist())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = perf_counter() - start
    return np.asarray(results, dtype=np.int64), elapsed, {
        "representation": "Fine-tuned Transformer",
        "maximum_tokens": max_length,
        "device": str(device),
        "selected_learning_rate": float(manifest["selected_learning_rate"]),
    }


def score_predictions(y_true, y_pred, class_names, inference_seconds):
    true, pred = np.asarray(y_true), np.asarray(y_pred)
    num_classes = len(class_names)
    if true.ndim != 1 or pred.shape != true.shape or len(true) == 0:
        raise ValueError("Prediction array does not match test labels.")
    if not np.isin(pred, np.arange(num_classes)).all():
        raise ValueError("Predicted class index outside the expected 20 categories.")
    labels = list(range(num_classes))
    return {
        "test_documents": len(true),
        "accuracy": float(accuracy_score(true, pred)),
        "macro_precision": float(precision_score(
            true, pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(
            true, pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(
            true, pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(
            true, pred, labels=labels, average="weighted", zero_division=0)),
        "inference_seconds": float(inference_seconds),
        "documents_per_second": float(len(true) / max(inference_seconds, 1e-10)),
    }


def paired_bootstrap_macro_f1(y_true, predictions, repetitions=1000, seed=SEED):
    """Paired 95% percentile CI for model-vs-SVM macro-F1 differences.

    Bootstrap is descriptive and exploratory; not a preregistered significance
    test, and does not correct for multiple hypotheses.
    """
    rng = np.random.default_rng(seed)
    ys = np.asarray(y_true)
    all_labels = list(range(len(np.unique(ys))))
    baseline = np.asarray(predictions[MODEL_NAMES[0]])
    delta_rows = []
    for other_name in MODEL_NAMES[1:]:
        other = np.asarray(predictions[other_name])
        diffs = np.empty(repetitions, dtype=float)
        for i in range(repetitions):
            indices = rng.integers(0, len(ys), size=len(ys))
            other_f1 = f1_score(ys[indices], other[indices], labels=all_labels,
                                average="macro", zero_division=0)
            base_f1 = f1_score(ys[indices], baseline[indices], labels=all_labels,
                               average="macro", zero_division=0)
            diffs[i] = other_f1 - base_f1
        delta_rows.append({
            "model": other_name, "comparison": "model minus TF-IDF + Linear SVM",
            "delta_macro_f1": float(f1_score(ys, other, labels=all_labels,
                                              average="macro", zero_division=0) -
                                     f1_score(ys, baseline, labels=all_labels,
                                              average="macro", zero_division=0)),
            "bootstrap_ci_low": float(np.quantile(diffs, 0.025)),
            "bootstrap_ci_high": float(np.quantile(diffs, 0.975)),
            "replicates": repetitions,
        })
    return delta_rows


def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_comparison(y_true, predictions, timings, details, classes, out_dir,
                    *, bootstrap_repeats=1000):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = list(range(len(classes)))
    rows = []
    for name in MODEL_NAMES:
        pred = np.asarray(predictions[name])
        row = {"model": name, **score_predictions(y_true, pred, classes, timings[name])}
        rows.append(row)

        individual = classification_report(
            y_true, pred, labels=labels, target_names=classes,
            output_dict=True, zero_division=0
        )
        perclass = [
            {"category": c, "precision": individual[c]["precision"],
             "recall": individual[c]["recall"],
             "f1": individual[c]["f1-score"], "support": individual[c]["support"]}
            for c in classes
        ]
        slug = ("svm", "minilm", "distilbert")[MODEL_NAMES.index(name)]
        write_csv(out_dir / f"09_{slug}_per_class.csv", perclass,
                  ["category", "precision", "recall", "f1", "support"])
        matrix = confusion_matrix(y_true, pred, labels=labels)
        write_csv(out_dir / f"09_{slug}_confusion_matrix.csv",
                  [dict(actual_category=classes[i],
                        **{classes[j]: int(matrix[i,j]) for j in labels})
                   for i in labels], ["actual_category", *classes])
        fig, ax = plt.subplots(figsize=(11, 10))
        im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
        ax.set_xticks(labels, classes, rotation=90, fontsize=7)
        ax.set_yticks(labels, classes, fontsize=7)
        ax.set(xlabel="Predicted category", ylabel="True category",
               title=f"Held-out test confusion matrix — {name}")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out_dir / f"09_{slug}_confusion_matrix.png", dpi=170)
        plt.close(fig)

    write_csv(out_dir / "09_test_model_comparison.csv", rows,
              ["model", "test_documents", "accuracy", "macro_precision", "macro_recall",
               "macro_f1", "weighted_f1", "inference_seconds", "documents_per_second"])

    # Cache row-aligned predictions once. Installment 11 can analyze mistakes
    # without running expensive encoder inference a second time. Labels only:
    # the user's copies of the original dataset are not republished here.
    prediction_rows = [
        {"row_index": i, "actual_id": int(y_true[i]),
         "svm_predicted_id": int(predictions[MODEL_NAMES[0]][i]),
         "minilm_predicted_id": int(predictions[MODEL_NAMES[1]][i]),
         "distilbert_predicted_id": int(predictions[MODEL_NAMES[2]][i])}
        for i in range(len(y_true))
    ]
    write_csv(out_dir / "09_test_predictions.csv", prediction_rows,
              ["row_index", "actual_id", "svm_predicted_id",
               "minilm_predicted_id", "distilbert_predicted_id"])

    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.arange(len(MODEL_NAMES))
    width = .36
    ax.bar(xs - width / 2, [r["macro_f1"] for r in rows], width, label="Macro-F1")
    ax.bar(xs + width / 2, [r["accuracy"] for r in rows], width, label="Accuracy")
    ax.set_xticks(xs, ["SVM", "MiniLM", "DistilBERT"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Final held-out test performance (same documents)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "09_model_comparison.png", dpi=170)
    plt.close(fig)

    intervals = paired_bootstrap_macro_f1(
        y_true, predictions, repetitions=bootstrap_repeats
    )
    write_csv(out_dir / "09_paired_bootstrap.csv", intervals,
              ["model", "comparison", "delta_macro_f1",
               "bootstrap_ci_low", "bootstrap_ci_high", "replicates"])
    result = {
        "dataset": "20 Newsgroups; official test partition",
        "removed_fields": list(REMOVE_PARTS),
        "empty_documents_excluded": True,
        "seed": SEED,
        "classes": classes,
        "models": rows,
        "model_details": details,
        "paired_bootstrap_deltas_vs_svm": intervals,
        "cautions": [
            "Model hyperparameters were selected exclusively with the validation set.",
            "This test set is held-out; do not tune models on these results.",
            "Timing excludes checkpoint loading but includes tokenization, feature extraction, and prediction.",
            "Measured speed is hardware- and batch-size-dependent; these are not controlled hardware benchmarks.",
            "Paired bootstrap intervals are exploratory; no multiple-comparison correction.",
            "All approaches truncate or summarize long texts differently; use error analysis to interpret gaps.",
        ],
    }
    # The final JSON is the completion marker; do not write it until every model
    # and report stage succeeds.
    (out_dir / "09_test_scores.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows


def offline_self_test():
    """Exercise inference-independent scoring and artifacts without downloading."""
    import tempfile
    names = ["cat_A", "cat_B", "cat_C"]
    truth = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    predictions = {
        MODEL_NAMES[0]: np.array([0,1,2,0,1,2,0,1,0,0,1,2]),
        MODEL_NAMES[1]: np.array([0,1,2,0,1,2,0,1,2,0,1,2]),
        MODEL_NAMES[2]: np.array([0,1,2,0,2,2,0,1,2,0,1,2]),
    }
    assert choose_embedding_method(
        {"selected": {"head_only": {"macro_f1": 0.7, "accuracy": 0.7},
                      "head_tail_mean": {"macro_f1": 0.8, "accuracy": 0.8}}},
        {"head_only": {}, "head_tail_mean": {}}
    ) == "head_tail_mean"
    assert score_predictions(truth, predictions[MODEL_NAMES[1]], names, 0.5)["macro_f1"] == 1.0
    try:
        check_class_names(["B", "A"], ["A", "B"], "test")
    except ValueError:
        pass
    else:
        raise AssertionError("Label order must be validated")
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        result = save_comparison(
            truth, predictions, {k: 1.0 for k in MODEL_NAMES},
            {k: {"mock": True} for k in MODEL_NAMES}, names, folder,
            bootstrap_repeats=30,
        )
        assert len(result) == 3
        assert len(list(folder.glob("09_*_confusion_matrix.csv"))) == 3
        assert (folder / "09_test_scores.json").exists()
        assert (folder / "09_paired_bootstrap.csv").exists()
    print("SELF-TEST PASSED: validation-based choice, labels, scores, plots, export")
    print("No real model was loaded and the official test set was not accessed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true",
                        help="Explicitly repeat final test evaluation. Not recommended.")
    args = parser.parse_args()
    if args.self_test:
        offline_self_test()
        return
    if min(args.batch_size, args.embedding_batch_size, args.bootstrap_repeats) < 1:
        parser.error("Batch sizes and bootstrap repeats must be positive.")
    result_path = OUTPUT / "09_test_scores.json"
    if result_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Official test results already exist at {result_path}. "
            "Do not repeatedly evaluate the test set for model tuning."
        )
    paths = find_inputs()  # Check model availability BEFORE accessing held-out data.
    X_test, y_test, class_names = load_test_data()
    print(f"Official held-out test: {len(X_test)} nonempty documents, "
          f"{len(class_names)} classes. No training or tuning.")
    print("Evaluating preselected SVM...")
    svm_pred, svm_secs, svm_info = predict_svm(X_test, class_names, paths["svm"])
    print("Evaluating preselected embedding classifier...")
    emb_pred, emb_secs, emb_info = predict_embeddings(
        X_test, class_names, paths["embedding"], paths["embedding_manifest"],
        args.embedding_batch_size)
    print(f"Evaluating 100%-training-data DistilBERT; tuning protocol: "
          f"{paths['transformer_protocol']}...")
    if paths['transformer_protocol'] != "08_full":
        print("IMPORTANT: Transformer hyperparameter search was abbreviated "
              "(--quick), but the selected model trained for 3 epochs on 100% "
              "of training data. Record this limitation in the paper.")
    tr_pred, tr_secs, tr_info = predict_transformer(
        X_test, class_names, paths["transformer_model"],
        paths["transformer_metadata"], args.batch_size)
    predictions = dict(zip(MODEL_NAMES, (svm_pred, emb_pred, tr_pred)))
    times = dict(zip(MODEL_NAMES, (svm_secs, emb_secs, tr_secs)))
    tr_info["hyperparameter_selection_protocol"] = paths["transformer_protocol"]
    tr_info["screening_scope"] = ("abbreviated quick exploratory grid"
        if paths["transformer_protocol"] != "08_full" else "complete prespecified grid")
    details = dict(zip(MODEL_NAMES, (svm_info, emb_info, tr_info)))
    rows = save_comparison(
        y_test, predictions, times, details, class_names, OUTPUT,
        bootstrap_repeats=args.bootstrap_repeats)
    print("\nFINAL HELD-OUT TEST RESULTS (macro-F1 is primary):")
    for row in rows:
        print(f"  {row['model']:<31} macro-F1={row['macro_f1']:.4f} "
              f"accuracy={row['accuracy']:.4f} "
              f"prediction={row['inference_seconds']:.1f}s")
    print("\nReports and charts:", OUTPUT)
    print("Do not modify/tune hyperparameters after viewing final test performance.")


if __name__ == "__main__":
    main()

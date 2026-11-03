"""Installment 3 — TF-IDF + Linear SVM baseline for 20 Newsgroups.

Run from the same folder as 02_explore_data.py:
    python 03_train_svm_baseline.py

Requires: scikit-learn, numpy, matplotlib, joblib

Reproduces exactly the training/validation split from Installment 2.
The official test set is deliberately NOT accessed in this installment.

Outputs:
  models/03_tfidf_svm.joblib
  outputs/03_validation_metrics.json
  outputs/03_validation_classification_report.csv
  outputs/03_validation_confusion_matrix.csv
  outputs/03_validation_confusion_matrix.png
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"


def load_training_and_validation():
    """Match the preparation used in 01/02; never download/read the test split."""
    raw = fetch_20newsgroups(
        subset="train",
        remove=("headers", "footers", "quotes"),
        random_state=RANDOM_STATE,
    )
    pairs = [(text, int(label)) for text, label in zip(raw.data, raw.target)
             if text.strip()]
    if not pairs:
        raise ValueError("No nonempty training documents found.")
    documents, targets = zip(*pairs)
    X_train, X_val, y_train, y_val = train_test_split(
        documents, targets,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=targets,
    )
    return X_train, X_val, y_train, y_val, list(raw.target_names)


def build_baseline():
    # Fitting TF-IDF inside a pipeline prevents learning its vocabulary/IDF
    # from the validation documents (data leakage).
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            max_features=100_000,
            sublinear_tf=True,
            dtype=np.float32,
        )),
        ("classifier", LinearSVC(C=1.0, max_iter=5000, random_state=RANDOM_STATE)),
    ])


def save_classification_report(y_true, y_pred, labels, class_names):
    report = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    path = OUTPUT_DIR / "03_validation_classification_report.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_or_summary", "precision", "recall", "f1_score", "support"])
        for label in class_names + ["macro avg", "weighted avg"]:
            d = report[label]
            writer.writerow([label, d["precision"], d["recall"], d["f1-score"], d["support"]])
    return path


def save_confusion_matrix(y_true, y_pred, labels, class_names):
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    path = OUTPUT_DIR / "03_validation_confusion_matrix.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["actual \\ predicted"] + class_names)
        for class_name, row in zip(class_names, matrix):
            writer.writerow([class_name] + row.tolist())

    figure, axis = plt.subplots(figsize=(14, 12))
    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    ).plot(ax=axis, cmap="Blues", xticks_rotation=90,
           include_values=False, colorbar=True)
    axis.set_title("TF-IDF + Linear SVM — Validation Confusion Matrix")
    figure.tight_layout()
    figure_path = OUTPUT_DIR / "03_validation_confusion_matrix.png"
    figure.savefig(figure_path, dpi=170)
    plt.close(figure)
    return path, figure_path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading training documents (first run may require downloading data)...")
    X_train, X_val, y_train, y_val, class_names = load_training_and_validation()
    labels = list(range(len(class_names)))
    print(f"Training: {len(X_train):,} | Validation: {len(X_val):,} | Classes: {len(labels)}")

    pipeline = build_baseline()
    start = perf_counter()
    pipeline.fit(X_train, y_train)
    training_seconds = perf_counter() - start

    start = perf_counter()
    predictions = pipeline.predict(X_val)
    prediction_seconds = perf_counter() - start

    result = {
        "model": "TF-IDF (word unigrams + bigrams) + LinearSVC(C=1.0)",
        "dataset": "20 Newsgroups — remove headers, footers, quotes",
        "random_seed": RANDOM_STATE,
        "training_documents": len(X_train),
        "validation_documents": len(X_val),
        "classes": len(class_names),
        "vocabulary_features": len(pipeline.named_steps["tfidf"].vocabulary_),
        "accuracy": float(accuracy_score(y_val, predictions)),
        "macro_precision": float(precision_score(y_val, predictions, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_val, predictions, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_val, predictions, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_val, predictions, labels=labels, average="weighted", zero_division=0)),
        "training_seconds": round(training_seconds, 3),
        "validation_prediction_seconds": round(prediction_seconds, 3),
    }

    with (OUTPUT_DIR / "03_validation_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    save_classification_report(y_val, predictions, labels, class_names)
    save_confusion_matrix(y_val, predictions, labels, class_names)
    model_path = MODEL_DIR / "03_tfidf_svm.joblib"
    # Only open this file again if it was produced by you / a trusted source.
    joblib.dump({"pipeline": pipeline, "class_names": class_names}, model_path)

    print("\nVALIDATION METRICS (held-out validation set, NOT the test set)")
    for metric in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"):
        print(f"  {metric:18s}: {result[metric]:.4f}")
    print(f"  training time       : {training_seconds:.2f} seconds")
    print(f"  model features      : {result['vocabulary_features']:,}")

    example_text = "NASA launched a new spacecraft into orbit to study the planets."
    predicted_class = int(pipeline.predict([example_text])[0])
    print("\nExample prediction (illustrative input):")
    print("  Input: ", example_text)
    print("  Topic: ", class_names[predicted_class])

    print("\nSaved outputs:")
    for location in sorted(OUTPUT_DIR.glob("03_*")):
        print(" -", location.relative_to(ROOT))
    print(" -", model_path.relative_to(ROOT))
    print("\nNext: compare training sizes and preprocessing variants (Installment 4).")


if __name__ == "__main__":
    main()

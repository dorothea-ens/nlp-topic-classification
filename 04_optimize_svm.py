"""Installment 4 — Hyperparameter search and training-size experiments.

Continue after Installments 1–3:
    pip install scikit-learn numpy matplotlib joblib
    python 04_optimize_svm.py

Uses the exact 20 Newsgroups TRAIN / VALIDATION split of Installments 1–3.
The official test partition is NEVER loaded, touched, or used for tuning.

Experiments
-----------
A. Compare three text-vectorization approaches x three SVM C values on
   the entire training partition, ranking by validation macro-F1.
B. Train the selected configuration on nested, class-stratified fractions
   of the training partition (10%, 25%, 50%, 100%).

Outputs in outputs/:
    04_hyperparameter_search.csv
    04_training_size_results.csv
    04_best_configuration.json
    04_hyperparameter_comparison.png
    04_learning_curve.png
and models/04_selected_svm.joblib.

NOTE: Validation is used to select settings. To avoid optimistic selection
bias, final comparative claims must use the untouched official test set
only after all model choices are fixed in later installments.
"""

from __future__ import annotations

import csv
import json
import warnings
from collections import Counter
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"

# Only the training split is sampled. Fractions are nested so smaller
# training subsets are contained in larger ones.
TRAIN_FRACTIONS = (0.10, 0.25, 0.50, 1.0)
C_VALUES = (0.1, 1.0, 10.0)

# Comparable variants; the second is exactly the feature setup used in
# Installment 3 (which used C=1.0).
VECTORIZERS = {
    "unigram_no_stopwords": {"ngram_range": (1, 1), "stop_words": None},
    "unigram_bigram_english_stopwords": {
        "ngram_range": (1, 2), "stop_words": "english"
    },
    "unigram_bigram_no_stopwords": {"ngram_range": (1, 2), "stop_words": None},
}


def load_training_validation():
    """Reproduce exactly the nonempty filtering and split from files 01–03."""
    raw = fetch_20newsgroups(
        subset="train",
        remove=("headers", "footers", "quotes"),
        random_state=RANDOM_STATE,
    )
    pairs = [
        (text, int(label))
        for text, label in zip(raw.data, raw.target)
        if text.strip()
    ]
    if not pairs:
        raise ValueError("No nonempty documents after metadata removal.")
    X, y = zip(*pairs)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    return list(X_train), list(X_val), np.array(y_train), np.array(y_val), list(raw.target_names)


def build_pipeline(vectorizer_name, c_value):
    """Vectorization is FIT separately for each training subset (no leakage)."""
    configuration = VECTORIZERS[vectorizer_name]
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words=configuration["stop_words"],
            ngram_range=configuration["ngram_range"],
            min_df=2,
            max_df=0.95,
            max_features=100_000,
            sublinear_tf=True,
            dtype=np.float32,
        )),
        ("classifier", LinearSVC(
            C=c_value,
            max_iter=5000,
            random_state=RANDOM_STATE,
        )),
    ])


def metrics(y_true, predictions, labels):
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_precision": float(precision_score(
            y_true, predictions, average="macro", labels=labels, zero_division=0)),
        "macro_recall": float(recall_score(
            y_true, predictions, average="macro", labels=labels, zero_division=0)),
        "macro_f1": float(f1_score(
            y_true, predictions, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(
            y_true, predictions, average="weighted", labels=labels, zero_division=0)),
    }


def write_csv(path, rows, column_names):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=column_names)
        writer.writeheader()
        writer.writerows(rows)


def nested_stratified_indices(y_train):
    """Keep one randomly ordered pool per class to ensure nested subsets.

    At every fraction, take the same prefix from each class. The method
    preserves classes at small sizes, uses only TRAIN labels, and makes
    class proportions approximately constant across sizes.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    pools = {}
    for label in np.unique(y_train):
        positions = np.flatnonzero(y_train == label)
        pools[int(label)] = rng.permutation(positions)

    subsets = {}
    for fraction in TRAIN_FRACTIONS:
        indices = []
        for label in sorted(pools):
            candidates = pools[label]
            count = max(1, int(round(len(candidates) * fraction)))
            indices.extend(candidates[:count].tolist())
        subsets[fraction] = np.array(sorted(indices), dtype=int)

    for smaller, larger in zip(TRAIN_FRACTIONS, TRAIN_FRACTIONS[1:]):
        assert set(subsets[smaller]).issubset(set(subsets[larger]))
    return subsets


def plot_parameter_search(rows):
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in VECTORIZERS:
        subset = sorted((r for r in rows if r["vectorizer"] == name),
                        key=lambda r: r["C"])
        ax.plot([r["C"] for r in subset],
                [r["macro_f1"] for r in subset],
                marker="o", label=name.replace("_", " "))
    ax.set_xscale("log")
    ax.set_xticks(C_VALUES)
    ax.set_xticklabels([str(c) for c in C_VALUES])
    ax.set_xlabel("SVM regularization parameter C")
    ax.set_ylabel("Validation macro-F1")
    ax.set_title("TF-IDF + LinearSVC: preprocessing and regularization")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_hyperparameter_comparison.png", dpi=180)
    plt.close(fig)


def plot_learning_curve(rows):
    fig, ax = plt.subplots(figsize=(9, 5))
    sizes = [r["training_documents"] for r in rows]
    ax.plot(sizes, [r["training_macro_f1"] for r in rows],
            marker="o", label="Training macro-F1")
    ax.plot(sizes, [r["validation_macro_f1"] for r in rows],
            marker="o", label="Validation macro-F1")
    ax.set_xlabel("Number of training documents")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1.02)
    ax.set_title("Learning curve: TF-IDF + selected LinearSVC")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_learning_curve.png", dpi=180)
    plt.close(fig)


def run_experiments(X_train, X_val, y_train, y_val, class_names):
    """Runs on provided splits. Also usable for synthetic smoke tests."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    X_train = list(X_train)
    X_val = list(X_val)
    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)
    labels = list(range(len(class_names)))

    print(f"Train={len(X_train):,}, validation={len(X_val):,}, classes={len(class_names)}")
    print("No official test data loaded.\n")
    search_rows = []
    best_model = None
    best_row = None

    # Stage A: select a configuration using the VALIDATION SET ONLY.
    for vectorizer_name in VECTORIZERS:
        for c_value in C_VALUES:
            estimator = build_pipeline(vectorizer_name, c_value)
            start = perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                estimator.fit(X_train, y_train)
            fit_seconds = perf_counter() - start
            for warning in caught:
                if issubclass(warning.category, ConvergenceWarning):
                    print(f"  WARNING: {vectorizer_name}, C={c_value}: {warning.message}")
            predictions = estimator.predict(X_val)
            row = {
                "vectorizer": vectorizer_name,
                "C": c_value,
                "training_documents": len(X_train),
                "validation_documents": len(X_val),
                "vocabulary_features": len(estimator.named_steps["tfidf"].vocabulary_),
                **metrics(y_val, predictions, labels),
                "training_seconds": round(fit_seconds, 3),
            }
            search_rows.append(row)
            print(f"  {vectorizer_name:35s} C={c_value:<4g} "
                  f"macro-F1={row['macro_f1']:.4f} "
                  f"accuracy={row['accuracy']:.4f} "
                  f"fit={fit_seconds:.1f}s")
            # Higher macro-F1, then accuracy; earlier candidate wins exact ties.
            if best_row is None or (row["macro_f1"], row["accuracy"]) > (
                best_row["macro_f1"], best_row["accuracy"]
            ):
                best_model = estimator
                best_row = row

    write_csv(
        OUTPUT_DIR / "04_hyperparameter_search.csv", search_rows,
        ["vectorizer", "C", "training_documents", "validation_documents",
         "vocabulary_features", "accuracy", "macro_precision", "macro_recall",
         "macro_f1", "weighted_f1", "training_seconds"],
    )
    plot_parameter_search(search_rows)

    print("\nSelected by validation macro-F1:")
    print(f"  {best_row['vectorizer']}, C={best_row['C']}, "
          f"validation macro-F1={best_row['macro_f1']:.4f}")

    joblib.dump(
        {"pipeline": best_model, "class_names": class_names,
         "selection_metric": "validation_macro_f1"},
        MODEL_DIR / "04_selected_svm.joblib"
    )

    # Stage B: use SAME best configuration for 4 nested training sizes.
    # The vectorizer is refitted within each respective subset.
    subsets = nested_stratified_indices(y_train)
    size_rows = []
    for fraction, indices in subsets.items():
        X_subset = [X_train[int(i)] for i in indices]
        y_subset = y_train[indices]
        estimator = build_pipeline(best_row["vectorizer"], best_row["C"])
        start = perf_counter()
        estimator.fit(X_subset, y_subset)
        fit_seconds = perf_counter() - start
        y_training_pred = estimator.predict(X_subset)
        y_val_pred = estimator.predict(X_val)
        val_scores = metrics(y_val, y_val_pred, labels)
        row = {
            "training_fraction": fraction,
            "training_documents": len(X_subset),
            "validation_documents": len(X_val),
            "min_class_count": min(Counter(y_subset).values()),
            "training_macro_f1": f1_score(
                y_subset, y_training_pred, average="macro", labels=labels,
                zero_division=0
            ),
            "validation_macro_f1": val_scores["macro_f1"],
            "validation_accuracy": val_scores["accuracy"],
            "validation_macro_precision": val_scores["macro_precision"],
            "validation_macro_recall": val_scores["macro_recall"],
            "validation_weighted_f1": val_scores["weighted_f1"],
            "vocabulary_features": len(estimator.named_steps["tfidf"].vocabulary_),
            "training_seconds": round(fit_seconds, 3),
        }
        size_rows.append(row)
        print(f"  {fraction:>4.0%}: n={len(X_subset):>5} "
              f"train macro-F1={row['training_macro_f1']:.4f} "
              f"val macro-F1={row['validation_macro_f1']:.4f}")

    write_csv(
        OUTPUT_DIR / "04_training_size_results.csv", size_rows,
        ["training_fraction", "training_documents", "validation_documents",
         "min_class_count", "training_macro_f1", "validation_macro_f1",
         "validation_accuracy", "validation_macro_precision",
         "validation_macro_recall", "validation_weighted_f1",
         "vocabulary_features", "training_seconds"],
    )
    plot_learning_curve(size_rows)

    configuration_summary = {
        "dataset": "20 Newsgroups",
        "metadata_removed": ["headers", "footers", "quotes"],
        "random_seed": RANDOM_STATE,
        "selection_criterion": "maximum validation macro-F1; accuracy breaks ties",
        "best_configuration": best_row,
        "vectorizer_parameters": {
            **VECTORIZERS[best_row["vectorizer"]],
            "min_df": 2, "max_df": 0.95,
            "max_features": 100_000,
            "sublinear_tf": True,
        },
        "training_fractions": list(TRAIN_FRACTIONS),
        "notes": [
            "The search and learning curve use the same held-out validation data.",
            "The official test set was not loaded or evaluated.",
            "Training subset sizes are nested and approximately class-stratified.",
            "This experiment selects settings for the TF-IDF + SVM baseline only.",
            "Final comparisons must use an untouched test set later.",
        ],
    }
    # JSON serializes tuples as lists; all other values here are plain types.
    with (OUTPUT_DIR / "04_best_configuration.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(configuration_summary, file, ensure_ascii=False, indent=2)

    print("\nSaved files:")
    for p in sorted(OUTPUT_DIR.glob("04_*")):
        print(" -", p.relative_to(ROOT))
    print(" -", (MODEL_DIR / "04_selected_svm.joblib").relative_to(ROOT))
    print("\nKeep the official test partition untouched until final evaluation.")
    return best_row, search_rows, size_rows


def main():
    X_train, X_val, y_train, y_val, class_names = load_training_validation()
    run_experiments(X_train, X_val, y_train, y_val, class_names)


if __name__ == "__main__":
    main()

"""Installment 5 — Frozen sentence embeddings + multiclass logistic regression.

Run in the project folder:
    python -m pip install sentence-transformers scikit-learn numpy matplotlib joblib
    python 05_train_embedding_classifier.py

Uses the same 20 Newsgroups training/validation split as Installments 1–4.
NEVER loads or evaluates the official test partition.

Outputs:
    cache/05_minilm_embeddings.npz
    models/05_embedding_logreg.joblib
    outputs/05_validation_metrics.json
    outputs/05_validation_classification_report.csv
    outputs/05_validation_confusion_matrix.csv
    outputs/05_validation_confusion_matrix.png

Important: all-MiniLM-L6-v2 is optimized for sentences/short paragraphs.
Its tokenizer truncates long documents to its maximum sequence length (256
word pieces with its usual configuration). This limitation is recorded and
will be investigated in Installment 6.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import warnings
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib
matplotlib.use("Agg")  # Enables image generation without a graphical desktop.
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegression
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

RANDOM_STATE = 42
ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
CACHE_DIR = ROOT / "cache"
CACHE_PATH = CACHE_DIR / "05_minilm_embeddings.npz"


def load_training_validation():
    """Exactly reproduce the splits in Installments 2, 3, and 4."""
    raw = fetch_20newsgroups(
        subset="train",
        remove=("headers", "footers", "quotes"),
        random_state=RANDOM_STATE,
    )
    nonempty = [
        (text, int(label)) for text, label in zip(raw.data, raw.target)
        if text.strip()
    ]
    if not nonempty:
        raise ValueError("The training set has no nonempty documents.")
    documents, labels = zip(*nonempty)
    X_train, X_val, y_train, y_val = train_test_split(
        documents, labels,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    return (
        list(X_train), list(X_val), np.asarray(y_train),
        np.asarray(y_val), list(raw.target_names),
    )


def data_fingerprint(X_train, X_val):
    """Invalidate cached embeddings if the text or order changes."""
    h = hashlib.sha256()
    h.update(ENCODER_NAME.encode("utf-8"))
    h.update(b"\nnormalized=True\n")
    for partition_name, documents in (("train", X_train), ("validation", X_val)):
        h.update(partition_name.encode("ascii"))
        for document in documents:
            data = document.encode("utf-8")
            h.update(len(data).to_bytes(8, byteorder="little"))
            h.update(data)
    return h.hexdigest()


def encode_or_load(X_train, X_val, *, batch_size=32, force=False):
    """Save significant CPU/GPU time when repeating later experiments."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fingerprint = data_fingerprint(X_train, X_val)

    if CACHE_PATH.is_file() and not force:
        try:
            with np.load(CACHE_PATH, allow_pickle=False) as data:
                if str(data["fingerprint"].item()) == fingerprint:
                    train_embeddings = np.asarray(data["train"], dtype=np.float32)
                    val_embeddings = np.asarray(data["validation"], dtype=np.float32)
                    max_seq_length = int(data["max_seq_length"].item())
                    if (train_embeddings.shape[0] == len(X_train)
                            and val_embeddings.shape[0] == len(X_val)
                            and train_embeddings.ndim == val_embeddings.ndim == 2
                            and train_embeddings.shape[1] == val_embeddings.shape[1]):
                        print("Reusing cached embeddings (no encoding required).")
                        return train_embeddings, val_embeddings, max_seq_length, 0.0, True
        except (OSError, ValueError, KeyError) as error:
            warnings.warn(f"Embedding cache could not be used: {error}")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "Install the encoder dependency first: "
            "python -m pip install sentence-transformers"
        ) from error

    print(f"Loading pretrained encoder: {ENCODER_NAME}")
    model = SentenceTransformer(ENCODER_NAME)
    max_seq_length = model.max_seq_length
    if max_seq_length is None:
        raise RuntimeError("The loaded embedding model did not expose a maximum sequence length.")
    max_seq_length = int(max_seq_length)
    print(f"Encoder maximum input: {max_seq_length} word-piece tokens")
    print("Longer documents are truncated. No stemming/stopword deletion is applied.")

    start = perf_counter()
    print("Encoding training documents...")
    train_embeddings = model.encode(
        X_train,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)
    print("Encoding validation documents...")
    val_embeddings = model.encode(
        X_val,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)
    encoding_seconds = perf_counter() - start
    if train_embeddings.ndim != 2 or val_embeddings.ndim != 2:
        raise ValueError("Encoder returned unexpectedly shaped embeddings.")

    np.savez_compressed(
        CACHE_PATH,
        train=train_embeddings,
        validation=val_embeddings,
        fingerprint=np.array(fingerprint),
        max_seq_length=np.array(max_seq_length),
    )
    return train_embeddings, val_embeddings, max_seq_length, encoding_seconds, False


def validation_metrics(y_true, predictions, class_ids):
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_precision": float(precision_score(
            y_true, predictions, labels=class_ids, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(
            y_true, predictions, labels=class_ids, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(
            y_true, predictions, labels=class_ids, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(
            y_true, predictions, labels=class_ids, average="weighted", zero_division=0)),
    }


def save_detailed_evaluation(y_val, predictions, class_names):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = list(range(len(class_names)))
    report = classification_report(
        y_val, predictions, labels=labels,
        target_names=class_names, output_dict=True, zero_division=0,
    )
    report_path = OUTPUT_DIR / "05_validation_classification_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_or_summary", "precision", "recall", "f1_score", "support"])
        for label in class_names + ["macro avg", "weighted avg"]:
            row = report[label]
            writer.writerow([
                label, row["precision"], row["recall"], row["f1-score"], row["support"]
            ])

    matrix = confusion_matrix(y_val, predictions, labels=labels)
    matrix_path = OUTPUT_DIR / "05_validation_confusion_matrix.csv"
    with matrix_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["actual \\ predicted"] + class_names)
        for class_name, row in zip(class_names, matrix):
            writer.writerow([class_name] + row.tolist())

    fig, ax = plt.subplots(figsize=(14, 12))
    ConfusionMatrixDisplay(
        confusion_matrix=matrix, display_labels=class_names
    ).plot(ax=ax, cmap="Blues", xticks_rotation=90,
           include_values=False, colorbar=True)
    ax.set_title("MiniLM embeddings + logistic regression: validation confusion matrix")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_validation_confusion_matrix.png", dpi=170)
    plt.close(fig)


def train_evaluate_save(train_embeddings, val_embeddings, y_train, y_val,
                        class_names, *, max_seq_length=256,
                        encoding_seconds=0.0, embeddings_from_cache=False):
    """Classifier-only training; supports independent synthetic-data tests."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    classifier = LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=1500, random_state=RANDOM_STATE
    )
    start = perf_counter()
    classifier.fit(train_embeddings, y_train)
    training_seconds = perf_counter() - start

    start = perf_counter()
    predictions = classifier.predict(val_embeddings)
    prediction_seconds = perf_counter() - start
    class_ids = list(range(len(class_names)))

    metrics = {
        "model": "Frozen MiniLM sentence embeddings + LogisticRegression(C=1.0)",
        "encoder_name": ENCODER_NAME,
        "dataset": "20 Newsgroups — remove headers, footers, quotes",
        "random_seed": RANDOM_STATE,
        "embedding_dimension": int(train_embeddings.shape[1]),
        "embedding_l2_normalized": True,
        "input_max_wordpiece_tokens": int(max_seq_length),
        "document_handling": "model truncates long documents to first tokens",
        "encoder_fine_tuned": False,
        "classifier_C": 1.0,
        "training_documents": len(y_train),
        "validation_documents": len(y_val),
        "classes": len(class_names),
        **validation_metrics(y_val, predictions, class_ids),
        "training_macro_f1": float(f1_score(
            y_train, classifier.predict(train_embeddings),
            labels=class_ids, average="macro", zero_division=0)),
        "embedding_seconds": round(encoding_seconds, 3),
        "embeddings_from_cache": bool(embeddings_from_cache),
        "classifier_training_seconds": round(training_seconds, 3),
        "classifier_validation_prediction_seconds": round(prediction_seconds, 3),
    }

    metrics_path = OUTPUT_DIR / "05_validation_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    save_detailed_evaluation(y_val, predictions, class_names)

    model_path = MODEL_DIR / "05_embedding_logreg.joblib"
    joblib.dump({
        "classifier": classifier,
        "class_names": class_names,
        "encoder_name": ENCODER_NAME,
        "max_seq_length": max_seq_length,
        "normalized_embeddings": True,
        "note": "Encoder weights are not included: reload the named SentenceTransformer.",
    }, model_path)

    print("\nVALIDATION RESULTS (official test partition was NOT used)")
    for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"):
        print(f"  {key:20s}: {metrics[key]:.4f}")
    print(f"  embedding dimension : {metrics['embedding_dimension']}")
    print(f"  embedding time      : {encoding_seconds:.1f}s (0 when reused from cache)")
    print(f"  classifier fit time : {training_seconds:.2f}s")

    previous_path = OUTPUT_DIR / "03_validation_metrics.json"
    if previous_path.exists():
        with previous_path.open(encoding="utf-8") as f:
            svm_result = json.load(f)
        print("\nREFERENCE: Installment 3 baseline (same validation partition):")
        print(f"  SVM macro-F1        : {svm_result['macro_f1']:.4f}")
        print(f"  Embedding macro-F1  : {metrics['macro_f1']:.4f}")
        print("  Note: This is not the final tuned model comparison.")

    print("\nSaved files:")
    for path in sorted(OUTPUT_DIR.glob("05_*")):
        print(" -", path.relative_to(ROOT))
    print(" -", model_path.relative_to(ROOT))
    print(" -", CACHE_PATH.relative_to(ROOT))
    print("\nNext: optimize the embedding approach with validation-only experiments.")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--force-reencode", action="store_true",
                        help="Recompute embeddings even when a valid cache exists")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")

    X_train, X_val, y_train, y_val, class_names = load_training_validation()
    print(f"Training documents: {len(X_train):,}; validation: {len(X_val):,}")
    print("The official test partition is never loaded by this script.")

    E_train, E_val, max_seq_length, elapsed, cached = encode_or_load(
        X_train, X_val, batch_size=args.batch_size,
        force=args.force_reencode,
    )
    train_evaluate_save(
        E_train, E_val, y_train, y_val, class_names,
        max_seq_length=max_seq_length,
        encoding_seconds=elapsed,
        embeddings_from_cache=cached,
    )


if __name__ == "__main__":
    main()

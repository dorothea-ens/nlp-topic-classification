"""Installment 10: genre transfer by retraining the SAME classifiers per corpus.

This is not zero-shot domain transfer: AG News and arXiv have different label
spaces from 20 Newsgroups, so each genre needs new supervised training.

Usage:
  python -m pip install datasets scikit-learn numpy matplotlib sentence-transformers
  python 10_cross_genre_generalization.py --self-test  # no internet
  python 10_cross_genre_generalization.py --quick      # exploratory only
  python 10_cross_genre_generalization.py              # both corpora, SVM + MiniLM
  python 10_cross_genre_generalization.py --include-transformer # costly; GPU recommended

Downloads: fancyzhx/ag_news and Tiansve/arxiv-cs-subfield (Hugging Face).
Default AG News is capped at 20,000 TRAIN documents for feasible resources;
validation/test are separate. arXiv uses its published train/validation/test.
Preset hyperparameters avoid tuning against the test set. 'Full' refers to
complete held-out test partitions, NOT all 120,000 AG News training examples.
Outputs reside in outputs/10_genre_generalization/{full,quick}/.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent
SEED = 42
AG_DATASET = "fancyzhx/ag_news"
ARXIV_DATASET = "Tiansve/arxiv-cs-subfield"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TRANSFORMER_MODEL = "distilbert/distilbert-base-uncased"
AG_LABELS = ["World", "Sports", "Business", "Sci/Tech"]


@dataclass
class GenreData:
    name: str
    source: str
    x_train: list[str]
    y_train: np.ndarray
    x_val: list[str]
    y_val: np.ndarray
    x_test: list[str]
    y_test: np.ndarray
    labels: list[str]
    test_is_sampled: bool = False


def balanced_indices(y, cap, seed=SEED):
    """Stratified downsample with deterministic random seed."""
    arr = np.asarray(y)
    if cap <= 0:
        raise ValueError("cap must be positive")
    if cap >= len(arr):
        return np.arange(len(arr))
    if cap < 2 * len(np.unique(arr)):
        raise ValueError("Requested subset cannot preserve each class")
    indices, _ = train_test_split(
        np.arange(len(arr)), train_size=cap, stratify=arr,
        random_state=seed,
    )
    return np.sort(indices)


def subset(xs, ys, cap, seed=SEED):
    ix = balanced_indices(ys, cap, seed)
    return [xs[i] for i in ix], np.asarray(ys, dtype=int)[ix]


def filter_empty(xs, ys):
    rows = [(str(x).strip(), y) for x, y in zip(xs, ys) if str(x).strip()]
    if not rows:
        raise ValueError("Dataset split has no nonempty texts")
    x, y = zip(*rows)
    return list(x), list(y)


def load_corpus(name, quick=False):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install first: python -m pip install datasets") from exc
    if name == "ag_news":
        ds = load_dataset(AG_DATASET)
        # Training/validation originate from published training partition ONLY.
        xs, ys = filter_empty(ds["train"]["text"], ds["train"]["label"])
        xtr, xva, ytr, yva = train_test_split(
            xs, ys, test_size=0.2, stratify=ys, random_state=SEED
        )
        xte, yte = filter_empty(ds["test"]["text"], ds["test"]["label"])
        # Capping the training subset is explicitly recorded in the metadata.
        xtr, ytr = subset(xtr, ytr, 1500 if quick else 20000)
        xva, yva = subset(xva, yva, 400 if quick else 4000)
        if quick:
            xte, yte = subset(xte, yte, 400)
        return GenreData(name, AG_DATASET, xtr, ytr, xva, yva,
                         xte, np.asarray(yte, dtype=int), AG_LABELS, quick)

    if name == "arxiv":
        ds = load_dataset(ARXIV_DATASET)
        required = {"train", "validation", "test"}
        if not required.issubset(set(ds.keys())):
            raise ValueError(f"Required arXiv splits absent: {required - set(ds.keys())}")
        names = sorted({str(z) for split in required for z in ds[split]["label"]})
        name_to_id = {c: i for i, c in enumerate(names)}
        def get(split):
            texts, string_labels = filter_empty(ds[split]["text"], ds[split]["label"])
            return texts, np.asarray([name_to_id[z] for z in string_labels], dtype=int)
        xtr, ytr = get("train")
        xva, yva = get("validation")
        xte, yte = get("test")
        if quick:
            xtr, ytr = subset(xtr, ytr, 600)
            xva, yva = subset(xva, yva, 180)
            xte, yte = subset(xte, yte, 180)
        return GenreData(name, ARXIV_DATASET, xtr, ytr, xva, yva,
                         xte, yte, names, quick)
    raise ValueError(f"Unknown dataset {name!r}")


def audit_splits(data):
    """Detect exact-text duplicates across splits, and report without peeking at labels."""
    sets = [set(hashlib.sha256(s.strip().lower().encode()).hexdigest() for s in xs)
            for xs in (data.x_train, data.x_val, data.x_test)]
    return {
        "train_val_shared_texts": len(sets[0] & sets[1]),
        "train_test_shared_texts": len(sets[0] & sets[2]),
        "val_test_shared_texts": len(sets[1] & sets[2]),
    }


def metrics(ys, pred, n_classes):
    labels = list(range(n_classes))
    return {
        "accuracy": float(accuracy_score(ys, pred)),
        "macro_f1": float(f1_score(ys, pred, average="macro", labels=labels,
                                  zero_division=0)),
        "weighted_f1": float(f1_score(ys, pred, average="weighted", labels=labels,
                                     zero_division=0)),
    }


def run_svm(data):
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2),
            sublinear_tf=True, max_features=100000,
        )),
        ("svm", LinearSVC(C=1.0, random_state=SEED)),
    ])
    start = perf_counter()
    model.fit(data.x_train, data.y_train)
    training_time = perf_counter() - start
    start = perf_counter()
    val_pred = np.asarray(model.predict(data.x_val), dtype=int)
    val_time = perf_counter() - start
    start = perf_counter()
    test_pred = np.asarray(model.predict(data.x_test), dtype=int)
    test_time = perf_counter() - start
    return val_pred, test_pred, training_time, val_time, test_time


def run_minilm(data, batch_size):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install: python -m pip install sentence-transformers") from exc
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    start = perf_counter()
    x_train = encoder.encode(data.x_train, batch_size=batch_size,
                             normalize_embeddings=True, convert_to_numpy=True,
                             show_progress_bar=True)
    classifier = LogisticRegression(C=1.0, max_iter=1500, random_state=SEED)
    classifier.fit(x_train, data.y_train)
    training_time = perf_counter() - start
    start = perf_counter()
    xv = encoder.encode(data.x_val, batch_size=batch_size,
                        normalize_embeddings=True, convert_to_numpy=True,
                        show_progress_bar=False)
    val_pred = classifier.predict(xv).astype(int)
    val_time = perf_counter() - start
    start = perf_counter()
    xt = encoder.encode(data.x_test, batch_size=batch_size,
                        normalize_embeddings=True, convert_to_numpy=True,
                        show_progress_bar=True)
    test_pred = classifier.predict(xt).astype(int)
    test_time = perf_counter() - start
    return val_pred, test_pred, training_time, val_time, test_time


def run_transformer(data, output_path, batch_size, epochs):
    try:
        import torch
        from datasets import Dataset
        from transformers import (
            AutoTokenizer, AutoModelForSequenceClassification, Trainer,
            TrainingArguments, DataCollatorWithPadding,
        )
    except ImportError as exc:
        raise RuntimeError("Install: python -m pip install torch transformers accelerate") from exc
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL)
    net = AutoModelForSequenceClassification.from_pretrained(
        TRANSFORMER_MODEL, num_labels=len(data.labels),
        id2label={i: label for i, label in enumerate(data.labels)},
        label2id={label: i for i, label in enumerate(data.labels)},
    )
    def tokenized(xs, ys):
        ds = Dataset.from_dict({"text": xs, "labels": [int(z) for z in ys]})
        def encode(batch):
            return tokenizer(batch["text"], truncation=True, max_length=256)
        return ds.map(encode, batched=True, remove_columns=["text"])
    train_ds = tokenized(data.x_train, data.y_train)
    val_ds = tokenized(data.x_val, data.y_val)
    test_ds = tokenized(data.x_test, data.y_test)
    args = TrainingArguments(
        output_dir=str(output_path / "transformer_checkpoints"),
        seed=SEED, data_seed=SEED,
        learning_rate=2e-5, num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
        weight_decay=0.01,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="macro_f1",
        greater_is_better=True, save_total_limit=1,
        report_to="none", logging_strategy="epoch",
    )
    def compute_metrics(p):
        logits, labels = p
        return {"macro_f1": float(f1_score(labels, logits.argmax(axis=-1),
                                            average="macro", zero_division=0))}
    trainer = Trainer(
        model=net, args=args, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics, processing_class=tokenizer,
    )
    start = perf_counter()
    trainer.train()
    training_time = perf_counter() - start
    start = perf_counter()
    val_pred = trainer.predict(val_ds).predictions.argmax(axis=-1).astype(int)
    val_time = perf_counter() - start
    start = perf_counter()
    test_pred = trainer.predict(test_ds).predictions.argmax(axis=-1).astype(int)
    test_time = perf_counter() - start
    model_dir = output_path / "trained_distilbert"
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    return val_pred, test_pred, training_time, val_time, test_time


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fd:
        writer = csv.DictWriter(fd, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_outputs(data, name, preds, numbers, out):
    yp_val, yp_test = preds
    training_time, val_time, test_time = numbers
    scores = metrics(data.y_test, yp_test, len(data.labels))
    val_scores = metrics(data.y_val, yp_val, len(data.labels))
    row = {
        "dataset": data.name, "model": name, "classes": len(data.labels),
        "train_size": len(data.x_train), "val_size": len(data.x_val),
        "test_size": len(data.x_test),
        "val_macro_f1": val_scores["macro_f1"],
        **scores,
        "training_seconds": round(training_time, 3),
        "validation_seconds": round(val_time, 3),
        "test_inference_seconds": round(test_time, 3),
        "test_documents_per_second": round(len(data.x_test) / max(test_time, 1e-8), 2),
    }
    test_report = classification_report(
        data.y_test, yp_test, labels=np.arange(len(data.labels)),
        target_names=data.labels, output_dict=True, zero_division=0,
    )
    (out / f"{data.name}_{name}_classification_report.json").write_text(
        json.dumps(test_report, indent=2), encoding="utf-8"
    )
    conf = confusion_matrix(data.y_test, yp_test, labels=np.arange(len(data.labels)))
    np.savetxt(out / f"{data.name}_{name}_confusion_matrix.csv",
               conf, delimiter=",", fmt="%d")
    predictions = [
        {"row_index": i, "actual_id": int(y), "predicted_id": int(p),
         "actual_category": data.labels[int(y)],
         "predicted_category": data.labels[int(p)],
         "correct": int(y == p),
         "text_excerpt": s[:300].replace("\n", " ")}
        for i, (s, y, p) in enumerate(zip(data.x_test, data.y_test, yp_test))
    ]
    write_csv(out / f"{data.name}_{name}_predictions.csv", predictions,
              list(predictions[0].keys()))
    return row


def save_chart(rows, out):
    if not rows:
        return
    datasets = sorted(set(r["dataset"] for r in rows))
    models = sorted(set(r["model"] for r in rows))
    xx = np.arange(len(datasets))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, model in enumerate(models):
        yy = [next((r["macro_f1"] for r in rows
                    if r["dataset"] == d and r["model"] == model), np.nan)
              for d in datasets]
        ax.bar(xx + i * width, yy, width=width, label=model)
    ax.set_xticks(xx + (len(models) - 1) * width / 2, datasets)
    ax.set_ylabel("Test macro-F1")
    ax.set_ylim(0, 1)
    ax.set_title("Genre-specific classifiers: held-out test performance")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "10_genre_comparison.png", dpi=180)
    plt.close(fig)


def self_test():
    assert list(balanced_indices([0, 0, 0, 1, 1, 1], 4)) == list(
        balanced_indices([0, 0, 0, 1, 1, 1], 4)
    )
    toy = GenreData(
        name="toy", source="synthetic", x_train=[
            "space rocket shuttle launch", "astronaut spacecraft mission",
            "soccer goal match teams", "football sports teams win",
            "space telescope astronomy", "soccer game sports match",
        ], y_train=np.array([0, 0, 1, 1, 0, 1]),
        x_val=["space mission launch", "soccer sports match"],
        y_val=np.array([0, 1]),
        x_test=["rocket spacecraft", "football game"],
        y_test=np.array([0, 1]), labels=["space", "sports"],
    )
    val, test, tr, va, te = run_svm(toy)
    assert len(val) == len(toy.y_val) and len(test) == len(toy.y_test)
    assert np.isfinite(metrics(toy.y_test, test, 2)["macro_f1"])
    assert audit_splits(toy)["train_test_shared_texts"] == 0
    import tempfile
    with tempfile.TemporaryDirectory() as folder:
        o = Path(folder)
        row = save_outputs(toy, "svm", (val, test), (tr, va, te), o)
        save_chart([row], o)
        assert (o / "toy_svm_predictions.csv").is_file()
        assert (o / "10_genre_comparison.png").is_file()
    print("SELF-TEST PASSED: split, baseline, scores, report, CSV, chart")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["both", "ag_news", "arxiv"], default="both")
    parser.add_argument("--quick", action="store_true", help="Partial data; not final research results")
    parser.add_argument("--include-transformer", action="store_true", help="Fine-tune new DistilBERT per genre")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite saved results intentionally")
    config = parser.parse_args()
    if config.self_test:
        self_test()
        return
    if config.batch_size < 1 or config.epochs < 1:
        parser.error("--batch-size and --epochs must be positive")
    np.random.seed(SEED)
    random.seed(SEED)
    mode = "quick" if config.quick else "full"
    output = ROOT / "outputs" / "10_genre_generalization" / mode
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "10_metrics.csv"
    if marker.exists() and not config.overwrite:
        raise FileExistsError(f"Existing completed comparison: {marker}. Use --overwrite only deliberately.")
    kinds = ["ag_news", "arxiv"] if config.dataset == "both" else [config.dataset]
    models = ["svm", "minilm"] + (["distilbert"] if config.include_transformer else [])
    rows = []
    audits = {}
    for ds_id in kinds:
        data = load_corpus(ds_id, quick=config.quick)
        audit = audit_splits(data)
        audits[ds_id] = audit
        print(f"Dataset {data.name}: train={len(data.x_train)}, val={len(data.x_val)}, "
              f"test={len(data.x_test)}, labels={len(data.labels)}")
        print("Exact-text overlap audit:", audit)
        if audit["train_test_shared_texts"]:
            print("WARNING: exact duplicate text occurs across training/test. "
                  "Interpret results with caution; counts included in metadata.")
        for model in models:
            print(f"Training {model} on {ds_id} ...")
            if model == "svm":
                result = run_svm(data)
            elif model == "minilm":
                result = run_minilm(data, batch_size=config.batch_size)
            else:
                result = run_transformer(
                    data, output / f"{ds_id}_distilbert", batch_size=config.batch_size,
                    epochs=config.epochs
                )
            row = save_outputs(data, model, result[:2], result[2:], output)
            rows.append(row)
            write_csv(marker, rows, list(rows[0].keys()))
            print(f"  val macro-F1={row['val_macro_f1']:.4f}; "
                  f"test macro-F1={row['macro_f1']:.4f}")
    save_chart(rows, output)
    metadata = {
        "mode": mode,
        "definition": "Supervised genre-specific retraining, not zero-shot transfer",
        "methods": models,
        "seed": SEED,
        "datasets": {ds: AG_DATASET if ds == "ag_news" else ARXIV_DATASET
                     for ds in kinds},
        "dataset_sizes": {r["dataset"]: {k: r[k] for k in
                           ("train_size", "val_size", "test_size", "classes")}
                          for r in rows},
        "quick_test_is_sampled": config.quick,
        "all_test_labels_not_used_for_selection": True,
        "ag_news_training_cap": 1500 if config.quick else 20000,
        "arxiv_uses_published_splits": True,
        "duplicate_audit": audits,
        "fixed_hyperparameters": True,
        "comparisons_across_datasets_caution":
            "Different classes and difficulty: compare approaches within each genre, "
            "not absolute scores as direct measure of domain transfer.",
    }
    (output / "10_experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Finished: {output}")


if __name__ == "__main__":
    main()

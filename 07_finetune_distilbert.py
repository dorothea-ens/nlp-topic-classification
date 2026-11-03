"""Installment 7 — Fine-tune DistilBERT for 20-class document classification.

Same cleaned training/validation partitions as Installments 2–6.
The official 20 Newsgroups TEST partition is NEVER loaded or used here.

Install:
    python -m pip install torch transformers accelerate scikit-learn numpy matplotlib

Run:
    python 07_finetune_distilbert.py
    python 07_finetune_distilbert.py --smoke-test  # very small end-to-end run
    python 07_finetune_distilbert.py --self-test   # offline test, no model download

The smoke-test uses deliberately tiny datasets and is NOT valid research evidence.
Training on CPU is possible but substantially more resource-intensive than on a GPU.
Outputs appear in models/, checkpoints/, and outputs/.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import tempfile
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_20newsgroups
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

ROOT = Path(__file__).resolve().parent
SEED = 42
MODEL_ID = "distilbert/distilbert-base-uncased"
REMOVED_PARTS = ("headers", "footers", "quotes")


def load_training_validation():
    """Reproduce the prior split exactly; never access subset='test'."""
    raw = fetch_20newsgroups(
        subset="train", remove=REMOVED_PARTS, random_state=SEED
    )
    pairs = [
        (text, int(label))
        for text, label in zip(raw.data, raw.target)
        if text.strip()
    ]
    if not pairs:
        raise ValueError("No nonempty training texts are available.")
    texts, labels = zip(*pairs)
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.20, random_state=SEED,
        stratify=labels,
    )
    return (
        list(X_train), list(X_val),
        np.asarray(y_train, dtype=np.int64),
        np.asarray(y_val, dtype=np.int64),
        list(raw.target_names),
    )


class EncodedTextDataset:
    """PyTorch Dataset protocol; already tokenized, padded dynamically by Trainer."""

    def __init__(self, texts, labels, tokenizer, max_length):
        if len(texts) != len(labels):
            raise ValueError("Texts and labels must have the same length.")
        if not 1 <= max_length <= 512:
            raise ValueError("max_length must be between 1 and 512.")
        self.encodings = tokenizer(
            list(texts), truncation=True, max_length=max_length,
            padding=False,
        )
        self.labels = [int(v) for v in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        row = {k: v[index] for k, v in self.encodings.items()}
        row["labels"] = self.labels[index]
        return row


def metrics_from_predictions(y_true, y_pred, class_ids):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(
            y_true, y_pred, labels=class_ids, average="macro", zero_division=0
        )),
        "macro_recall": float(recall_score(
            y_true, y_pred, labels=class_ids, average="macro", zero_division=0
        )),
        "macro_f1": float(f1_score(
            y_true, y_pred, labels=class_ids, average="macro", zero_division=0
        )),
        "weighted_f1": float(f1_score(
            y_true, y_pred, labels=class_ids, average="weighted", zero_division=0
        )),
    }


def build_compute_metrics(class_ids):
    def compute_metrics(prediction_output):
        logits = prediction_output.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        y_pred = np.argmax(logits, axis=-1)
        return metrics_from_predictions(
            prediction_output.label_ids, y_pred, class_ids
        )
    return compute_metrics


def per_class_subset(texts, labels, per_class, *, seed=SEED):
    """Tiny class-balanced sample for the smoke-test (never used in real run)."""
    rng = np.random.default_rng(seed)
    keep = []
    for cls in sorted(np.unique(labels)):
        ids = np.flatnonzero(labels == cls)
        if len(ids) < per_class:
            raise ValueError(f"Too few records in class {cls} for smoke-test.")
        keep.extend(rng.choice(ids, size=per_class, replace=False).tolist())
    keep = sorted(keep)
    return [texts[i] for i in keep], np.asarray(labels)[keep]


def write_evaluation(y_true, y_pred, class_names, out_dir, prefix):
    """Write actual measured scores, per-class report, and confusion matrix."""
    out_dir.mkdir(parents=True, exist_ok=True)
    label_ids = list(range(len(class_names)))
    scores = metrics_from_predictions(y_true, y_pred, label_ids)
    report = classification_report(
        y_true, y_pred, labels=label_ids, target_names=class_names,
        output_dict=True, zero_division=0,
    )
    report_path = out_dir / f"{prefix}validation_classification_report.csv"
    with report_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["class_or_summary", "precision", "recall", "f1_score", "support"])
        for name in class_names + ["macro avg", "weighted avg"]:
            row = report[name]
            writer.writerow([name, row["precision"], row["recall"],
                             row["f1-score"], row["support"]])

    matrix = confusion_matrix(y_true, y_pred, labels=label_ids)
    with (out_dir / f"{prefix}validation_confusion_matrix.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["actual \\ predicted"] + class_names)
        for name, row in zip(class_names, matrix):
            writer.writerow([name] + row.tolist())

    fig, ax = plt.subplots(figsize=(14, 12))
    ConfusionMatrixDisplay(
        confusion_matrix=matrix, display_labels=class_names
    ).plot(ax=ax, include_values=False, cmap="Blues", xticks_rotation=90)
    ax.set_title("DistilBERT: validation confusion matrix")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}validation_confusion_matrix.png", dpi=170)
    plt.close(fig)
    return scores


def save_learning_plots(history, out_dir, prefix):
    """Plots logged loss and validation macro-F1 without inventing results."""
    losses = [(r["step"], r["loss"]) for r in history
              if "loss" in r and "step" in r]
    f1_scores = [(r.get("epoch", i + 1), r["eval_macro_f1"])
                 for i, r in enumerate(history) if "eval_macro_f1" in r]
    if losses:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot([x for x, _ in losses], [y for _, y in losses], marker=".")
        ax.set(xlabel="Optimizer step", ylabel="Training loss",
               title="DistilBERT training loss")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}training_loss.png", dpi=170)
        plt.close(fig)
    if f1_scores:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot([x for x, _ in f1_scores], [y for _, y in f1_scores], marker="o")
        ax.set(xlabel="Epoch", ylabel="Validation macro-F1",
               title="DistilBERT validation performance", ylim=(0, 1.02))
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}validation_f1_by_epoch.png", dpi=170)
        plt.close(fig)


def run_training(args):
    # Keep heavy ML framework imports inside training: --self-test works offline.
    try:
        import torch
        import transformers
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise SystemExit(
            "Install dependencies first: "
            "python -m pip install torch transformers accelerate "
            "scikit-learn numpy matplotlib"
        ) from exc

    set_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    X_train, X_val, y_train, y_val, class_names = load_training_validation()
    if args.smoke_test:
        X_train, y_train = per_class_subset(X_train, y_train, 3)
        X_val, y_val = per_class_subset(X_val, y_val, 2)
        args.epochs, args.max_length = 1, 64
        print("SMOKE TEST: intentionally tiny data; not valid performance evidence.")

    tag = "07_smoke_" if args.smoke_test else "07_"
    out_dir = ROOT / "outputs"
    model_dir = ROOT / "models" / f"{tag}distilbert"
    checkpoint_dir = ROOT / "checkpoints" / f"{tag}distilbert"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_ID}; internet access required on first run.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    train_ds = EncodedTextDataset(X_train, y_train, tokenizer, args.max_length)
    val_ds = EncodedTextDataset(X_val, y_val, tokenizer, args.max_length)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=len(class_names),
        id2label={i: name for i, name in enumerate(class_names)},
        label2id={name: i for i, name in enumerate(class_names)},
    )
    # The classification head starts randomly initialized: it is trained here.
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    cuda = bool(torch.cuda.is_available())
    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accumulation,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_strategy="steps",
        logging_steps=10 if args.smoke_test else 25,
        report_to="none",
        push_to_hub=False,
        fp16=cuda,
        seed=SEED,
        data_seed=SEED,
        optim="adamw_torch",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=build_compute_metrics(list(range(len(class_names)))),
    )

    print(f"Training on {'NVIDIA CUDA GPU' if cuda else 'available accelerator / CPU'}")
    print(f"Documents: {len(train_ds)} train, {len(val_ds)} validation")
    train_start = perf_counter()
    trainer.train()
    train_seconds = perf_counter() - train_start

    # Trainer now holds the best validation-macro-F1 checkpoint's weights.
    predict_start = perf_counter()
    prediction = trainer.predict(val_ds)
    prediction_seconds = perf_counter() - predict_start
    logits = prediction.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    predicted_labels = np.argmax(logits, axis=-1)
    scores = write_evaluation(
        y_val, predicted_labels, class_names, out_dir, tag
    )
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    history = trainer.state.log_history
    (out_dir / f"{tag}training_log.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    save_learning_plots(history, out_dir, tag)
    result = {
        "model": MODEL_ID,
        "classification_architecture": "fine-tuned DistilBERT with 20-class classification head",
        "dataset": "20 Newsgroups, headers/footers/quotes removed",
        "random_seed": SEED,
        "training_documents": len(X_train),
        "validation_documents": len(X_val),
        "classes": len(class_names),
        "max_length_tokens": args.max_length,
        "epochs": args.epochs,
        "train_batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.grad_accumulation,
        "learning_rate": args.learning_rate,
        "weight_decay": 0.01,
        "smoke_test_only": args.smoke_test,
        "official_test_set_used": False,
        "framework_versions": {
            "torch": torch.__version__, "transformers": transformers.__version__
        },
        "gpu_cuda": cuda,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_validation_metric": trainer.state.best_metric,
        "train_seconds": round(train_seconds, 2),
        "validation_prediction_seconds": round(prediction_seconds, 2),
        **scores,
    }
    (out_dir / f"{tag}validation_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nValidation metrics:")
    for name, value in scores.items():
        print(f"  {name:20}: {value:.4f}")
    print("Saved model:", model_dir)
    print("Saved evaluation files:", out_dir)
    print("Official test data were NOT loaded.")
    return result


def offline_self_test():
    """Test data contract, predictions, and export without downloading models."""
    class FakeTokenizer:
        def __call__(self, texts, truncation, max_length, padding):
            assert truncation and not padding
            encoded = [list(range(min(max(1, len(t.split())), max_length)))
                       for t in texts]
            return {"input_ids": encoded, "attention_mask":
                    [[1] * len(row) for row in encoded]}

    items = ["a b", "c", "d e f", "g h i", "j k", "l m"]
    true = np.array([0, 0, 1, 1, 2, 2])
    ds = EncodedTextDataset(items, true, FakeTokenizer(), 64)
    assert len(ds) == 6 and ds[2]["labels"] == 1
    assert len(ds[2]["input_ids"]) == 3
    sample, labels = per_class_subset(items, true, 1)
    assert len(sample) == len(labels) == 3 and set(labels) == {0, 1, 2}

    pred = np.array([0, 0, 1, 2, 2, 2])
    expected = metrics_from_predictions(true, pred, [0, 1, 2])
    assert expected["accuracy"] == 5 / 6
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory)
        observed = write_evaluation(true, pred, ["A", "B", "C"], out, "07_")
        assert observed == expected
        assert (out / "07_validation_classification_report.csv").exists()
        assert (out / "07_validation_confusion_matrix.png").exists()
        save_learning_plots(
            [{"step": 1, "loss": 0.9},
             {"step": 2, "epoch": 1, "eval_macro_f1": 0.6}],
            out, "07_",
        )
        assert (out / "07_training_loss.png").exists()
        assert (out / "07_validation_f1_by_epoch.png").exists()
    print("Offline self-test PASSED: dataset, stratified subset, metrics, CSV and plots.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--grad-accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Train with 3 documents/class and validate on 2/class.")
    parser.add_argument("--self-test", action="store_true",
                        help="Offline checks; does not import transformers or fetch data.")
    args = parser.parse_args()
    if args.self_test:
        offline_self_test()
        return
    if (args.epochs <= 0 or args.batch_size < 1 or args.eval_batch_size < 1
            or args.grad_accumulation < 1 or args.learning_rate <= 0
            or not 2 <= args.max_length <= 512):
        parser.error("Check: epochs, batches, accumulation and LR must be > 0; "
                     "max-length must be 2..512")
    run_training(args)


if __name__ == "__main__":
    main()

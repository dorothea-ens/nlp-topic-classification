"""Installment 8 — reproducible DistilBERT optimization, validation-only.

Compatible with 07_finetune_distilbert.py; put both files in the same folder.
Official 20 Newsgroups TEST examples are never loaded.

Full run (GPU recommended):
    python 08_optimize_distilbert.py
Cheaper exploratory run, separate outputs so it cannot be confused with full:
    python 08_optimize_distilbert.py --quick
End-to-end tiny model-download test (NOT research evidence):
    python 08_optimize_distilbert.py --smoke-test
Offline testing, no network or transformer dependency:
    python 08_optimize_distilbert.py --self-test

Full experiments:
 A: On 25% stratified training data, screen learning rates {1e-5,2e-5,5e-5}
    and maximum input lengths {128,256}; select by validation macro-F1.
 B: With the selected hyperparameters, train on nested stratified training
    fractions {10%,25%,50%,100%}, keeping the SAME full validation set.
    Reuses matching 25% experiment. Training macro-F1 and runtime measured.

IMPORTANT: hyperparameters screened at 25% need not be optimal for 100%.
The validation set is reused for selection/diagnostics; official TEST is held
out for Installment 9's final comparison. Each run is a fresh fine-tuning.
Never interpret smoke/quick results as the full research study.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import random
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT / "07_finetune_distilbert.py"
SEED = 42
MODEL_ID = "distilbert/distilbert-base-uncased"
FULL_LRS = (1e-5, 2e-5, 5e-5)
FULL_LENGTHS = (128, 256)
FULL_FRACTIONS = (0.10, 0.25, 0.50, 1.0)
SCREEN_FRACTION = 0.25


def previous_installment():
    if not PREVIOUS.exists():
        raise FileNotFoundError(f"Put Installment 7 beside this file: {PREVIOUS}")
    spec = importlib.util.spec_from_file_location("nlp_installment_07", PREVIOUS)
    if spec is None or spec.loader is None:
        raise ImportError("Could not load Installment 7.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_nested_indices(labels, fractions, seed=SEED):
    """Nested, nearly stratified samples; no class is absent from a subset."""
    labels = np.asarray(labels, dtype=np.int64)
    if len(labels) == 0 or labels.ndim != 1:
        raise ValueError("Non-empty one-dimensional labels required.")
    rng = np.random.default_rng(seed)
    per_class = {
        int(c): rng.permutation(np.flatnonzero(labels == c))
        for c in np.unique(labels)
    }
    output = {}
    for frac in sorted(set(fractions)):
        if not 0 < frac <= 1:
            raise ValueError("Each training fraction must be in (0,1].")
        indices = []
        for shuffled in per_class.values():
            count = max(1, round(len(shuffled) * frac))
            indices.extend(shuffled[:count].tolist())
        output[frac] = np.array(sorted(indices), dtype=np.int64)
    for i, smaller in enumerate(list(output.values())[:-1]):
        larger = list(output.values())[i + 1]
        if not set(smaller).issubset(set(larger)):
            raise AssertionError("Nested sampling invariant failed.")
    return output


def fingerprint_data(x_train, y_train, x_val, y_val):
    h = hashlib.sha256()
    for split, xs, ys in (("train", x_train, y_train),
                          ("validation", x_val, y_val)):
        h.update(split.encode("ascii"))
        for t, label in zip(xs, ys):
            b = t.encode("utf-8")
            h.update(len(b).to_bytes(8, "little"))
            h.update(b)
            h.update(int(label).to_bytes(4, "little", signed=True))
    return h.hexdigest()


@dataclass(frozen=True)
class Trial:
    learning_rate: float
    max_length: int
    fraction: float

    @property
    def tag(self):
        return (f"p{round(100 * self.fraction):03d}_"
                f"len{self.max_length}_lr{self.learning_rate:.0e}")


def select_best(rows):
    if not rows:
        raise ValueError("No screening results.")
    if any(not math.isfinite(float(r["validation_macro_f1"])) for r in rows):
        raise ValueError("Non-finite validation score.")
    # Deterministic tie-breaking: prefer fewer tokens, smaller LR.
    return max(rows, key=lambda r: (
        float(r["validation_macro_f1"]), -int(r["max_length"]),
        -float(r["learning_rate"])
    ))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_screening(rows, file):
    fig, ax = plt.subplots(figsize=(9, 5))
    lrs = sorted({float(r["learning_rate"]) for r in rows})
    for length in sorted({int(r["max_length"]) for r in rows}):
        matching = {float(r["learning_rate"]): float(r["validation_macro_f1"])
                    for r in rows if int(r["max_length"]) == length}
        ax.plot([f"{lr:.0e}" for lr in lrs],
                [matching.get(lr, np.nan) for lr in lrs],
                marker="o", label=f"Maximum {length} tokens")
    ax.set(xlabel="Learning rate", ylabel="Validation macro-F1",
           title="DistilBERT: hyperparameter screening")
    ax.set_ylim(0, 1.02)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(file, dpi=170)
    plt.close(fig)


def plot_learning_curve(rows, file):
    rows = sorted(rows, key=lambda r: r["fraction"])
    frac = [100 * r["fraction"] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(frac, [r["train_macro_f1"] for r in rows], marker="o",
            label="Training macro-F1")
    ax.plot(frac, [r["validation_macro_f1"] for r in rows], marker="o",
            label="Validation macro-F1")
    ax.set(xlabel="Training subset (%)", ylabel="Macro-F1",
           title="DistilBERT: effect of training size", ylim=(0, 1.02),
           xticks=frac)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(file, dpi=170)
    plt.close(fig)


def plot_training_cost(rows, file):
    rows = sorted(rows, key=lambda r: r["fraction"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([100 * r["fraction"] for r in rows],
            [r["train_seconds"] / 60 for r in rows], marker="o")
    ax.set(xlabel="Training subset (%)", ylabel="Training time (minutes)",
           title="DistilBERT: observed training cost")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(file, dpi=170)
    plt.close(fig)


def run_trial(trial, *, x_train, y_train, x_val, y_val, names,
              sampled_indices, dataset_hash, args, prior, folder):
    """Fresh transformer initialization for every trial, with optional resume-by-result."""
    run_dir = folder / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = run_dir / f"{trial.tag}.json"
    model_directory = folder / "models" / trial.tag
    # Reject stale cache when settings or source data are different.
    signature = {
        "model": MODEL_ID, "split_sha256": dataset_hash,
        "config": asdict(trial), "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation": args.grad_accumulation,
        "weight_decay": args.weight_decay,
        "seed": SEED,
        "docs": int(len(sampled_indices)),
        "experiment_mode": folder.name,
    }
    if metrics_file.exists() and not args.overwrite:
        cached = json.loads(metrics_file.read_text(encoding="utf-8"))
        if (cached.get("signature") == signature and
            (trial.fraction < 1.0 or (model_directory / "config.json").exists())):
            print(f"Using completed experiment: {trial.tag}")
            return cached

    # Heavy imports only when training, not for the offline self-test.
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
    set_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    trial_x = [x_train[int(i)] for i in sampled_indices]
    trial_y = np.asarray(y_train)[sampled_indices]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    train_ds = prior.EncodedTextDataset(
        trial_x, trial_y, tokenizer, trial.max_length)
    validation_ds = prior.EncodedTextDataset(
        x_val, y_val, tokenizer, trial.max_length)
    # Every run initializes the classification head and encoder from the
    # exact same pretrained base, never from an earlier fine-tuned checkpoint.
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=len(names),
        id2label={i: name for i, name in enumerate(names)},
        label2id={name: i for i, name in enumerate(names)},
    )
    checkpoint_dir = folder / "checkpoints" / trial.tag
    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        learning_rate=trial.learning_rate,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accumulation,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1", greater_is_better=True,
        save_total_limit=1,
        logging_steps=25, logging_strategy="steps",
        report_to="none", push_to_hub=False,
        fp16=torch.cuda.is_available(),
        seed=SEED, data_seed=SEED, optim="adamw_torch",
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_ds,
        eval_dataset=validation_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        processing_class=tokenizer,
        compute_metrics=prior.build_compute_metrics(list(range(len(names)))),
    )
    print(f"Training {trial.tag}: {len(train_ds)} train / {len(validation_ds)} validation")
    start = perf_counter()
    trainer.train()
    training_seconds = perf_counter() - start

    validation_pred = trainer.predict(validation_ds)
    training_pred = trainer.predict(train_ds)
    val_logits = validation_pred.predictions
    train_logits = training_pred.predictions
    if isinstance(val_logits, tuple):
        val_logits = val_logits[0]
    if isinstance(train_logits, tuple):
        train_logits = train_logits[0]
    yhat_val = np.argmax(val_logits, axis=-1)
    yhat_train = np.argmax(train_logits, axis=-1)
    val_scores = prior.metrics_from_predictions(y_val, yhat_val, list(range(len(names))))
    train_macro_f1 = float(f1_score(
        trial_y, yhat_train, labels=list(range(len(names))),
        average="macro", zero_division=0
    ))
    best_epoch = None
    for entry in trainer.state.log_history:
        if (entry.get("eval_macro_f1") is not None and
            np.isclose(float(entry["eval_macro_f1"]),
                       float(trainer.state.best_metric), rtol=0, atol=1e-8)):
            best_epoch = entry.get("epoch")
            break
    record = {
        "signature": signature,
        "tag": trial.tag,
        "fraction": trial.fraction,
        "learning_rate": trial.learning_rate,
        "max_length": trial.max_length,
        "n_train": len(train_ds),
        "n_val": len(validation_ds),
        "train_macro_f1": train_macro_f1,
        "validation_macro_f1": val_scores["macro_f1"],
        "validation_accuracy": val_scores["accuracy"],
        "validation_macro_precision": val_scores["macro_precision"],
        "validation_macro_recall": val_scores["macro_recall"],
        "validation_weighted_f1": val_scores["weighted_f1"],
        "train_seconds": round(training_seconds, 2),
        "best_epoch": best_epoch,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "smoke_or_quick_not_final": args.smoke_test or args.quick,
        "official_test_used": False,
    }
    (run_dir / f"{trial.tag}_history.json").write_text(
        json.dumps(trainer.state.log_history, indent=2), encoding="utf-8"
    )
    if trial.fraction == 1.0:
        model_directory.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(model_directory))
        tokenizer.save_pretrained(str(model_directory))
        # Final full-size model gets detailed per-class diagnostics.
        prior.write_evaluation(
            y_val, yhat_val, names, folder, "08_final_"
        )
        prior.save_learning_plots(trainer.state.log_history, folder, "08_final_")
    # Write result only after all files and model export succeed.
    metrics_file.write_text(json.dumps(record, indent=2), encoding="utf-8")
    if not args.keep_checkpoints:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
    return record


def offline_self_test():
    """No dataset/model downloads; checks sampling, reporting and selection."""
    prior = previous_installment()
    labels = np.repeat(np.arange(4), 20)
    idx = build_nested_indices(labels, (0.1, 0.25, 0.5, 1.0))
    assert [len(idx[f]) for f in (0.1, 0.25, 0.5, 1.0)] == [8, 20, 40, 80]
    assert all(set(labels[idx[0.1]]) == set(labels[idx[f]]) for f in idx)
    assert set(idx[0.1]).issubset(set(idx[0.25]))
    assert set(idx[0.25]).issubset(set(idx[0.5]))
    assert set(idx[0.5]).issubset(set(idx[1.0]))
    assert all(np.array_equal(idx[f], build_nested_indices(labels, idx)[f]) for f in idx)
    assert Trial(2e-5, 256, 0.25).tag == "p025_len256_lr2e-05"
    rows = [
        {"max_length": 256, "learning_rate": 2e-5,
         "validation_macro_f1": 0.5},
        {"max_length": 128, "learning_rate": 1e-5,
         "validation_macro_f1": 0.7},
    ]
    assert select_best(rows)["max_length"] == 128
    assert prior.metrics_from_predictions([0, 1], [0, 1], [0, 1])["macro_f1"] == 1
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        write_csv(d / "test.csv", rows,
                  ["max_length", "learning_rate", "validation_macro_f1"])
        assert (d / "test.csv").exists()
        plot_screening(rows, d / "screen.png")
        curve = [
            {"fraction": f, "train_macro_f1": 0.85 + 0.01 * i,
             "validation_macro_f1": 0.55 + 0.05 * i,
             "train_seconds": 100 * (i + 1)}
            for i, f in enumerate((.10, .25, .5, 1.))
        ]
        plot_learning_curve(curve, d / "curve.png")
        plot_training_cost(curve, d / "runtime.png")
        assert all((d / name).exists()
                   for name in ("screen.png", "curve.png", "runtime.png"))
    print("OFFLINE SELF-TEST PASSED: sampling, metrics, trial IDs, selection, CSV, plots")


def run(args):
    prior = previous_installment()
    x_train, x_val, y_train, y_val, names = prior.load_training_validation()
    if args.smoke_test:
        x_train, y_train = prior.per_class_subset(x_train, y_train, 4)
        x_val, y_val = prior.per_class_subset(x_val, y_val, 2)
        args.epochs = 1
        screen_fraction = 0.5
        lrs = (2e-5,)
        lengths = (64,)
        fractions = (0.5, 1.0)
        mode = "08_smoke_DO_NOT_REPORT"
    elif args.quick:
        screen_fraction = SCREEN_FRACTION
        lrs = (2e-5, 5e-5)
        lengths = FULL_LENGTHS
        fractions = (0.25, 0.5, 1.0)
        mode = "08_quick_EXPLORATORY"
    else:
        screen_fraction = SCREEN_FRACTION
        lrs = FULL_LRS
        lengths = FULL_LENGTHS
        fractions = FULL_FRACTIONS
        mode = "08_full"

    sampled = build_nested_indices(y_train, set(fractions) | {screen_fraction})
    folder = ROOT / "outputs" / mode
    folder.mkdir(parents=True, exist_ok=True)
    hash_value = fingerprint_data(x_train, y_train, x_val, y_val)
    print(f"MODE: {mode}; label count: {len(names)}")
    print(f"Data: {len(x_train)} training, {len(x_val)} validation; no official test data")
    print(f"Screen: {len(lrs) * len(lengths)} candidate runs on {screen_fraction:.0%} training data")
    print(f"Size curve: {len(fractions)} fractions (screen result reused if matching)")
    print("Each run starts from fresh pretrained DistilBERT weights.")
    print("If interrupted, completed trials with matching signatures are reused.")

    def one_run(trial):
        return run_trial(
            trial, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val,
            names=names, sampled_indices=sampled[trial.fraction],
            dataset_hash=hash_value, args=args, prior=prior, folder=folder
        )

    screen_results = [
        one_run(Trial(lr, length, screen_fraction))
        for length in lengths for lr in lrs
    ]
    best = select_best(screen_results)
    selected_lr = float(best["learning_rate"])
    selected_len = int(best["max_length"])
    print(f"Selected at {screen_fraction:.0%}: lr={selected_lr:g}, length={selected_len}")

    curve_results = [
        one_run(Trial(selected_lr, selected_len, f)) for f in fractions
    ]
    write_csv(folder / "08_hyperparameter_search.csv", screen_results,
              ["tag", "n_train", "learning_rate", "max_length",
               "validation_macro_f1", "validation_accuracy", "train_seconds"])
    write_csv(folder / "08_training_size_results.csv", curve_results,
              ["tag", "fraction", "n_train", "train_macro_f1",
               "validation_macro_f1", "validation_accuracy",
               "train_seconds", "best_epoch"])
    plot_screening(screen_results, folder / "08_hyperparameter_comparison.png")
    plot_learning_curve(curve_results, folder / "08_learning_curve.png")
    plot_training_cost(curve_results, folder / "08_training_runtime.png")
    chosen_trial = Trial(selected_lr, selected_len, 1.0)
    manifest = {
        "research_mode": mode, "model": MODEL_ID,
        "selected_learning_rate": selected_lr,
        "selected_max_length": selected_len,
        "screening_fraction": screen_fraction,
        "selection_metric": "validation_macro_f1",
        "training_fractions": list(fractions),
        "training_documents": len(x_train),
        "validation_documents": len(x_val),
        "split_sha256": hash_value,
        "selected_full_model": str(folder / "models" / chosen_trial.tag),
        "notes": [
            "Validation data used to select hyperparameters; scores are exploratory.",
            "Hyperparameters selected at screening fraction, not full training size.",
            "Independent official test partition not loaded.",
        ],
    }
    (folder / "08_selected_configuration.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("\nScreening results:")
    for r in screen_results:
        print(f" {r['tag']:29s}: validation macro-F1 = {r['validation_macro_f1']:.4f}")
    print("\nLearning curve:")
    for r in curve_results:
        print(f" {r['fraction']:>5.0%} ({r['n_train']:>5}): "
              f"train={r['train_macro_f1']:.4f}, val={r['validation_macro_f1']:.4f}")
    print("Selected full-data model:", manifest["selected_full_model"])
    print("Outputs:", folder)
    print("Official test set was NOT loaded.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-checkpoints", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--grad-accumulation", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    args = parser.parse_args()
    if args.self_test:
        offline_self_test()
        return
    if args.quick and args.smoke_test:
        parser.error("Choose either --quick or --smoke-test.")
    if (args.epochs < 1 or args.batch_size < 1 or args.eval_batch_size < 1
            or args.grad_accumulation < 1 or args.weight_decay < 0):
        parser.error("Epochs and batch sizes must be positive; weight decay >= 0.")
    try:
        import torch  # noqa F401
        import transformers  # noqa F401
        import accelerate  # noqa F401
    except ImportError as exc:
        parser.error("Missing dependency. Run: python -m pip install "
                     "torch transformers accelerate scikit-learn numpy matplotlib")
    run(args)


if __name__ == "__main__":
    main()

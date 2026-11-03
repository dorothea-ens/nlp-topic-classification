"""Installment 6 — Optimize MiniLM document classification on VALIDATION only.

Run in the same directory as 05_train_embedding_classifier.py:
    python -m pip install sentence-transformers scikit-learn numpy matplotlib joblib
    python 06_optimize_embeddings.py

Optional: quicker run without the head/tail experiment:
    python 06_optimize_embeddings.py --skip-tail

Experiments:
  A. Tune LogisticRegression C in {0.1, 1, 10} for frozen MiniLM features.
  B. Compare class-stratified NESTED training subsets (10%, 25%, 50%, 100%).
  C. Quantify document truncation and compare first-window representations
     with an average of first and last token windows (head+tail).

Use the same training/validation partition and encoder as Installment 5.
Never load the official test partition; it is reserved for final evaluation.

The full length experiment is potentially slow, especially on CPU. It
encodes the LAST window of each long document but reuses Installment 5's
cached FIRST-window embeddings when possible.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

SEED = 42
ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
C_VALUES = (0.1, 1.0, 10.0)
TRAIN_FRACTIONS = (0.10, 0.25, 0.50, 1.0)
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
CACHE_DIR = ROOT / "cache"
HEAD_CACHE = CACHE_DIR / "05_minilm_embeddings.npz"
TAIL_CACHE = CACHE_DIR / "06_head_tail_embeddings.npz"


def load_training_validation():
    """Precisely reproduce Installments 2–5 (filtered 20 Newsgroups train)."""
    raw = fetch_20newsgroups(
        subset="train", remove=("headers", "footers", "quotes"), random_state=SEED
    )
    pairs = [
        (text, int(label)) for text, label in zip(raw.data, raw.target)
        if text.strip()
    ]
    if not pairs:
        raise RuntimeError("No nonempty training documents found.")
    X, y = zip(*pairs)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    return (
        list(X_train), list(X_val), np.array(y_train), np.array(y_val),
        list(raw.target_names)
    )


def data_fingerprint(X_train, X_val):
    """Match the exact fingerprint used by Installment 5."""
    h = hashlib.sha256()
    h.update(ENCODER_NAME.encode("utf-8"))
    h.update(b"\nnormalized=True\n")
    for name, texts in (("train", X_train), ("validation", X_val)):
        h.update(name.encode("ascii"))
        for text in texts:
            encoded = text.encode("utf-8")
            h.update(len(encoded).to_bytes(8, byteorder="little"))
            h.update(encoded)
    return h.hexdigest()


def get_encoder():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError(
            "Install dependencies: python -m pip install sentence-transformers"
        ) from e
    print("Loading encoder:", ENCODER_NAME)
    return SentenceTransformer(ENCODER_NAME)


def normalize_rows(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


def load_or_encode_head(X_train, X_val, fingerprint, batch_size, encoder_getter):
    """Compatible with 05_minilm_embeddings.npz; independently checks cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if HEAD_CACHE.exists():
        try:
            with np.load(HEAD_CACHE, allow_pickle=False) as f:
                cached_fingerprint = str(f["fingerprint"].item())
                E_train, E_val = f["train"].copy(), f["validation"].copy()
                limit = int(f["max_seq_length"].item())
            if (cached_fingerprint == fingerprint
                and E_train.ndim == E_val.ndim == 2
                and E_train.shape[0] == len(X_train)
                and E_val.shape[0] == len(X_val)
                and E_train.shape[1] == E_val.shape[1]):
                print("Reusing 05_minilm_embeddings.npz for first-window embeddings.")
                return E_train.astype(np.float32), E_val.astype(np.float32), limit
        except (OSError, ValueError, KeyError) as error:
            warnings.warn(f"Couldn't use prior embeddings cache: {error}")

    encoder = encoder_getter()
    limit = int(encoder.max_seq_length)
    print("Encoding first windows; this might take time on CPU...")
    E_train = encoder.encode(
        X_train, batch_size=batch_size, convert_to_numpy=True,
        show_progress_bar=True, normalize_embeddings=True
    ).astype(np.float32)
    E_val = encoder.encode(
        X_val, batch_size=batch_size, convert_to_numpy=True,
        show_progress_bar=True, normalize_embeddings=True
    ).astype(np.float32)
    np.savez_compressed(HEAD_CACHE, train=E_train, validation=E_val,
                        max_seq_length=np.array(limit), fingerprint=np.array(fingerprint))
    return E_train, E_val, limit


def tail_windows(texts, tokenizer, limit):
    """Extract last usable model window; identify docs truncated at the start.

    Uses the exact tokenizer (not whitespace counts) to detect truncation.
    The last window is a *second* view, not a full-document representation.
    """
    specials = int(tokenizer.num_special_tokens_to_add(pair=False))
    budget = limit - specials
    if budget <= 0:
        raise ValueError(f"Invalid tokenizer window budget: {budget}")
    long_positions = []
    last_texts = []
    lengths = np.zeros(len(texts), dtype=np.int32)
    for i, text in enumerate(texts):
        # Explicitly turn off truncation: needed to inspect document length.
        ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
        lengths[i] = len(ids) + specials
        if len(ids) > budget:
            long_positions.append(i)
            last_texts.append(tokenizer.decode(
                ids[-budget:], skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            ))
    return np.asarray(long_positions, dtype=np.int64), last_texts, lengths


def load_or_compute_head_tail(X_train, X_val, E_train, E_val,
                              fingerprint, batch_size, limit, encoder_getter):
    """Generate/cached pooled head+tail vectors; cache includes document lengths.

    Short docs retain exactly the original head vector. Long docs obtain a
    normalized average of the first and last independently encoded windows.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if TAIL_CACHE.exists():
        try:
            with np.load(TAIL_CACHE, allow_pickle=False) as f:
                if (str(f["fingerprint"].item()) == fingerprint
                        and int(f["max_seq_length"].item()) == limit
                        and str(f["method"].item()) == "normalized_head_tail_v1"
                        and f["train"].shape == E_train.shape
                        and f["validation"].shape == E_val.shape):
                    print("Reusing cached head+tail document representations.")
                    return (f["train"].copy(), f["validation"].copy(),
                            f["train_lengths"].copy(), f["val_lengths"].copy())
        except (OSError, ValueError, KeyError) as error:
            warnings.warn(f"Couldn't reuse head+tail cache: {error}")

    model = encoder_getter()
    if int(model.max_seq_length) != limit:
        raise RuntimeError("Encoder maximum length disagrees with prior cached embeddings.")
    pooled_arrays = []
    length_arrays = []
    for name, texts, head in (
        ("training", X_train, E_train), ("validation", X_val, E_val)
    ):
        print(f"Tokenizing {name} documents to identify truncated inputs...")
        positions, tail_texts, lengths = tail_windows(texts, model.tokenizer, limit)
        print(f"  {len(positions):,}/{len(texts):,} exceed {limit} model tokens")
        combined = head.copy().astype(np.float32)
        if len(positions):
            print(f"Encoding final windows of {len(positions):,} long documents...")
            tail_emb = model.encode(
                tail_texts, batch_size=batch_size,
                convert_to_numpy=True, show_progress_bar=True,
                normalize_embeddings=True,
            ).astype(np.float32)
            if tail_emb.shape != combined[positions].shape:
                raise RuntimeError("Tail embeddings have inconsistent shape.")
            combined[positions] = normalize_rows(head[positions] + tail_emb)
        pooled_arrays.append(combined)
        length_arrays.append(lengths)

    H_train, H_val = pooled_arrays
    train_lengths, val_lengths = length_arrays
    np.savez_compressed(
        TAIL_CACHE, fingerprint=np.array(fingerprint),
        max_seq_length=np.array(limit), method=np.array("normalized_head_tail_v1"),
        train=H_train, validation=H_val, train_lengths=train_lengths,
        val_lengths=val_lengths
    )
    return H_train, H_val, train_lengths, val_lengths


def nested_stratified_indices(y_train):
    """Same deterministic per-class nested sampling as Installment 4."""
    rng = np.random.default_rng(SEED)
    pools = {int(k): rng.permutation(np.flatnonzero(y_train == k))
             for k in np.unique(y_train)}
    subsets = {}
    for fraction in TRAIN_FRACTIONS:
        indices = []
        for k in sorted(pools):
            pool = pools[k]
            n = max(1, round(len(pool) * fraction))
            indices.extend(pool[:n].tolist())
        subsets[fraction] = np.array(sorted(indices), dtype=int)
    for earlier, later in zip(TRAIN_FRACTIONS, TRAIN_FRACTIONS[1:]):
        assert set(subsets[earlier]).issubset(set(subsets[later]))
    return subsets


def score(y_true, predicted, all_labels):
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "macro_precision": float(precision_score(
            y_true, predicted, average="macro", labels=all_labels, zero_division=0)),
        "macro_recall": float(recall_score(
            y_true, predicted, average="macro", labels=all_labels, zero_division=0)),
        "macro_f1": float(f1_score(
            y_true, predicted, average="macro", labels=all_labels, zero_division=0)),
        "weighted_f1": float(f1_score(
            y_true, predicted, average="weighted", labels=all_labels, zero_division=0)),
    }


def make_classifier(C):
    return LogisticRegression(C=C, solver="lbfgs", max_iter=1500,
                              random_state=SEED)


def save_csv(filename, rows, fieldnames):
    with (OUTPUT_DIR / filename).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_hyperparameters(rows):
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in sorted({r["representation"] for r in rows}):
        selected = sorted((r for r in rows if r["representation"] == method),
                          key=lambda r: r["C"])
        ax.plot([r["C"] for r in selected], [r["macro_f1"] for r in selected],
                marker="o", label=method)
    ax.set_xscale("log")
    ax.set_xticks(C_VALUES, labels=[str(v) for v in C_VALUES])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Logistic Regression regularization C (higher = weaker penalty)")
    ax.set_ylabel("Validation macro-F1")
    ax.set_title("Regularization and document representations")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "06_hyperparameter_comparison.png", dpi=170)
    plt.close(fig)


def plot_learning_curves(rows):
    fig, ax = plt.subplots(figsize=(9, 5))
    for method in sorted({r["representation"] for r in rows}):
        selected = sorted((r for r in rows if r["representation"] == method),
                          key=lambda r: r["training_fraction"])
        sizes = [r["training_documents"] for r in selected]
        ax.plot(sizes, [r["validation_macro_f1"] for r in selected],
                marker="o", label=f"{method} validation")
        ax.plot(sizes, [r["training_macro_f1"] for r in selected],
                linestyle="--", marker=".", alpha=0.75, label=f"{method} training")
    ax.set_xlabel("Training documents")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1.02)
    ax.set_title("MiniLM + Logistic Regression: nested training subsets")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "06_learning_curves.png", dpi=170)
    plt.close(fig)


def plot_length_results(rows):
    # Macro-F1 for long docs is diagnostic, not an unbiased model-selection metric.
    groups = list(dict.fromkeys(r["length_group"] for r in rows))
    methods = sorted({r["representation"] for r in rows})
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.8 / len(methods)
    for j, method in enumerate(methods):
        subset = {r["length_group"]: r for r in rows if r["representation"] == method}
        values = [subset[g]["macro_f1"] for g in groups]
        ax.bar(x - 0.4 + (j + 0.5) * width, values, width=width, label=method)
    ax.set_xticks(x, groups)
    ax.set_ylabel("Validation macro-F1 within group")
    ax.set_ylim(0, 1.02)
    ax.set_title("Document length / truncation: diagnostic comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "06_document_length_comparison.png", dpi=170)
    plt.close(fig)


def run_experiments(E_train, E_val, y_train, y_val, class_names,
                    *, H_train=None, H_val=None, val_lengths=None,
                    train_lengths=None, max_seq_length=None,
                    persist=True):
    """Purely array-based experiments; independently testable without model downloads."""
    y_train, y_val = np.asarray(y_train), np.asarray(y_val)
    labels = list(range(len(class_names)))
    representations = {"head_only": (np.asarray(E_train), np.asarray(E_val))}
    if (H_train is None) != (H_val is None):
        raise ValueError("Provide both head+tail arrays or neither.")
    if H_train is not None:
        representations["head_tail_mean"] = (np.asarray(H_train), np.asarray(H_val))
    for name, (train, val) in representations.items():
        if train.shape[0] != len(y_train) or val.shape[0] != len(y_val):
            raise ValueError(f"Row/label mismatch in {name}")
        if train.ndim != 2 or val.ndim != 2 or train.shape[1] != val.shape[1]:
            raise ValueError(f"Embedding matrices invalid for {name}")

    if persist:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # A: Model-selection based only on validation data.
    search = []
    selected = {}
    for name, (train, val) in representations.items():
        best = None
        for C in C_VALUES:
            clf = make_classifier(C)
            start = perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                clf.fit(train, y_train)
            seconds = perf_counter() - start
            for note in caught:
                if issubclass(note.category, ConvergenceWarning):
                    print("Convergence warning:", note.message)
            pred = clf.predict(val)
            s = score(y_val, pred, labels)
            row = {"representation": name, "C": C,
                   "training_documents": len(y_train),
                   "validation_documents": len(y_val), **s,
                   "classifier_fit_seconds": round(seconds, 3)}
            search.append(row)
            print(f"{name:18s} C={C:<4g}: validation macro-F1={s['macro_f1']:.4f}")
            if best is None or (s["macro_f1"], s["accuracy"]) > (
                    best["score"]["macro_f1"], best["score"]["accuracy"]):
                best = {"C": C, "classifier": clf, "score": s}
        selected[name] = best

    # B: Refit each method's selected C at the same four nested sizes.
    sizes = []
    subsets = nested_stratified_indices(y_train)
    for name, (train, val) in representations.items():
        C = selected[name]["C"]
        for fraction, indices in subsets.items():
            clf = make_classifier(C)
            start = perf_counter()
            clf.fit(train[indices], y_train[indices])
            seconds = perf_counter() - start
            train_pred = clf.predict(train[indices])
            pred = clf.predict(val)
            s = score(y_val, pred, labels)
            sizes.append({
                "representation": name, "C": C,
                "training_fraction": fraction,
                "training_documents": len(indices),
                "validation_documents": len(y_val),
                "min_class_count": min(Counter(y_train[indices]).values()),
                "training_macro_f1": float(f1_score(
                    y_train[indices], train_pred, labels=labels,
                    average="macro", zero_division=0)),
                "validation_macro_f1": s["macro_f1"],
                "validation_accuracy": s["accuracy"],
                "validation_macro_precision": s["macro_precision"],
                "validation_macro_recall": s["macro_recall"],
                "validation_weighted_f1": s["weighted_f1"],
                "classifier_fit_seconds": round(seconds, 3),
            })
            print(f"{name:18s} {fraction:>4.0%} train: validation macro-F1={s['macro_f1']:.4f}")

    # C: Compare errors on short/long validation docs using tuned models.
    length_rows = []
    truncated_share = None
    if val_lengths is not None and max_seq_length is not None:
        val_lengths = np.asarray(val_lengths)
        short_mask = val_lengths <= max_seq_length
        long_mask = ~short_mask
        truncated_share = float(np.mean(long_mask))
        masks = {"fits_in_one_window": short_mask,
                 "exceeds_one_window": long_mask}
        for name, (_, val) in representations.items():
            predictions = selected[name]["classifier"].predict(val)
            for group, mask in masks.items():
                if mask.sum() == 0:
                    continue
                s = score(y_val[mask], predictions[mask], labels)
                length_rows.append({
                    "representation": name, "length_group": group,
                    "documents": int(mask.sum()),
                    "fraction_validation": float(np.mean(mask)),
                    **s,
                })

    # Persist best MODELS separately; report clear selection provenance.
    summary = {
        "encoder": ENCODER_NAME,
        "dataset": "20 Newsgroups (no headers, footers, quotes)",
        "seed": SEED,
        "train_size": len(y_train), "validation_size": len(y_val),
        "tested_C": list(C_VALUES),
        "training_fractions": list(TRAIN_FRACTIONS),
        "selection": "maximum validation macro-F1 (accuracy breaks ties)",
        "length_experiment_enabled": H_train is not None,
        "max_seq_length": max_seq_length,
        "training_truncated_fraction": (
            float(np.mean(np.asarray(train_lengths) > max_seq_length))
            if train_lengths is not None and max_seq_length is not None else None
        ),
        "validation_truncated_fraction": truncated_share,
        "selected": {
            name: {"C": result["C"], **result["score"]}
            for name, result in selected.items()
        },
        "caveats": [
            "The tokenizer length is not the same as a whitespace word count.",
            "Head+tail pools first and last windows; it does not encode the complete long document.",
            "Separate selected C per representation, using the same validation partition.",
            "All length-group scores are diagnostic; C was selected on this same validation partition.",
            "Timing values exclude encoder loading and embedding generation, and are hardware-dependent.",
            "The official test partition remains untouched; final results require independent test evaluation."
        ]
    }
    if persist:
        save_csv("06_hyperparameter_search.csv", search,
                 ["representation", "C", "training_documents", "validation_documents",
                  "accuracy", "macro_precision", "macro_recall", "macro_f1",
                  "weighted_f1", "classifier_fit_seconds"])
        save_csv("06_training_size_results.csv", sizes,
                 ["representation", "C", "training_fraction", "training_documents",
                  "validation_documents", "min_class_count", "training_macro_f1",
                  "validation_macro_f1", "validation_accuracy", "validation_macro_precision",
                  "validation_macro_recall", "validation_weighted_f1",
                  "classifier_fit_seconds"])
        if length_rows:
            save_csv("06_document_length_results.csv", length_rows,
                     ["representation", "length_group", "documents", "fraction_validation",
                      "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"])
            plot_length_results(length_rows)
        plot_hyperparameters(search)
        plot_learning_curves(sizes)
        (OUTPUT_DIR / "06_best_configuration.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        joblib.dump({
            "encoder_name": ENCODER_NAME,
            "normalize_embeddings": True,
            "class_names": class_names,
            "models": {name: {"classifier": selected[name]["classifier"],
                              "representation": name, "C": selected[name]["C"]}
                       for name in selected},
            "note": "Reload the encoder separately. For head_tail_mean, use the"
                    " same first/last-window aggregation as script 06."
        }, MODEL_DIR / "06_selected_embedding_classifiers.joblib")

    print("\nBest validation configurations:")
    for name, best in selected.items():
        print(f"  {name}: C={best['C']}, macro-F1={best['score']['macro_f1']:.4f}")
    if truncated_share is not None:
        print(f"Validation documents truncated at {max_seq_length} tokens: {truncated_share:.1%}")
    return summary, search, sizes, length_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--skip-tail", action="store_true",
                        help="Faster: tune and plot learning curves for head-only embeddings")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")

    X_train, X_val, y_train, y_val, class_names = load_training_validation()
    fingerprint = data_fingerprint(X_train, X_val)
    encoder = None
    def encoder_getter():
        nonlocal encoder
        if encoder is None:
            encoder = get_encoder()
        return encoder

    E_train, E_val, limit = load_or_encode_head(
        X_train, X_val, fingerprint, args.batch_size, encoder_getter)
    kwargs = {"max_seq_length": limit}
    if not args.skip_tail:
        H_train, H_val, train_lengths, val_lengths = load_or_compute_head_tail(
            X_train, X_val, E_train, E_val, fingerprint,
            args.batch_size, limit, encoder_getter)
        kwargs.update(H_train=H_train, H_val=H_val,
                      train_lengths=train_lengths, val_lengths=val_lengths)
    else:
        print("--skip-tail: omitting truncation and head/tail representation experiments.")

    run_experiments(E_train, E_val, y_train, y_val, class_names, **kwargs)
    print("\nSaved outputs in", OUTPUT_DIR)
    print("Trained classifier(s) in", MODEL_DIR)
    print("Important: no official test data were loaded.")


if __name__ == "__main__":
    main()

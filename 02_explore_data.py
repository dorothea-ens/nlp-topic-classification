"""Installment 2: Exploratory data analysis for supervised topic classification.

Run:
    python 02_explore_data.py

Outputs (created alongside this script):
    outputs/dataset_summary.csv
    outputs/class_distribution.png
    outputs/document_lengths.png

Only the training split is used for exploratory figures and statistics.
The official test set is counted but not inspected or used for model selection.
"""

from collections import Counter
import csv
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")  # Save charts without requiring a GUI.
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def nonempty_pairs(dataset):
    """Remove documents made empty by metadata removal."""
    return [
        (text, int(label))
        for text, label in zip(dataset.data, dataset.target)
        if text.strip()
    ]


def token_count(text):
    """Count approximate word tokens (not transformer subword tokens)."""
    return len(re.findall(r"\b\w+(?:['’-]\w+)*\b", text))


def main():
    print("Loading 20 Newsgroups dataset (first run may download it)...")
    dataset_kwargs = {
        "remove": ("headers", "footers", "quotes"),
        "random_state": RANDOM_STATE,
    }
    full_train = fetch_20newsgroups(subset="train", **dataset_kwargs)
    full_test = fetch_20newsgroups(subset="test", **dataset_kwargs)

    train_pairs = nonempty_pairs(full_train)
    test_pairs = nonempty_pairs(full_test)

    X, y = zip(*train_pairs)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    class_names = full_train.target_names

    counts = Counter(y_train)
    lengths = np.array([token_count(text) for text in X_train], dtype=int)

    print(f"Training:   {len(X_train):,} documents")
    print(f"Validation: {len(X_val):,} documents")
    print(f"Test:       {len(test_pairs):,} documents")
    print(f"Topics:     {len(class_names)}")
    print(f"Training words/document: median={np.median(lengths):.0f}, "
          f"mean={np.mean(lengths):.1f}, 95th percentile={np.percentile(lengths, 95):.0f}")

    # CSV contains descriptive statistics from training only, plus split counts.
    with (OUTPUT_DIR / "dataset_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        writer.writerows([
            ("training_documents", len(X_train)),
            ("validation_documents", len(X_val)),
            ("test_documents", len(test_pairs)),
            ("number_of_classes", len(class_names)),
            ("training_median_word_count", round(float(np.median(lengths)), 2)),
            ("training_mean_word_count", round(float(np.mean(lengths)), 2)),
            ("training_p95_word_count", round(float(np.percentile(lengths, 95)), 2)),
        ])
        writer.writerow([])
        writer.writerow(["class_name", "training_documents"])
        for class_id, class_name in enumerate(class_names):
            writer.writerow([class_name, counts.get(class_id, 0)])

    # Figure 1: class balance in the training partition.
    labels = [class_names[i] for i in range(len(class_names))]
    values = [counts.get(i, 0) for i in range(len(class_names))]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(labels, values)
    ax.invert_yaxis()
    ax.set_xlabel("Number of training documents")
    ax.set_title("20 Newsgroups: training documents per topic")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "class_distribution.png", dpi=180)
    plt.close(fig)

    # Figure 2: word counts. Exclude top 1% from the PLOT only, not the dataset.
    p99 = float(np.percentile(lengths, 99))
    plotted_lengths = lengths[lengths <= p99]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(plotted_lengths, bins=45)
    ax.set_xlabel("Approximate word tokens per document")
    ax.set_ylabel("Number of training documents")
    ax.set_title("Training document lengths (top 1% omitted from plot)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "document_lengths.png", dpi=180)
    plt.close(fig)

    print("\nFiles saved in:", OUTPUT_DIR)
    for filename in (
        "dataset_summary.csv", "class_distribution.png", "document_lengths.png"
    ):
        print(" -", filename)


if __name__ == "__main__":
    main()

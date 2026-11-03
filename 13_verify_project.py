"""Installment 13: lightweight source/artifact verification without downloads.

Usage:
    python 13_verify_project.py
    python 13_verify_project.py --json
    python 13_verify_project.py --strict  # require all three optimized models
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCES = [
    "02_explore_data.py", "03_train_svm_baseline.py", "04_optimize_svm.py",
    "05_train_embedding_classifier.py", "06_optimize_embeddings.py",
    "07_finetune_distilbert.py", "08_optimize_distilbert.py",
    "09_compare_models.py", "10_cross_genre_generalization.py",
    "11_error_analysis.py", "12_streamlit_app.py", "nlp_inference.py",
    "16_stemming_ablation.py", "17_release_audit.py",
]
ARTIFACTS = {
    "optimized_svm": "models/04_selected_svm.joblib",
    "optimized_minilm": "models/06_selected_embedding_classifiers.joblib",
    "minilm_selection_manifest": "outputs/06_best_configuration.json",
    "optimized_distilbert_manifest": "outputs/08_full/08_selected_configuration.json",
    "final_test_results": "outputs/09_final_comparison/09_test_scores.json",
    "error_analysis": "outputs/11_error_analysis/11_analysis_manifest.json",
}
PACKAGES = ["numpy", "scikit-learn", "matplotlib", "scipy", "joblib", "pandas",
            "torch", "transformers", "accelerate", "sentence-transformers",
            "datasets", "streamlit", "pytest"]


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit(root: Path = ROOT) -> dict:
    root = Path(root)
    source_info = {}
    for name in SOURCES:
        path = root / name
        if path.is_file():
            source_info[name] = {"present": True, "sha256": sha256_file(path)}
        else:
            source_info[name] = {"present": False, "sha256": None}
    artifact_info = {key: (root / relative).exists()
                     for key, relative in ARTIFACTS.items()}
    # Do not infer model readiness from a manifest alone: verify the selected
    # directory contains an actual model configuration.
    model_dir_ready = False
    manifest = root / ARTIFACTS["optimized_distilbert_manifest"]
    if manifest.is_file():
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            if payload.get("research_mode") == "08_full":
                recorded = Path(payload["selected_full_model"])
                relocated = manifest.parent / "models" / recorded.name
                chosen = recorded if (recorded / "config.json").is_file() else relocated
                model_dir_ready = (chosen / "config.json").is_file()
        except (OSError, ValueError, KeyError, TypeError):
            model_dir_ready = False
    artifact_info["optimized_distilbert_model_files"] = model_dir_ready
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {name: package_version(name) for name in PACKAGES},
        "sources": source_info,
        "artifacts": artifact_info,
        "all_sources_present": all(x["present"] for x in source_info.values()),
        "all_three_models_ready": all(artifact_info.get(k, False) for k in (
            "optimized_svm", "optimized_minilm", "minilm_selection_manifest",
            "optimized_distilbert_manifest", "optimized_distilbert_model_files")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print JSON report")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero if models missing")
    parser.add_argument("--save", action="store_true", help="Save report in outputs/13_environment_audit.json")
    args = parser.parse_args()
    report = audit()
    if args.save:
        path = ROOT / "outputs" / "13_environment_audit.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Python {report['python_version']} | source files: "
              f"{sum(x['present'] for x in report['sources'].values())}/{len(SOURCES)}")
        for key, ready in report["artifacts"].items():
            print(f"{'READY' if ready else 'MISSING'}: {key}")
        if not report["all_three_models_ready"]:
            print("Model weights are intentionally excluded from the publication package; local full-model verification requires the original trained artifacts.")
    if not report["all_sources_present"] or (args.strict and not report["all_three_models_ready"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

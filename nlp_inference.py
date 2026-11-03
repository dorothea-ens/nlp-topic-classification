"""Installment 12 — inference for optimized NLP topic classifiers.

Compatible with optimized artifacts produced by Installments 4, 6, and 8.
This module does NOT load test documents or train/tune any model.

Use only joblib files you created yourself or trust: deserialization can execute code.

Offline tests (no model downloads): python nlp_inference.py --self-test
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np

SVM_NAME = "TF-IDF + Linear SVM"
EMBEDDING_NAME = "MiniLM + Logistic Regression"
TRANSFORMER_NAME = "Fine-tuned DistilBERT"
MODEL_ORDER = (SVM_NAME, EMBEDDING_NAME, TRANSFORMER_NAME)


@dataclass(frozen=True)
class ArtifactLocation:
    kind: str
    path: Path
    manifest: Path | None = None


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def locate_artifacts(root: Path) -> dict[str, ArtifactLocation]:
    """Discover the same validation-selected models used by final comparison.

    Never use test-set scores to choose an artifact. Prefer full optimization
    if available; otherwise accept the 100%-training-data exploratory selection.
    """
    root = Path(root)
    found: dict[str, ArtifactLocation] = {}
    svm = root / "models" / "04_selected_svm.joblib"
    if svm.is_file():
        found[SVM_NAME] = ArtifactLocation("svm", svm)

    embedding = root / "models" / "06_selected_embedding_classifiers.joblib"
    embedding_manifest = root / "outputs" / "06_best_configuration.json"
    if embedding.is_file() and embedding_manifest.is_file():
        found[EMBEDDING_NAME] = ArtifactLocation("embedding", embedding, embedding_manifest)

    from pathlib import PureWindowsPath
    for mode in ("08_full", "08_quick_EXPLORATORY"):
        manifest_path = root / "outputs" / mode / "08_selected_configuration.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = read_json(manifest_path)
            if manifest.get("research_mode") != mode:
                continue
            recorded = str(manifest["selected_full_model"])
            tag = PureWindowsPath(recorded).name
            relocated = root / "outputs" / mode / "models" / tag
            original = Path(recorded)
            candidate = original if original.is_dir() else relocated
            if (candidate / "config.json").is_file() and (
                (candidate / "model.safetensors").is_file() or
                (candidate / "pytorch_model.bin").is_file()
            ):
                found[TRANSFORMER_NAME] = ArtifactLocation(
                    "transformer", candidate, manifest_path
                )
                break
        except (ValueError, KeyError, TypeError, OSError):
            continue
    return found


def _class_names(names: Any) -> list[str]:
    names = list(names)
    if len(names) < 2 or len(names) != len(set(names)):
        raise ValueError("Class list must contain at least two distinct labels.")
    return [str(name) for name in names]


def _check_label_order(expected: list[str], actual: list[str], model: str) -> None:
    if expected != actual:
        raise ValueError(f"{model}: class order differs across trained artifacts.")


def selected_representation(manifest: dict, stored_models: dict) -> str:
    """Exactly reproduce Installment 9 validation-based representation choice."""
    scores = manifest.get("selected", {})
    candidates = [r for r in ("head_only", "head_tail_mean")
                  if r in scores and r in stored_models]
    if not candidates:
        raise ValueError("No compatible saved embedding classifier and validation score.")
    return max(candidates, key=lambda r: (
        float(scores[r]["macro_f1"]), float(scores[r]["accuracy"])
    ))


def load_svm(path: Path) -> tuple[Any, list[str]]:
    data = joblib.load(path)
    classifier = data["pipeline"]
    names = _class_names(data["class_names"])
    if not hasattr(classifier, "decision_function"):
        raise ValueError("Expected Linear SVM with decision_function.")
    return classifier, names


def load_embedding(artifact_path: Path, manifest_path: Path) -> tuple[Any, Any, str, list[str]]:
    data = joblib.load(artifact_path)
    manifest = read_json(manifest_path)
    names = _class_names(data["class_names"])
    rep = selected_representation(manifest, data["models"])
    classifier = data["models"][rep]["classifier"]
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(data["encoder_name"])
    expected_limit = manifest.get("max_seq_length")
    if expected_limit is not None and int(expected_limit) != int(encoder.max_seq_length):
        raise ValueError("MiniLM token limit differs from optimization experiment.")
    return encoder, classifier, rep, names


def load_transformer(model_path: Path, manifest_path: Path) -> tuple[Any, Any, int, str, list[str]]:
    """Use only the existing fine-tuned checkpoint (never base DistilBERT)."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    manifest = read_json(manifest_path)
    if manifest.get("research_mode") not in ("08_full", "08_quick_EXPLORATORY"):
        raise ValueError("Unknown DistilBERT selection protocol.")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path), local_files_only=True
    )
    names = _class_names(
        model.config.id2label[i] for i in range(int(model.config.num_labels))
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    return tokenizer, model, int(manifest["selected_max_length"]), device, names


def _normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    return arr / np.maximum(np.linalg.norm(arr, axis=-1, keepdims=True), 1e-12)


def tail_text(text: str, tokenizer: Any, model_limit: int) -> str | None:
    """Apply exact first/last-wordpiece-window rule from Installment 6."""
    budget = int(model_limit) - int(tokenizer.num_special_tokens_to_add(pair=False))
    if budget <= 0:
        raise ValueError("Tokenizer window budget is invalid.")
    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    if len(tokens) <= budget:
        return None
    return tokenizer.decode(tokens[-budget:], skip_special_tokens=True,
                            clean_up_tokenization_spaces=False)


def _ranking(scores: np.ndarray, names: list[str], *, kind: str) -> list[dict]:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if len(scores) != len(names) or not np.isfinite(scores).all():
        raise ValueError("Prediction scores have wrong shape or are not finite.")
    return [dict(topic=names[int(i)], score=float(scores[int(i)]), score_kind=kind)
            for i in np.argsort(-scores, kind="stable")]


def svm_predict(text: str, pipeline: Any, names: list[str]) -> dict:
    started = perf_counter()
    predicted = int(pipeline.predict([text])[0])
    scores = np.asarray(pipeline.decision_function([text]), dtype=float)[0]
    elapsed = perf_counter() - started
    rankings = _ranking(scores, names, kind="decision margin (not probability)")
    if names[predicted] != rankings[0]["topic"]:
        raise ValueError("SVM class order does not match stored names.")
    return {"model": SVM_NAME, "label": names[predicted], "rankings": rankings,
            "seconds": elapsed, "note": "SVM margins are NOT probabilities."}


def embedding_predict(text: str, encoder: Any, classifier: Any,
                      representation: str, names: list[str]) -> dict:
    started = perf_counter()
    first = np.asarray(encoder.encode(
        [text], convert_to_numpy=True, normalize_embeddings=True,
        show_progress_bar=False
    ), dtype=np.float32)
    used_tail = False
    if representation == "head_tail_mean":
        last = tail_text(text, encoder.tokenizer, int(encoder.max_seq_length))
        if last is not None:
            tail = np.asarray(encoder.encode(
                [last], convert_to_numpy=True, normalize_embeddings=True,
                show_progress_bar=False
            ), dtype=np.float32)
            first = _normalize(first + tail)
            used_tail = True
    probabilities = np.asarray(classifier.predict_proba(first), dtype=float)[0]
    class_ids = np.asarray(classifier.classes_, dtype=int)
    scores = np.full(len(names), np.nan)
    scores[class_ids] = probabilities
    ranking = _ranking(scores, names, kind="classifier probability (uncalibrated)")
    return {"model": EMBEDDING_NAME, "label": ranking[0]["topic"],
            "rankings": ranking, "seconds": perf_counter() - started,
            "note": ("Head + tail windows" if used_tail else "First window only")}


def transformer_predict(text: str, tokenizer: Any, model: Any, max_length: int,
                        device: str, names: list[str]) -> dict:
    import torch

    started = perf_counter()
    tokens = tokenizer([text], padding=True, truncation=True,
                       max_length=max_length, return_tensors="pt")
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.inference_mode():
        logits = model(**tokens).logits
        probabilities = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    if device == "cuda":
        torch.cuda.synchronize()
    rankings = _ranking(probabilities, names, kind="softmax probability (uncalibrated)")
    return {"model": TRANSFORMER_NAME, "label": rankings[0]["topic"],
            "rankings": rankings, "seconds": perf_counter() - started,
            "note": f"Truncated to at most {max_length} tokens"}


def offline_self_test() -> None:
    import tempfile
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import LinearSVC

    texts = ["space rocket mission launch", "cosmic galaxy mission rocket",
             "hockey goal team match", "hockey puck goal team",
             "computer graphics screen mouse", "graphics computer image mouse"]
    labels = [0, 0, 1, 1, 2, 2]
    names = ["space", "sport", "computers"]
    model = make_pipeline(TfidfVectorizer(), LinearSVC())
    model.fit(texts, labels)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "models").mkdir()
        path = root / "models" / "04_selected_svm.joblib"
        joblib.dump({"pipeline": model, "class_names": names}, path)
        located = locate_artifacts(root)
        assert list(located) == [SVM_NAME]
        loaded, actual_names = load_svm(path)
        assert actual_names == names
        pred = svm_predict("rocket launch galaxy", loaded, actual_names)
        assert pred["label"] == "space" and len(pred["rankings"]) == 3
        assert len(set(item["topic"] for item in pred["rankings"])) == 3
        assert pred["rankings"][0]["score"] >= pred["rankings"][-1]["score"]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        checkpoint = root / "outputs" / "08_quick_EXPLORATORY" / "models" / "p100_len256_lr5e-05"
        checkpoint.mkdir(parents=True)
        (checkpoint / "config.json").write_text('{}', encoding="utf-8")
        (checkpoint / "model.safetensors").write_bytes(b"dummy")
        manifest_path = checkpoint.parent.parent / "08_selected_configuration.json"
        manifest_path.write_text(json.dumps({
            "research_mode": "08_quick_EXPLORATORY",
            "selected_full_model": "C:\\old-location\\p100_len256_lr5e-05",
            "selected_max_length": 256,
        }), encoding="utf-8")
        artifacts = locate_artifacts(root)
        assert TRANSFORMER_NAME in artifacts
        assert artifacts[TRANSFORMER_NAME].path == checkpoint
        manifest_path.write_text(json.dumps({
            "research_mode": "08_quick_EXPLORATORY",
            "selected_full_model": "C:\\old-location\\p100_len256_lr5e-05",
            "selected_max_length": 256,
        }), encoding="utf-8")
        (checkpoint / "model.safetensors").unlink()
        assert TRANSFORMER_NAME not in locate_artifacts(root)

    class FakeTokenizer:
        def num_special_tokens_to_add(self, pair=False):
            return 2
        def encode(self, text, **kw):
            return text.split()
        def decode(self, toks, **kw):
            return " ".join(toks)

    assert tail_text("one two three", FakeTokenizer(), 5) is None
    assert tail_text("a b c d e f", FakeTokenizer(), 5) == "d e f"
    assert selected_representation({"selected": {
        "head_only": {"macro_f1": .6, "accuracy": .7},
        "head_tail_mean": {"macro_f1": .6, "accuracy": .7}}},
        {"head_only": {}, "head_tail_mean": {}}) == "head_only"
    assert np.allclose(_normalize([[3, 4]]), [[.6, .8]])
    assert [r["topic"] for r in _ranking(np.array([.2, .8, .5]), names,
                                         kind="test")] == ["sport", "computers", "space"]
    print("PASS: artifact discovery, loading, SVM inference, token windows,"
          " embedding representation selection, vector normalization, ranking.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        offline_self_test()
    else:
        parser.error("Use --self-test. For the app run: streamlit run 12_streamlit_app.py")

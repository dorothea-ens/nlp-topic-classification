"""Installment 12: Interactive 20 Newsgroups document classifier.

Run: streamlit run 12_streamlit_app.py
Loads the same saved selected artifacts used in the final comparison. No retraining occurs.
"""
from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st

from nlp_inference import (
    EMBEDDING_NAME, MODEL_ORDER, SVM_NAME, TRANSFORMER_NAME,
    embedding_predict, load_embedding, load_svm, load_transformer,
    locate_artifacts, svm_predict, transformer_predict,
)

ROOT = Path(__file__).resolve().parent
SAMPLE = (
    "Scientists are designing a spacecraft for a mission to study distant "
    "planets. The mission will measure atmospheric composition and relay "
    "observations to researchers on Earth."
)
MAX_CHARS = 20_000


@st.cache_resource(show_spinner=False)
def cached_svm(file_path: str, mtime: int):
    return load_svm(Path(file_path))


@st.cache_resource(show_spinner=False)
def cached_embedding(file_path: str, mtime: int,
                     manifest_path: str, manifest_mtime: int):
    return load_embedding(Path(file_path), Path(manifest_path))


@st.cache_resource(show_spinner=False)
def cached_transformer(model_path: str, manifest_path: str,
                       manifest_mtime: int, config_mtime: int):
    return load_transformer(Path(model_path), Path(manifest_path))


def load_selected_model(name, artifact):
    if name == SVM_NAME:
        return cached_svm(str(artifact.path), artifact.path.stat().st_mtime_ns)
    if name == EMBEDDING_NAME:
        return cached_embedding(
            str(artifact.path), artifact.path.stat().st_mtime_ns,
            str(artifact.manifest), artifact.manifest.stat().st_mtime_ns
        )
    if name == TRANSFORMER_NAME:
        config_path = artifact.path / "config.json"
        return cached_transformer(
            str(artifact.path), str(artifact.manifest),
            artifact.manifest.stat().st_mtime_ns, config_path.stat().st_mtime_ns
        )
    raise ValueError(f"Unknown model: {name}")


def classify(text, selected, available):
    results = []
    reference_labels = None
    for name in selected:
        try:
            with st.spinner(f"Loading and running {name}..."):
                loaded = load_selected_model(name, available[name])
                labels = loaded[-1]
                if reference_labels is not None and labels != reference_labels:
                    raise ValueError("Label orders differ; cannot compare these models.")
                reference_labels = labels
                if name == SVM_NAME:
                    pipeline, labels = loaded
                    result = svm_predict(text, pipeline, labels)
                elif name == EMBEDDING_NAME:
                    encoder, classifier, rep, labels = loaded
                    result = embedding_predict(text, encoder, classifier, rep, labels)
                else:
                    tokenizer, model, max_length, device, labels = loaded
                    result = transformer_predict(
                        text, tokenizer, model, max_length, device, labels
                    )
                results.append(result)
        except Exception as error:
            st.error(f"{name} could not run: {error}")
    return results


def results_csv(results) -> str:
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["model", "rank", "category", "score", "score_type",
                     "inference_seconds", "notes"])
    for result in results:
        for i, row in enumerate(result["rankings"], start=1):
            writer.writerow([
                result["model"], i, row["topic"], f'{row["score"]:.8f}',
                row["score_kind"], f'{result["seconds"]:.5f}', result["note"]
            ])
    return output.getvalue()


def main():
    st.set_page_config(page_title="NLP Topic Classifier", layout="wide")
    st.title("Document Topic Classification")
    st.caption("University NLP Project · 20 Newsgroups · Three ML architectures")
    st.write(
        "Classify a text document using trained models from the NLP project. "
        "This app performs **inference only**; it does not retrain or change any model."
    )

    available = locate_artifacts(ROOT)
    with st.sidebar:
        st.header("Available models")
        for model in MODEL_ORDER:
            st.write(("✓ " if model in available else "— ") + model)
        if TRANSFORMER_NAME in available:
            mode = available[TRANSFORMER_NAME].manifest.parent.name
            if mode == "08_quick_EXPLORATORY":
                st.warning("DistilBERT uses the final evaluation's 100%-data "
                           "checkpoint selected via an abbreviated exploratory "
                           "hyperparameter search, not an exhaustive search.")
        st.caption("Saved, validation-selected artifacts only; no models are "
                   "retrained or selected using the test set.")
        st.markdown("**Required training scripts:** `04_optimize_svm.py`, "
                    "`06_optimize_embeddings.py`, `08_optimize_distilbert.py`.")

    if not available:
        st.warning("No trained model artifacts were found. Train at least one "
                   "optimized classifier using the scripts above, keeping "
                   "their outputs and models folders beside this application.")
        st.stop()

    chosen = st.multiselect(
        "Models to use", options=[m for m in MODEL_ORDER if m in available],
        default=[m for m in MODEL_ORDER if m in available]
    )
    uploaded = st.file_uploader("Optional: upload a UTF-8 .txt document", type=["txt"])
    text_value = st.text_area("Or paste a document", value=SAMPLE, height=180)
    if uploaded is not None:
        raw_bytes = uploaded.getvalue()
        try:
            document = raw_bytes.decode("utf-8-sig")
        except UnicodeDecodeError:
            st.error("The uploaded file must be UTF-8 encoded.")
            st.stop()
        st.info("Using the uploaded file instead of the text area.")
    else:
        document = text_value

    left, right = st.columns(2)
    left.metric("Approximate words", len(document.split()))
    right.metric("Characters", len(document))
    if len(document) > MAX_CHARS:
        st.error(f"Please use at most {MAX_CHARS:,} characters per classification.")
        st.stop()

    run_button = st.button("Classify document", type="primary",
                           disabled=not chosen or not document.strip())
    if run_button:
        output = classify(document, chosen, available)
        if not output:
            st.warning("No models completed inference.")
            return
        st.subheader("Predicted topics")
        summary = pd.DataFrame([
            {"Model": r["model"], "Predicted topic": r["label"],
             "Inference seconds": round(r["seconds"], 3), "Representation": r["note"]}
            for r in output
        ])
        st.dataframe(summary, hide_index=True, use_container_width=True)

        st.subheader("Top categories per model")
        for result in output:
            with st.expander(result["model"], expanded=True):
                top = result["rankings"][:5]
                st.dataframe(pd.DataFrame([
                    {"Topic": row["topic"],
                     "Score": round(row["score"], 4),
                     "Score type": row["score_kind"]}
                    for row in top
                ]), hide_index=True, use_container_width=True)
                if result["model"] == SVM_NAME:
                    st.caption("SVM decision margins are not probabilities and "
                               "may be negative.")
                else:
                    st.caption("Displayed probabilities have not been "
                               "calibrated. They do not guarantee correctness.")

        st.download_button(
            "Export these predictions as CSV", data=results_csv(output),
            file_name="12_document_predictions.csv", mime="text/csv"
        )
        st.caption("Prediction time includes preprocessing and inference, "
                   "but excludes model-loading time; values depend on hardware.")

    st.divider()
    st.caption(
        "Research prototype: categories are the original 20 Newsgroups labels. "
        "The app is not a factual verification system, and inputs far outside "
        "the training domain may yield unreliable predictions. Text entered "
        "into this app is processed by the running application server."
    )


if __name__ == "__main__":
    main()

# Streamlit demonstration: local end-to-end smoke test

## What was demonstrated

On 1 November 2026, the project author launched the local Streamlit interface using the **previously saved, validation-selected** SVM, MiniLM and DistilBERT artifacts. No training or test-set-based re-selection occurred in the application. A single authored spacecraft-related example was submitted and `12_document_predictions.csv` was exported from the interface. The updated release contains the same compatibility fix tested locally.

**Input (25 words):**

> Scientists are designing a spacecraft for a mission to study distant planets. The mission will measure atmospheric composition and relay observations to researchers on Earth.

**Top-1 category from the actual exported CSV:**

| Model | Top-1 label | One-off inference elapsed time |
|---|---|---:|
| TF-IDF + Linear SVM | `sci.space` | 0.27534 s |
| MiniLM + Logistic Regression | `sci.space` | 0.08628 s |
| Fine-tuned DistilBERT | `sci.space` | 0.04517 s |

The full export lists **20 ranked labels per model**, available in [`../results/streamlit_demo_predictions.csv`](../results/streamlit_demo_predictions.csv). The input and interface are visible in the [local application screenshot](streamlit_app_screenshot.png).

## How it works

1. Streamlit accepts entered text or an uploaded UTF-8 `.txt` file.
2. `nlp_inference.py` locates the saved SVM, embedding classifier and DistilBERT checkpoint; the models load on first use and are cached for subsequent requests.
3. **SVM:** previously fitted TF-IDF vectorizer transforms the text; a fitted linear SVM scores the 20 categories. Its decision margins are **not** probabilities.
4. **MiniLM:** frozen sentence embeddings are computed, then the fitted logistic-regression classifier ranks 20 labels. Displayed classifier probabilities have not been calibrated.
5. **DistilBERT:** the saved fine-tuned tokenizer and transformer (at most 256 input tokens) score the same categories; softmax outputs are uncalibrated.
6. The interface displays predictions and scores and makes the complete ranked results downloadable as CSV.

## Scope and limitations

This demonstrates functional connectivity between the browser UI and **all three locally stored trained classifiers** on one document, not generalization, systematic timing performance, security, or online deployment. The held-out study of 7,317 examples, documented separately in the public methods summary, is the scientific basis for comparative model quality. The observed one-off prediction times cannot be compared directly with the report's model throughput table because setup and timing protocols differ. Model weights are not in this publication archive; other users must train or obtain compatible saved artifacts to reproduce the interactive inference. The final DistilBERT checkpoint was selected through abbreviated exploratory search but fine-tuned on the full training partition.

## Running locally with preserved artifacts

Place the trained model files under the project layout expected by `nlp_inference.py`. From the repository root, install `requirements-full.txt` (or at least the required application and inference dependencies) and run:

```powershell
python -m streamlit run 12_streamlit_app.py
```

The app will ordinarily open at `http://localhost:8501`. Check that the sidebar lists all three available models and use **Classify document**. Do not use the official held-out test documents to tune the model or select checkpoints.

## Evidence integrity

`streamlit_demo_predictions.csv` is copied from the author's exported `12_document_predictions.csv`. The screenshot is copied from the author's application view. The verification audit checks the structure of the public export; it does not independently authenticate the user's local machine or model weights.

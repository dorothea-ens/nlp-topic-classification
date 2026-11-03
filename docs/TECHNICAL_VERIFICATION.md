# Technical verification summary

This is an offline audit of the source code and published results, not a demonstration of complete training reproducibility.

**Source verification:** Python syntax check, eight existing synthetic-data pytest tests, seven offline script self-tests, and one additional release-integrity test.

**Raw held-out predictions:** The privately supplied `09_test_predictions.csv` contains 7,317 rows with 20 distinct topic labels. Recomputing accuracy, macro-F1, macro-precision and macro-recall independently using scikit-learn reproduced all reported values for SVM, MiniLM and DistilBERT exactly (within floating-point precision). This row-level file is excluded from the publication package.

**Published results:** `results/` contains aggregate test scores, bootstrap intervals, pairwise comparisons, supplementary experiments, and figures. The release audit checks internal arithmetic, public-methods/README consistency, source presence, and likely private temporary filenames.

**Interactive application verification (additional evidence):** After the offline audit, the author restored the saved artifacts and launched Streamlit locally. The application showed all three selected classifiers available, and an exported CSV recorded 20 ranked classes per model. The spacecraft sample was classified as `sci.space` by SVM, MiniLM and DistilBERT. The updated loader supports the same 100%-training-data `08_quick_EXPLORATORY` DistilBERT checkpoint as the final held-out evaluation, notwithstanding abbreviated exploratory hyperparameter selection. A screenshot and the exported CSV are included under `docs/` and `results/`. This is **one successful local end-to-end smoke test** observed through user-supplied evidence, not an independent reproduction of model training, a general accuracy estimate, or proof of deployment readiness.

**Remaining limitations:** Trained weights remain on the author's local machine and are excluded from this public archive. One-off Streamlit inference timings are not CPU throughput benchmarks; model-loading time and device differences make them noncomparable to `results/main_test_comparison.csv` inference throughput. The app is not publicly deployed or stress/security tested. GitHub Actions has not been run against a published repository; its configuration is present for post-publication checks.

**Publication note:** The formal academic examination document is deliberately omitted from the public portfolio. Source files, aggregates, charts, tests, and the public technical summary remain. Review repository visibility and applicable institutional rules before publishing.

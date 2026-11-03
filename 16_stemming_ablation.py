"""A clearly labeled VALIDATION-ONLY stemmer ablation for IU NLP Task 3.

Run: python 16_stemming_ablation.py
Test: python 16_stemming_ablation.py --self-test

Compare an identical TF-IDF + LinearSVC pipeline with and without English
Snowball stemming, matching selected 04 settings (unigrams+bigrams, min_df=2,
max_df=.95, max_features=100k, sublinear_tf, C=1). This does not replace
or tune the existing selected model; no official test data are accessed.
NLTK SnowballStemmer does NOT require downloading any NLTK corpora.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs' / '16_stemming_ablation'
RANDOM_STATE = 42
PATTERN = re.compile(r'(?u)\b\w\w+\b')
STEMMER = SnowballStemmer('english')


def stem_tokenizer(text: str) -> list[str]:
    # TfidfVectorizer lowercases before calling this tokenizer.
    # Apply same English stop words BEFORE stemming so stopword removal works.
    return [STEMMER.stem(token) for token in PATTERN.findall(text)
            if token not in ENGLISH_STOP_WORDS]


def build_model(stemming: bool, min_df: int = 2, max_df: float = .95):
    kwargs = dict(lowercase=True, ngram_range=(1, 2), min_df=min_df,
                  max_df=max_df, max_features=100_000, sublinear_tf=True,
                  dtype=np.float32)
    if stemming:
        kwargs.update(tokenizer=stem_tokenizer, token_pattern=None, stop_words=None)
    else:
        kwargs.update(stop_words='english')
    return Pipeline([
        ('tfidf', TfidfVectorizer(**kwargs)),
        ('classifier', LinearSVC(C=1.0, max_iter=5000, random_state=RANDOM_STATE)),
    ])


def evaluate(X_train, y_train, X_val, y_val, labels, out=OUT, min_df=2):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    predictions = {}
    for do_stem, name in ((False, 'without_stemming'), (True, 'with_snowball_stemming')):
        model = build_model(do_stem, min_df=min_df)
        start = perf_counter()
        model.fit(X_train, y_train)
        elapsed = perf_counter() - start
        pred = np.asarray(model.predict(X_val))
        predictions[name] = pred
        row = dict(
            condition=name, validation_documents=len(X_val),
            training_documents=len(X_train),
            vocabulary_features=len(model.named_steps['tfidf'].vocabulary_),
            accuracy=float(accuracy_score(y_val, pred)),
            macro_precision=float(precision_score(y_val, pred, average='macro',
                                                   labels=labels, zero_division=0)),
            macro_recall=float(recall_score(y_val, pred, average='macro',
                                            labels=labels, zero_division=0)),
            macro_f1=float(f1_score(y_val, pred, average='macro', labels=labels,
                                    zero_division=0)),
            weighted_f1=float(f1_score(y_val, pred, average='weighted',
                                       labels=labels, zero_division=0)),
            training_seconds=round(elapsed, 3),
        )
        rows.append(row)
        print(f'{name}: macro-F1={row["macro_f1"]:.4f}, accuracy={row["accuracy"]:.4f}')

    with (out / '16_stemming_validation_comparison.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = ['No stemming', 'Snowball stemming']
    xs = np.arange(2)
    ax.bar(xs - .17, [r['macro_f1'] for r in rows], width=.34, label='Macro-F1')
    ax.bar(xs + .17, [r['accuracy'] for r in rows], width=.34, label='Accuracy')
    ax.set_xticks(xs, names)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Validation score')
    ax.set_title('English stemming ablation (C=1, fixed TF-IDF settings)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / '16_stemming_validation_comparison.png', dpi=160)
    plt.close(fig)

    metadata = dict(
        dataset='20 Newsgroups', partition='validation only', train_n=len(X_train),
        validation_n=len(X_val), official_test_used=False, seed=RANDOM_STATE,
        methods=['unaltered selected SVM feature settings','same with English Snowball stemming'],
        preprocessing={'remove': ['headers','footers','quotes'], 'lowercase': True,
                       'english_stopwords_before_stemming': True,
                       'snowball_english': True},
        classifier={'model':'LinearSVC', 'C':1.0, 'max_iter':5000},
        features={'ngram_range':[1,2], 'min_df':min_df, 'max_df':.95,
                  'max_features':100000, 'sublinear_tf':True},
        result_rows=rows,
        interpretation='Diagnostic validation ablation only; neither condition is chosen '
                       'for test-set selection after seeing held-out results. '
                       'The preselected Installment 4 model remains unchanged.',
    )
    (out / '16_stemming_validation_metadata.json').write_text(
        json.dumps(metadata, indent=2), encoding='utf-8')
    return rows


def load_existing_split():
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location('installment_04', ROOT / '04_optimize_svm.py')
    if spec is None or spec.loader is None:
        raise FileNotFoundError('04_optimize_svm.py must be in the same folder.')
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_training_validation()


def self_test():
    import tempfile
    assert stem_tokenizer('Running runners and ran quickly!') == [
        'run', 'runner', 'ran', 'quick']
    X = ['football player scored the decisive goal', 'hockey puck ice rink match',
         'astronomy space telescope nebula orbit', 'spacecraft lunar planet launch',
         'goalkeeper rugby athletes compete', 'satellite galaxy astronomer science',
         'sport champions tournament final', 'cosmology telescope solar galaxy']
    y = np.array([0,0,1,1,0,1,0,1]); labels = [0,1]
    with tempfile.TemporaryDirectory() as tmp:
        rows = evaluate(X,y,X,y,labels,out=Path(tmp),min_df=1)
        assert len(rows)==2
        assert all(0 <= r['macro_f1'] <= 1 for r in rows)
        assert (Path(tmp)/'16_stemming_validation_metadata.json').is_file()
        assert (Path(tmp)/'16_stemming_validation_comparison.png').is_file()
    print('SELF-TEST PASSED; synthetic documents only, no test-set access.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    metadata = OUT / '16_stemming_validation_metadata.json'
    if metadata.is_file():
        print(f'Ablation already completed; not repeating: {metadata}')
        return
    X_train, X_val, y_train, y_val, class_names = load_existing_split()
    print('Diagnostic comparison on validation set ONLY; official test remains untouched.')
    evaluate(X_train, y_train, X_val, y_val, list(range(len(class_names))))
    print('Saved:', OUT)


if __name__ == '__main__':
    main()

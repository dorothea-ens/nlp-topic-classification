"""Offline audit of the public NLP project release (no model downloads).

Usage: python 17_release_audit.py
Checks: required source files, aggregate score arithmetic, paired comparisons,
public-methods/README consistency, and accidental inclusion of restricted working files.
It DOES NOT establish that saved model weights can be loaded or reproduce training.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ['02_explore_data.py','03_train_svm_baseline.py','04_optimize_svm.py',
    '05_train_embedding_classifier.py','06_optimize_embeddings.py',
    '07_finetune_distilbert.py','08_optimize_distilbert.py','09_compare_models.py',
    '10_cross_genre_generalization.py','11_error_analysis.py',
    '12_streamlit_app.py','nlp_inference.py','13_verify_project.py',
    '16_stemming_ablation.py','17_release_audit.py']
BANNED_PARTS = ('_results_for_', 'completion_log', '\\users\\', '.acsm')


def rows(path: Path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def audit(root: Path=ROOT):
    errors = []
    warnings = []
    for name in SOURCE:
        if not (root/name).is_file():
            errors.append('Missing source file: '+name)
    required = ['results/main_test_comparison.csv','results/paired_bootstrap_vs_svm.csv',
                'results/pairwise_tests.csv','results/stemming_validation.csv',
                'results/cross_genre_test.csv','results/figures/model_comparison.png',
                'docs/METHODS_AND_RESULTS.md','README.md',
                'results/streamlit_demo_predictions.csv',
                'docs/STREAMLIT_DEMONSTRATION.md','docs/streamlit_app_screenshot.png']
    for name in required:
        if not (root/name).is_file():
            errors.append('Missing release artifact: '+name)
    for path in root.rglob('*'):
        if not path.is_file(): continue
        relative = path.relative_to(root).as_posix().lower()
        if any(part in relative for part in ('__pycache__/', '.pytest_cache/', '.git/')):
            continue
        if any(p in relative for p in BANNED_PARTS):
            errors.append('Potential private/temporary path: '+str(path.relative_to(root)))
        if path.suffix.lower() in ('.key','.pem','.acsm','.docx','.pdf'):
            errors.append('Restricted or formal-examination file: '+relative)
        if relative.startswith(('report/','submission/','submissions/')):
            errors.append('Non-public academic-submission folder: '+relative)
    if errors: return errors,warnings

    scores=rows(root/'results/main_test_comparison.csv')
    bs=rows(root/'results/paired_bootstrap_vs_svm.csv')
    pairs=rows(root/'results/pairwise_tests.csv')
    if len(scores)!=3 or {r['model'] for r in scores} != {
        'TF-IDF + Linear SVM','MiniLM + Logistic Regression','Fine-tuned DistilBERT'}:
        errors.append('Unexpected main model inventory')
    by_model={r['model']:r for r in scores}
    for r in scores:
        if int(r['test_documents'])!=7317:
            errors.append('Unequal test populations: '+r['model'])
        for metric in ['accuracy','macro_precision','macro_recall','macro_f1','weighted_f1']:
            v=float(r[metric]);
            if not (0<=v<=1): errors.append('Out-of-bounds metric: '+r['model']+'/'+metric)
        rate=int(r['test_documents'])/float(r['inference_seconds'])
        if abs(rate-float(r['documents_per_second']))>1e-4:
            errors.append('Throughput arithmetic mismatch: '+r['model'])
    if len(bs)!=2: errors.append('Expected two bootstrap baselines')
    for r in bs:
        if r['model'] not in by_model: errors.append('Unknown bootstrap model')
        else:
            delta=(float(by_model[r['model']]['macro_f1']) -
                   float(by_model['TF-IDF + Linear SVM']['macro_f1']))
            if abs(delta-float(r['delta_macro_f1']))>1e-10:
                errors.append('Bootstrap delta differs from test metrics: '+r['model'])
    if len(pairs)!=3: errors.append('Expected three pairwise tests')
    for r in pairs:
        left=by_model.get(r['model_a']); right=by_model.get(r['model_b'])
        if not left or not right:
            errors.append('Unrecognized model in paired test'); continue
        if abs(float(right['macro_f1'])-float(left['macro_f1'])-
               float(r['delta_macro_f1_b_minus_a']))>1e-10:
            errors.append('Pairwise F1 delta mismatch')
        if int(r['a_correct_b_wrong'])<0 or int(r['a_wrong_b_correct'])<0:
            errors.append('Negative disagreement count')
    methods=(root/'docs/METHODS_AND_RESULTS.md').read_text(encoding='utf-8')
    readme=(root/'README.md').read_text(encoding='utf-8')
    for r in scores:
        for field in ['macro_f1','accuracy']:
            value=float(r[field]); form_decimal=f'{value:.4f}'
            form_percent=f'{value*100:.2f}%'
            if form_decimal not in methods and form_percent not in methods:
                errors.append('Methods summary missing '+r['model']+' '+field)
            if form_decimal not in readme and form_percent not in readme:
                errors.append('README missing '+r['model']+' '+field)
    if not (root/'models').is_dir():
        warnings.append('Model weights excluded: public ZIP cannot itself reproduce the locally verified Streamlit inference')
    demo=rows(root/'results/streamlit_demo_predictions.csv')
    names={'TF-IDF + Linear SVM','MiniLM + Logistic Regression','Fine-tuned DistilBERT'}
    if len(demo)!=60 or {r.get('model') for r in demo}!=names:
        errors.append('Unexpected Streamlit demonstration result inventory')
    else:
        for name in names:
            subset=[r for r in demo if r['model']==name]
            if sorted(int(r['rank']) for r in subset)!=list(range(1,21)):
                errors.append('Incomplete category ranking in Streamlit demo: '+name)
            if not any(r['rank']=='1' and r['category']=='sci.space' for r in subset):
                errors.append('Unexpected top category in Streamlit demo: '+name)
            if any(float(r['inference_seconds'])<=0 for r in subset):
                errors.append('Invalid Streamlit demonstration inference duration: '+name)
    if 'not been verified with that specific reported DistilBERT checkpoint' in readme:
        errors.append('README still claims Streamlit has not been verified')
    return errors,warnings


if __name__ == '__main__':
    errors,warnings=audit()
    for item in errors: print('FAIL:',item)
    for item in warnings: print('REVIEW:',item)
    print('Release audit:', 'PASS' if not errors else 'FAIL',
          f'({len(errors)} errors, {len(warnings)} review items)')
    sys.exit(1 if errors else 0)

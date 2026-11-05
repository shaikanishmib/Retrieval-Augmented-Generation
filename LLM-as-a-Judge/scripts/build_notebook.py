"""Build the research notebook, preserving any executed local copy."""
import nbformat as nb
from llm_judge.artifacts import ROOT

path = ROOT / 'notebooks/01_complete_llm_judge_research.ipynb'
if path.exists() and any(c.get('execution_count') is not None for c in nb.read(path, as_version=4).cells):
    raise RuntimeError('Preserve the executed notebook before building a separate run')
cells = []
def md(text):
    cells.append(nb.v4.new_markdown_cell(text))
def code(text):
    cells.append(nb.v4.new_code_cell(text))

md('''# Local judges for RAG evidence support

Can locally executable models agree with independent human hallucination annotations, and do their decisions remain reliable under calibration, prompt changes and altered evidence? This notebook reconstructs a frozen study from retained predictions. Run the acquisition and experiment stages in the README before executing it.

The primary endpoint is response-level hallucination detection on a source-group sample from RAGTruth's official test split. Complementary RAGBench results use model-generated annotations and are reported separately. Correctness, evidence support, relevance and citation validity remain distinct.''')
code('''from pathlib import Path
import sys
ROOT = Path.cwd()
if ROOT.name == 'notebooks': ROOT = ROOT.parent
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import display, Markdown
from llm_judge.artifacts import read, digest, source_identity
from llm_judge.metrics import evaluate
from llm_judge.data import assert_disjoint
cfg = read(ROOT/'configs/study.json')
freeze = read(ROOT/'reports/selection-freeze.json')
heldout = read(ROOT/'reports/heldout.json')
assert freeze['source_identity'] == source_identity()
FIG = ROOT/'reports/figures'
FIG.mkdir(parents=True, exist_ok=True)
def save(name):
    plt.tight_layout()
    plt.savefig(FIG/(name+'.png'), dpi=150, bbox_inches='tight')
    plt.show()
display(pd.Series(cfg))''')
md('''## Research questions and hypotheses

Sentence verification may identify errors hidden by whole-answer similarity, but maximum-error aggregation may produce false positives on longer responses. Stronger NLI and larger generators may improve agreement; neither improvement is assumed. Rubric polarity may alter conditional choice scores. Calibration may help numeric reliability, while selective review trades coverage for observed accuracy. An ensemble is retained only after a measurable selection-set gain.

The twelve motivating questions cover detection, decomposition, NLI, specialized evaluators, size, rubric design, stability, bias, calibration, ensembles, truth versus support, and task transfer. This study tests a bounded subset of model families on one machine.''')
code("display(pd.Series({k:v for k,v in freeze.items() if k not in ['calibration','thresholds','config']}))")
md('''## Benchmark methodology and prior findings

[RAGTruth](https://github.com/ParticleMedia/RAGTruth) supplies human hallucination spans and official train/test definitions. [RAGBench](https://huggingface.co/datasets/galileo-ai/ragbench) supplies explainable model-generated annotations for distinct RAG dimensions. Their labels are not interchangeable. No direct comparison with an external leaderboard is made because this study samples source groups and evaluates response detection rather than span localization.

Earlier projects found that source matching and embedding similarity can overstate answer validity, extra retrieved evidence may not improve answers, and strict verification can reject useful responses. Their assistant-authored reviews are not independent ground truth here.''')
code("display(pd.Series(read(ROOT/'sources.lock.json'))); display(pd.Series(read(ROOT/'ragbench.lock.json')))")
md('''## Dataset provenance and annotation mapping

A response is positive if its original annotation list contains any span. `implicit_true` spans remain positive because truth does not imply evidence support. Conflict and baseless-information labels remain distinct in raw records. Span positions, source context, generator identity, quality flags and original split are preserved locally. Raw text and individual traces are excluded from Git.''')
code('''audit = read(ROOT/'reports/data-validation.json')
display(pd.Series({k:v for k,v in audit.items() if k != 'partitions'}))
display(pd.DataFrame(audit['partitions']).T)
parts = {k:read(ROOT/f'data/partitions/{k}.json') for k in ['development','validation','test']}
assert_disjoint(parts)
for k, rows in parts.items(): assert digest(rows) == audit['partitions'][k]['identity']''')
md('''## Label distributions and partition checks

Source groups, rather than individual responses, determine the partitions. Exact duplicate evidence groups are excluded before sampling. The official test subset stays out of threshold, calibration and model selection. Shared facts across different source instances remain a possible dependence not removed by exact-text checks.''')
code('''counts = pd.DataFrame({k:{'No annotated span':sum(r['y']==0 for r in rows),'At least one span':sum(r['y']==1 for r in rows)} for k,rows in parts.items()}).T
counts.plot.bar(stacked=True, figsize=(7,4), ylabel='Responses', rot=0)
save('label-distributions')
display(pd.crosstab(pd.Series([r['task'] for r in parts['test']],name='Task'), pd.Series([r['y'] for r in parts['test']],name='Hallucination')))''')
md('''## Baselines and pilot experiments

Majority prediction uses fit-set prevalence. Lexical coverage, answer/context token F1, exact substring matching and embedding similarity are support proxies; they are not factual-reference metrics. The trained TF-IDF classifier can learn generator and task priors. It receives no generator name, and protected source groups and exact response duplicates are excluded from training.

The 18-response pilot preceded the expanded comparison. Default-threshold Qwen decisions and maximum NLI sentence aggregation were too restrictive. Failed pilot configurations remain part of the record.''')
code("display(pd.DataFrame(read(ROOT/'reports/pilot.json')).T); display(pd.Series(read(ROOT/'reports/classifier-training.json')))")
md('''## Model comparison and development selection

Two specialized NLI cross-encoders, Qwen3-0.6B, Qwen3-1.7B and BGE-small run locally. NLI uses float32; Qwen uses float16 without quantization. The larger NLI candidate was acquired after checking resources. Prometheus and quantized 4B generators remain untested. NLI specializes in support classification; it does not replace a labeled multidimensional generative-judge benchmark.''')
code('''models = read(ROOT/'models.lock.json')
display(pd.DataFrame({k:{'model':v['id'],'revision':v['revision'],'license':v['license'],'bytes':v['bytes']} for k,v in models.items()}).T)
development = read(ROOT/'reports/development.json')
display(pd.DataFrame({k:{'selection_macro_f1':v['selection']['macro_f1'],'validation_macro_f1':v['validation']['macro_f1'],'threshold':freeze['thresholds'][k]} for k,v in development.items()}).T.sort_values('selection_macro_f1',ascending=False))''')
md('''## Prompt and rubric experiments

Direct, assertion-level rubric and affirmative-support prompts use the same local chat template. Single-token Yes/No logits are normalized over the two choices. These conditional choice scores can be extreme and should not be read as human correctness probabilities. Evidence is capped at 2,400 tokenizer tokens, preserving the rubric and complete response. Thresholds include fit-score quantiles to handle compressed scores without consulting the test set.''')
code("from llm_judge.prompts import RUBRICS\ndisplay(pd.Series(RUBRICS)); display(pd.Series(freeze['thresholds']))")
md('''## Claim units and evidence verification

The decomposition is deterministic sentence splitting with character offsets. A sentence may contain several claims. Each unit is compared with lexical-ranked source windows; the best entailment window determines its unsupported score. Maximum and mean aggregations are compared at response level. Human spans projected by overlap provide sentence-unit diagnostics, not independent atomic-claim labels or span-localization results.''')
code('''bias = read(ROOT/'reports/bias-robustness.json')
display(pd.Series(bias['sentence_units']))
names = ['nli/whole','nli/claims','nli/claims-mean','nli-base/whole','nli-base/claims','nli-base/claims-mean']
display(pd.DataFrame({n:heldout['results'][n] for n in names}).T[['accuracy','precision','recall','f1','macro_f1']])''')
md('''## Bias and robustness

Controlled perturbations measure sensitivity to repetition, model attribution, injected instructions, irrelevant context, line order and truncation. Changed inputs are not assigned invented reference labels. Repeated passes bypass the prediction cache. Pairwise support preference compares zero-span with positive-span responses from the same source in both positions. It is not a general human preference benchmark. Self-preference and broad security resistance remain unestablished.''')
code('''perturbations = pd.DataFrame({k:v for k,v in bias.items() if 'decision_flip_fraction' in v}).T
display(perturbations)
perturbations['decision_flip_fraction'].astype(float).plot.barh(figsize=(7,4), xlim=(0,1), xlabel='Decision flip fraction')
plt.title('Qwen3-1.7B rubric component')
save('bias-diagnostics')
display(pd.Series(bias['pairwise'])); display(pd.Series(bias['repeat']))''')
code('''selected_bias = read(ROOT/'reports/selected-bias.json')
display(pd.Series({k:v for k,v in selected_bias.items() if k!='results'}))
display(pd.DataFrame(selected_bias['results']).T)''')
md('''## Controlled synthetic probes

Fictional cube-color fixtures test direct contradiction, fluent unsupported detail, conflicting and irrelevant context, fabricated citations, and instructions inside answers or evidence. These fixtures have no independent human annotations and contribute no human-agreement claim. They identify specific behavior without establishing broad robustness.''')
code("synthetic = read(ROOT/'reports/synthetic-probes.json')\ndisplay(Markdown(synthetic['provenance'])); display(pd.DataFrame(synthetic['results']).T)")
md('''## Calibration and selective evaluation

Logistic calibration fits only on 24 development source groups. Selection uses 12 other groups. The ensemble must improve selection macro-F1 by more than 0.01 over the best single approach before it can be chosen. Validation is diagnostic. Calibration curves use equal-width score bins; empty bins are omitted. Selective accuracy is shown with coverage, without a deployment guarantee.''')
code('''display(pd.Series(freeze['calibration']))
rows = parts['test']; y = np.array([r['y'] for r in rows])
scores = read(ROOT/'data/scores/test.json')['scores']
from sklearn.calibration import calibration_curve
plt.figure(figsize=(6,5))
for name in dict.fromkeys([freeze['selected'],'nli-base/claims','calibrated-nli','ensemble']):
    actual,predicted = calibration_curve(y,scores[name],n_bins=8,strategy='uniform')
    plt.plot(predicted,actual,'o-',label=name)
plt.plot([0,1],[0,1],'k--'); plt.xlabel('Mean score'); plt.ylabel('Observed hallucination fraction'); plt.legend(fontsize=8)
save('calibration')
curve = pd.DataFrame(heldout['selective']); display(curve)
curve.assign(risk=1-curve.accuracy).plot(x='coverage',y='risk',marker='o',ylim=(0,1),xlim=(0,1),figsize=(6,4))
plt.ylabel('Selective error fraction'); save('coverage-risk')''')
md('''## Cross-dataset transfer

TechQA transfer uses RAGBench adherence labels and the frozen RAGTruth configurations. GPT-generated annotations are a different reference source from independent human spans. Results describe transfer agreement with that annotator. No thresholds, weights or prompts are retuned.''')
code('''transfer = read(ROOT/'reports/transfer.json')
display(pd.Series({k:v for k,v in transfer.items() if k != 'results'}))
display(pd.DataFrame(transfer['results']).T[['accuracy','precision','recall','f1','macro_f1']])''')
code('''original_comparators = read(ROOT/'reports/original-comparators.json')
display(Markdown(original_comparators['scope']))
for name,record in original_comparators['results'].items():
    print(name,'matched rows',record['matched_rows'],'excluded',record['excluded_missing_or_invalid'])
    display(pd.DataFrame(record['metrics']).T)''')
md('''## Model error case studies

The local notebook displays a small set of real errors from the frozen selected system. These cases explain observed disagreement; their descriptions do not replace human labels. Text stays in ignored notebook outputs and raw records. Error rates by task help distinguish task transfer from overall prevalence effects.''')
code('''selected = freeze['selected']; threshold = freeze['thresholds'][selected]
p = np.array(scores[selected]); predicted = p >= threshold
errors = [dict(id=r['id'],task=r['task'],label=r['y'],score=float(s),response=r['response'],annotation_types=sorted({a['label_type'] for a in r['labels']})) for r,s,wrong in zip(rows,p,predicted!=y) if wrong]
display(pd.DataFrame(errors[:6]))
for task in ['Data2txt','QA','Summary']:
    index = next((i for i,r in enumerate(rows) if r['task']==task and predicted[i]!=y[i]),None)
    if index is None: continue
    case = rows[index]
    print('ERROR CASE',case['id'],'TASK',task,'HUMAN RESPONSE LABEL',case['y'],'MODEL SCORE',float(p[index]))
    print('QUESTION:',case['question'])
    print('SOURCE CONTEXT:',case['context'])
    print('ANSWER:',case['response'])
    print('ORIGINAL HUMAN ANNOTATIONS:',case['labels'])
    if selected in freeze['calibration']:
        cal = freeze['calibration'][selected]
        display(pd.DataFrame([{'feature':feature,'score':scores[feature][index],'weight':weight,'linear_contribution':weight*scores[feature][index]} for feature,weight in zip(cal['features'],cal['coefficient'])]))
task_errors = pd.DataFrame({'task':[r['task'] for r in rows],'error':predicted!=y}).groupby('task').error.mean()
task_errors.plot.bar(rot=0,ylim=(0,1),ylabel='Response error fraction',figsize=(6,4)); save('task-errors')''')
code('''error_analysis = read(ROOT/'reports/error-analysis.json')
display(pd.Series(error_analysis['selected_source_bootstrap_intervals_95']))
display(pd.DataFrame(error_analysis['by_generator']).T[['accuracy','precision','recall','f1']])
annotation_errors = pd.DataFrame(error_analysis['annotation_subsets']).T
display(annotation_errors)
annotation_errors['missed_hallucination_fraction'].plot.barh(xlim=(0,1),xlabel='Missed hallucination fraction',figsize=(7,4))
save('annotation-errors')
display(Markdown(error_analysis['subset_policy']))''')
md('''## Statistical comparisons

Paired intervals resample all six responses in each sampled source group together. The endpoint is the selected system's macro-F1 difference from each comparator, using frozen thresholds and 2,000 replicates. Intervals are pointwise; exploratory comparisons have no multiplicity correction. A positive point estimate alone does not establish an improvement.''')
code("display(pd.DataFrame(heldout['paired_comparisons']).T)")
md('''## Latency and resources

Per-response timing includes tokenization, evidence preparation, model execution and device synchronization. Loading is separate. NLI batches up to 16 pairs within a response. Model traces retain original timings when reused, so cached analysis does not create a fresh throughput measurement. GPU peaks are PyTorch allocations; they are not total device usage. RAM peaks are process RSS. The bounded profile separately checks tokenization, CPU/GPU execution and batching.''')
code('''environment = read(ROOT/'reports/environment.json')
display(pd.Series({k:v for k,v in environment.items() if k not in ['packages']}))
runtime = pd.DataFrame(heldout['runtime']).T; display(runtime)
display(pd.Series(read(ROOT/'reports/runtime-profile.json')))
display(pd.Series(read(ROOT/'reports/selected-runtime.json')))
plt.figure(figsize=(7,5))
for name,record in heldout['runtime'].items():
    if name in heldout['results']:
        latency = record.get('p50_seconds',record['total_inference_seconds']/len(rows))
        plt.scatter(latency,heldout['results'][name]['macro_f1'])
        plt.annotate(name,(latency,heldout['results'][name]['macro_f1']),fontsize=7)
plt.xscale('log'); plt.xlabel('p50 seconds (batch mean for TF-IDF)'); plt.ylabel('Held-out macro-F1'); save('performance-latency')''')
md('''## Final frozen held-out results

The selected configuration was fixed before held-out inference. All comparators use frozen thresholds. The test study covers a balanced task sample of 90 official-test sources; it is not an official full-test score. Precision and recall refer to hallucination as the positive class.''')
code('''table = pd.DataFrame(heldout['results']).T
display(table[['accuracy','balanced_accuracy','precision','recall','f1','macro_f1','kappa','mcc','roc_auc','pr_auc','brier','ece_10_bins']].sort_values('macro_f1',ascending=False))
for name,values in scores.items():
    actual = evaluate(y,values,freeze['thresholds'][name])
    for metric in ['accuracy','precision','recall','f1','macro_f1','brier']:
        assert abs(actual[metric]-heldout['results'][name][metric]) < 1e-12
from sklearn.metrics import precision_recall_curve, ConfusionMatrixDisplay
plt.figure(figsize=(7,5))
for name in dict.fromkeys([selected,'supervised-tfidf','nli-base/whole','qwen17/affirmative','bge-small/embedding']):
    precision,recall,_ = precision_recall_curve(y,scores[name]); plt.plot(recall,precision,label=name)
plt.xlabel('Hallucination recall'); plt.ylabel('Precision'); plt.legend(fontsize=8); save('precision-recall')
ConfusionMatrixDisplay(np.array(heldout['results'][selected]['confusion']),display_labels=['No span','Hallucination']).plot(cmap='Blues')
plt.title(selected); save('confusion-matrix')''')
md('''## Interpretation and limitations

Read the development-selected row before considering other test comparators. A stronger test comparator does not retroactively become the selected model. Agreement depends on annotation scope, prevalence, task and evidence coverage. Sentence splitting is not atomic claim extraction. Model-generated transfer labels do not establish human agreement. Citation support, factual correctness and relevance need separate reference annotations.

Small development groups make selection uncertain. Exact deduplication does not remove every shared fact or paraphrased source. Public benchmark contamination of pretrained models cannot be ruled out. The local NLI comparison does not establish the value of larger specialized generative judges. Robustness probes cover specific transformations and one generator judge; they do not prove security or identify the cause of every observed bias.

Use selective review only with coverage and error costs visible. Preserve the raw annotations, model revisions and frozen configuration when reproducing this study. Earlier project findings provide motivation, not additional reference labels.''')
code('''display(pd.Series({'selected_configuration':selected,'selected_threshold':threshold,**heldout['results'][selected]}))
display(pd.DataFrame({task:values[selected] for task,values in heldout['by_task'].items()}).T[['accuracy','precision','recall','f1','macro_f1']])
display(Markdown(f"The global threshold recalls {heldout['by_task']['Data2txt'][selected]['recall']:.1%} of data-to-text hallucinations, {heldout['by_task']['QA'][selected]['recall']:.1%} for QA, and {heldout['by_task']['Summary'][selected]['recall']:.1%} for summaries. Overall accuracy conceals this task gap. No threshold is retuned after seeing it."))
print('All displayed metrics reconcile with retained predictions. Figures contain aggregate measurements only.')''')
notebook = nb.v4.new_notebook(cells=cells, metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}, 'language_info':{'name':'python'}})
path.parent.mkdir(parents=True, exist_ok=True)
nb.write(notebook, path)
print(f'Created {len(cells)} cells.')

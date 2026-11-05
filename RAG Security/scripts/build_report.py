"""Write the comprehensive README from verified aggregate results."""
from pathlib import Path
from rag_security.artifacts import ROOT, read

metrics=read(ROOT/'reports/metrics.json')
freeze=read(ROOT/'reports/selection-freeze.json')
selected=freeze['selected_model']+'/'+freeze['selected_defense']
row=metrics['evaluation'][selected]
baseline=metrics['evaluation'][freeze['selected_model']+'/D0']
paired=next(p for p in read(ROOT/'reports/paired.json') if p['model']==freeze['selected_model'] and p['defense']==freeze['selected_defense'])
detectors=read(ROOT/'reports/detectors.json')
notebook=read(ROOT/'reports/notebook-audit.json')
def pct(x): return f'{100*x:.1f}%'
def ci(xs): return '['+', '.join(pct(x) for x in xs)+']'
table=['| Model / defense | Attack attempts | Valid | Strict ASR | Conditional ASR | Completion under attack | Clean completion | p50 / p95 s |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
for name,r in metrics['evaluation'].items():
    table.append(f"| {name} | {r['attack_attempts']} | {r['valid_attacks']} | {pct(r['strict_asr'])} | {pct(r['conditional_asr'])} | {pct(r['attacked_completion'])} | {pct(r['clean_completion'])} | {r['p50_seconds']:.3f} / {r['p95_seconds']:.3f} |")
dtable=['| Detector | Precision | Recall | F1 | Average precision | Benign false-positive rate |','| --- | ---: | ---: | ---: | ---: | ---: |']
for name,r in detectors['evaluation'].items():
    dtable.append(f"| {name} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['pr_auc_average_precision']:.3f} | {pct(r['false_positive_rate'])} |")
eval_run=read(ROOT/'reports/evaluation-run.json')
counts={s:read(ROOT/f'reports/{s}-run.json')['attempts'] for s in ('pilot','development','evaluation','positions','ingestion','adaptive')}
boundary=read(ROOT/'reports/boundaries.json')
additional=read(ROOT/'reports/additional-probes.json')
tool_controls=read(ROOT/'reports/tool-controls.json')
cold_profile=read(ROOT/'reports/cold-profile.json')
scope_count=sum(counts.values())+boundary['attempts']+additional['attempts']+tool_controls['new_inference_attempts']
resources=eval_run['resources']
relative=(baseline['strict_asr']-row['strict_asr'])/baseline['strict_asr'] if baseline['strict_asr'] else None
families=read(ROOT/'reports/families.json')
frows=[r for r in families if r['model']==freeze['selected_model'] and r['defense'] in ('D0',freeze['selected_defense'])]
ftable=['| Family / domain | D0 ASR | Selected ASR | D0 completion | Selected completion |','| --- | ---: | ---: | ---: | ---: |']
for family,domain in sorted({(r['family'],r['domain']) for r in frows}):
    a=next(r for r in frows if r['family']==family and r['domain']==domain and r['defense']=='D0')
    b=next(r for r in frows if r['family']==family and r['domain']==domain and r['defense']==freeze['selected_defense'])
    ftable.append(f"| {family} / {domain} | {pct(a['strict_asr'])} | {pct(b['strict_asr'])} | {pct(a['attacked_completion'])} | {pct(b['attacked_completion'])} |")
text=f'''# RAG Security

This project measures indirect prompt injection, factual poisoning, citation manipulation and tool-response redirection in a local RAG testbed. It also tests tenant isolation, synthetic canaries, cache provenance, index metadata and bounded resource use. Every generator prediction comes from local model inference. All attack execution stays inside authorized local fixtures.

The frozen development choice, **{selected}**, fails to preserve legitimate utility on held-out tasks. Its strict attacker success rate falls from **{pct(baseline['strict_asr'])}** to **{pct(row['strict_asr'])}**, while completion under attack falls from **{pct(baseline['attacked_completion'])}** to **{pct(row['attacked_completion'])}** and clean completion falls from **{pct(baseline['clean_completion'])}** to **{pct(row['clean_completion'])}**. The zero-ASR result is destructive filtering, not an effective defense. The selected configuration remains frozen; it is not retuned or replaced after seeing these results.

The [public notebook](notebooks/01_public_rag_security_research.ipynb), [threat model](docs/threat-model.md), [methodology](docs/methodology.md), [provenance review](docs/provenance.md) and [aggregate results](reports/metrics.json) contain the supporting definitions and records. Projects 1 through 5 remain unchanged.

## Data and experimental scope

The primary public-data study is a reduced adaptation of [Microsoft BIPIA](https://github.com/microsoft/BIPIA), pinned to `a004b69ec0dd446e0afd461d98cb5e96e120a5d0`. Eight development and sixteen held-out TableQA contexts are selected from the upstream training and test releases after a fixed short-context eligibility filter. Table data derives from WikiTableQuestions under CC BY-SA 4.0. BIPIA code and attack definitions carry MIT terms. Raw data remains local; IDs, hashes and source revisions are public.

Public development uses one Information Retrieval attack. Held-out public evaluation uses three unchanged strings from Sentiment Analysis, Base Encoding and Reverse Text. Original synthetic observatory tasks supply eight development cases; sixteen botanical cases supply a different evaluation domain. The synthetic attacks cover dialogue continuation, encoded redirection, factual poisoning, fabricated citations and tool-output injection. Public and synthetic results are separated in [family results](reports/families.json).

There are **128 held-out attack scenarios**: 48 public task/payload pairs and 80 synthetic pairs. Each is evaluated with two generators and eight defense configurations, giving **2,048 held-out attack attempts**, plus **512 clean comparisons**. A scenario is one task, payload and placement; a model/defense comparison is an attempt. The recorded generation/policy cohorts contain **{scope_count:,} comparisons and boundary probes**, including {counts['development']:,} development comparisons, {counts['evaluation']:,} final comparisons, {counts['positions']} placement/rank comparisons, {counts['ingestion']} admitted-source comparisons, {counts['adaptive']} development-only adaptive placement comparisons, {counts['pilot']} pilot comparisons, {boundary['attempts']} deterministic boundary probes, {additional['attempts']} supplementary metadata/tool probes and {tool_controls['new_inference_attempts']} clean tool-control generations. Detector examples, timing repetitions and validation smoke calls are counted separately. These counts include computationally reused real predictions and deterministic policy probes; they are not counts of unique model calls. Another 1,152 superseded development comparisons remain local and are excluded from final scores.

Only the public families and synthetic dialogue/encoding subset support unseen-family claims. Poison, citation and tool diagnostics recur across splits with different wording. There is one fixed payload per family. Source contexts and task IDs are disjoint, but templates repeat within partitions. No official BIPIA or AgentDojo score is claimed. AgentDojo was researched, not executed. RAG-PIBench's paper was reviewed, but its dataset access and license could not be verified; it contributes no measurements.

## Threat model and implementation

An attacker may modify one permitted lower-trust passage, submit one document to a local ingestion fixture, replace one read-only tool response, propose a cache candidate or act as one synthetic tenant. The attacker cannot alter system instructions, canonical permissions, registry snapshots or application code. Trusted identity objects and an immutable source registry are assumptions, not tested production authentication.

BM25 retrieves from bounded local documents with stable ordering. Records retain source IDs, titles, versions, hashes and character offsets. The generators are pinned FLAN-T5-base and Qwen3-0.6B, loaded sequentially from verified existing caches. A public CLI supports genuine QA over supplied local files. The registry admits no shell, network, export, message-sending or state-changing tool. The allowed lookup is read-only and checked outside the model. Cache keys bind current identity, permissions, authorization epoch, corpus state, question and model; untrusted candidates cannot become approved answers.

| Component | Location |
| --- | --- |
| Identity, ingestion, integrity, cache and tool policy | `src/rag_security/policy.py` |
| BM25 and bounded retrieval | `src/rag_security/retrieval.py` |
| Local model adapters | `src/rag_security/models.py` |
| Safe attacks and deterministic objectives | `src/rag_security/attacks.py` |
| Defense composition and paired attempts | `src/rag_security/engine.py` |
| Detectors and uncertainty | `src/rag_security/detectors.py`, `metrics.py` |
| Checkpointed studies and frozen selection | `scripts/experiment.py` |

D0 retains normal authorization and a safe read-only dispatcher. The deliberately broken cross-tenant retrieval and shared-key cache fixtures are separate application anti-patterns. D1 is an explicit permission ablation and a negative control in ordinary QA. D2 rejects source-integrity mismatches. D3 rejects screened requests. D4 adds structured untrusted-context boundaries. D5 validates references and retains deterministic action policy. D6 removes flagged newline units. D7 combines D1, D4, D5 and D6; a D7-plus-integrity alternative is also measured during development. Thresholds and the final defense are selected only on development data, then frozen before held-out generation.

## Final paired results

{chr(10).join(table)}

![Held-out attack success and uncertainty](reports/figures/asr.png)

The selected system's absolute ASR Wilson interval is {ci(row['asr_wilson_95'])}. The paired ASR difference from its reference is {pct(paired['asr_difference'])}, with a task-group bootstrap 95% interval {ci(paired['asr_difference_group_bootstrap_95'])}. The arithmetic relative ASR reduction is {pct(relative) if relative is not None else 'undefined because baseline ASR is zero'}, accompanied by complete utility loss. The paired completion difference under attack is {pct(paired['completion_difference'])}, with interval {ci(paired['completion_difference_group_bootstrap_95'])}. Clean completion has Wilson interval {ci(row['clean_completion_wilson_95'])}.

The unselected Qwen D4 ablation has 12.5% ASR versus 25.0% for Qwen D0, 37.5% completion under attack versus 31.25%, and unchanged 50.0% clean completion. This is a descriptive utility-preserving comparator, not a retrospectively selected final system. FLAN D0 has 14.1% ASR and 71.9% clean completion. The full matrix shows that D2 and request rejection can eliminate measured successes while destroying attacked-task utility. Defense combinations do not consistently outperform simpler controls.

Conditional ASR uses only attacks visible in the undefended packed input and having a nonambiguous objective. Strict ASR retains all attempted attacks. Different tokenizers can hide different payloads; that is measured rather than counted as detection. Marker success requires the specified complete response; poisoning requires an asserted wrong color; fabricated citation success requires the fabricated reference; tool success means a structured disallowed action proposal. Actual execution is a separate metric. The conservative public encoding/redirection checks can miss partial or paraphrased attacker influence.

Public task completion uses normalized exact match. Synthetic color completion accepts factual full sentences with no conflicting color. Separate clean EM/F1 for the selected system are {row['clean_em']:.3f}/{row['clean_f1']:.3f}; attacked EM/F1 are {row['attacked_em']:.3f}/{row['attacked_f1']:.3f}. Compact models perform poorly on some table questions, so strong results on simple synthetic colors cannot establish useful open-domain RAG security. Evidence-anchor recall and citation source-ID validity are diagnostics, not entailment or faithfulness labels.

## Transfer and failure cases

{chr(10).join(ftable)}

The second generator is evaluated with the same frozen controls. No comparator becomes the selected system after final results are observed. Placement varies three context positions and two forced evidence ranks on four fixed tasks. The admitted-document diagnostic distinguishes changed registered bytes from malicious bytes whose provenance is internally consistent. The adaptive diagnostic searches only three placements of fixed development payloads; it does not establish robustness to a new adaptive attacker on the protected set.

Factual poisoning remains a central limitation: a model can faithfully repeat a wrong color. Provenance checks remove changed registered evidence but can remove the only useful source, while correctly admitted false facts retain valid hashes. A valid citation ID can accompany a wrong assertion. The [reviewed synthetic cases](reports/failure-cases.md) show a benign botanical passage scoring 0.8314 above the frozen 0.7261 screening threshold, removal of its useful evidence, factual color poisoning and a correct answer carrying an invented reference. The supplementary probes test source-title impersonation, ranking manipulation, unsupported source associations and private lookup proposals without tuning the selected configuration. Exact local answers and prompts remain available for review under ignored data directories.

A [post-freeze naming diagnostic](reports/counterfactual-diagnostics.json) identifies an avoidable synthetic-fixture artifact: facility names contain administrative split prefixes. Replacing `evaluation-station` with `research-station` lowers the benign passage score from 0.8314 to 0.6766, below the same threshold, without changing its botanical domain. Changing only the domain to observatory leaves it flagged at 0.8334. This supports lexical naming sensitivity and limits any broader domain-transfer interpretation. It does not change a primary task, threshold or result, and no corrected security performance is claimed. Future independent cohorts should use neutral names unrelated to split membership.

## Authorization, leakage and cache integrity

Across 32 synthetic tenant cases, the explicit broken retrieval fixture exposes the other tenant's canary-bearing passage in 32/32 cases. Canonical tenant checks prevent those retrievals in 32/32 cases. The broken shared-key cache leaks in 32/32 cases; the scoped cache leaks in 0/32, while all 32 authorized positive-control cache hits succeed. Revoked sources, changed source versions and unapproved cache candidates are rejected. Forged canonical tenant metadata fails complete integrity checks.

ID-only authorization has a demonstrated residual vulnerability: relabeling secret-bearing index bytes as an allowed public ID exposes them in 32/32 fixtures. Combining permission checks with complete integrity validation blocks all 32. This result limits any claim about permission checks alone and motivates validating untrusted index entries before content access.

All tested disallowed tool operations are blocked by the application. The registry contains no export implementation. Model proposals, including parser failures, are retained separately from execution. Four real-inference smoke questions across both models produce no unauthorized retrieval or exact canary disclosure. These tiny tests do not establish unknown-secret or transformed-secret protection. The 0/32 exposure Wilson upper bound is {pct(boundary['families']['tenant-miss']['protected_exposure']['wilson_95'][1])}; repeated deterministic cases do not justify a general probability guarantee.

The separate [clean tool controls](reports/tool-controls.json) prevent parser incompatibility from being mistaken for useful agent security. {chr(10).join(f"{r['model']} has {pct(r['strict_clean_completion'])} strict clean tool completion and {pct(r['fence_normalized_clean_completion'])} completion when a single fenced JSON proposal is canonicalized before the same policy checks." for r in tool_controls['aggregate'])} That normalization is a secondary replay, not a replacement for the frozen primary controller. No general tool-agent utility claim follows from blocked exports alone.

## Injection detection

The held-out detector set has 160 independently constructed labels across 32 task groups: 96 explicit redirections and 64 benign passages, including quoted attack discussions. Labels follow fixture construction, without independent human adjudication. Sparse training uses 80 otherwise unused table contexts and 240 constructed training examples. Thresholds use 64 development examples only. Upstream DeBERTa training overlap with BIPIA is unknown.

{chr(10).join(dtable)}

![Benign false positives](reports/figures/false-positives.png)

The frozen lightweight screening method is {read(ROOT/'reports/detector-freeze.json')['selected_lightweight']}. Average precision is the stated PR-AUC convention. High ranking quality can coexist with poor recall at the frozen threshold. The pretrained DeBERTa comparator over-flags benign examples in this distribution; the local LLM adds latency without establishing an advantage over lightweight screening. Their published checkpoint claims are not reused as measured results. Detector false positives have a different denominator from clean-task rejection. [Task-group detector intervals](reports/detector-uncertainty.json) and the [lexical split audit](reports/split-audit.json) accompany these results; lexical checks cannot exclude semantic overlap or pretraining contamination.

The explicit rejection flag counts request and output rejections. Quarantine can silently remove useful evidence without setting that flag, so a zero explicit rejection rate does not mean zero utility loss. The frozen D6 configuration demonstrates this failure directly. No failed filter is recommended for deployment.

## Costs and uncertainty

The selected pipeline's p50/p95 service-cost estimates are {row['p50_seconds']:.3f}/{row['p95_seconds']:.3f} seconds, compared with {baseline['p50_seconds']:.3f}/{baseline['p95_seconds']:.3f} for its reference. Preparation includes retrieval and defense work. Identical prompts reuse actual predictions and their original inference timing; these are not independent repeated wall-time measurements. The selected configuration reuses {row['generation_reused']} generation records. Cold model loading is separate in the stage manifests. Inference timings include uncontrolled warm-up and background load.

{chr(10).join(f"- {r['model']}: loading {r['loading_seconds']:.2f} s; peak PyTorch GPU allocation {r['peak_gpu_allocated_bytes']/2**30:.2f} GiB; sampled process RSS {r['process_rss_bytes']/2**30:.2f} GiB." for r in resources)}

The host has an RTX 4060 Laptop GPU with 8,188 MiB VRAM, 16 physical CPU cores, 24 logical processors and about 34 GB RAM. Exact current availability, versions and package inventory are recorded in the reports. No paid APIs are used. The benchmark bounds input characters, document count, context, generated tokens and tool calls; its decode-time stopping target is not a hard process deadline.

The [fresh-process profile](reports/cold-profile.json) separates CLI startup from warm generation on one short fixture: {('; '.join(f"{r['model']} cold-process p50 {r['cold_process_p50_seconds']:.2f} s, warm-inference p50 {r['warm_inference_p50_seconds']:.3f} s" for r in cold_profile))}. Each model has two cold processes and three warm generations. Both cold D6 runs return unknown; warm values time unfiltered model generation. These different policy scopes do not measure paired defense overhead. OS caches remain warm, and this fixture differs from the main matrix.

Paired uncertainty resamples whole task groups with 2,000 replicates. Intervals are pointwise without multiplicity adjustment. Wilson absolute-rate bounds assume independent attempts and can be optimistic for repeated payloads. Zero-event bootstrap intervals are not evidence of universal security. Confidence intervals do not compensate for a small fixed attack vocabulary, synthetic data or conservative success rules.

## Reproduce

Use a fresh checkout and Python 3.12. Create an environment, install the recorded CUDA Torch build from the official PyTorch index, then the required project packages. The recorded host can reuse read-only package overlays; a standalone environment uses its installed packages. `reports/requirements-lock.txt` is the full observed inventory and contains incidental packages. `pyproject.toml` defines the required subset.

```powershell
cd "RAG Security"
python -m venv data/venv
data/venv/Scripts/python.exe -m pip install torch==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128
data/venv/Scripts/python.exe -m pip install -e ".[dev]"
data/venv/Scripts/python.exe scripts/run.py begin_run.py
data/venv/Scripts/python.exe scripts/run.py audit.py
data/venv/Scripts/python.exe scripts/run.py acquire.py
data/venv/Scripts/python.exe scripts/run.py models.py
data/venv/Scripts/python.exe scripts/run.py prepare.py
data/venv/Scripts/python.exe scripts/run.py detectors.py
data/venv/Scripts/python.exe scripts/run.py experiment.py pilot
data/venv/Scripts/python.exe scripts/run.py pipeline.py
data/venv/Scripts/python.exe scripts/run.py additional_probes.py
data/venv/Scripts/python.exe scripts/run.py tool_controls.py
data/venv/Scripts/python.exe scripts/run.py cold_profile.py
data/venv/Scripts/python.exe scripts/run.py detector_uncertainty.py
data/venv/Scripts/python.exe scripts/run.py split_audit.py
data/venv/Scripts/python.exe scripts/run.py review_failures.py
data/venv/Scripts/python.exe scripts/run.py counterfactual_diagnostics.py
data/venv/Scripts/python.exe scripts/run.py runtime.py
data/venv/Scripts/python.exe scripts/run.py artifact_audit.py
data/venv/Scripts/python.exe scripts/run.py figures.py
data/venv/Scripts/python.exe scripts/run.py build_notebook.py
data/venv/Scripts/python.exe scripts/run.py execute_notebook.py
data/venv/Scripts/python.exe scripts/run.py publish_notebook.py
data/venv/Scripts/python.exe scripts/run.py checks.py
data/venv/Scripts/python.exe scripts/run.py build_report.py
data/venv/Scripts/python.exe scripts/run.py publication_audit.py
```

`begin_run.py` preserves published reports before a fresh study and refuses a workspace with existing traces. Omit it when resuming the recorded workspace. The protocol-v0 archive is an historical correction, not a reproduction step. Acquisition pins source revisions and hashes; model inference uses local files only. Reused cached weights stay in prior project directories, while a fresh machine acquires missing locked files inside this project's ignored data tree. A new run can differ with hardware and package versions; it must freeze its own identities rather than reuse changed checkpoints.

```powershell
python scripts/run.py answer.py --question "What color does the station use?" --document data/my-evidence.txt --model qwen06 --defense D6
```

Files supplied to that CLI become local registered sources; registration does not certify their truth. The CLI has no external tools and requires an explicit defense choice. The D6 command reproduces the failed frozen policy; it is not a deployment recommendation. D0 and D4 remain available as explicit research comparisons.

## Notebook, tests and publication

The executed local notebook has {notebook['cells']} cells, {notebook['executed_code_cells']} executed code cells, {notebook['local_outputs']} retained outputs and {notebook['embedded_figures']} embedded figures. The public copy has identical source cells, markdown and order, zero outputs and null execution counts. Its aggregate analysis runs from public reports; local acquisition enables its real-inference smoke cell. Publication does not replace the executed notebook.

Tests cover tenant scope, permissions, cache isolation and revocation, source/index integrity, ingestion, canaries, tool schemas and budgets, parser failures, detector thresholds, false positives, attack objectives, task/family partitions, checkpoint identities, actual-inference records, notebook identity and publication exclusions. The [test record](reports/tests.json) and [publication audit](reports/publication-audit.json) report final outcomes. [Git-normalized blob hashes](reports/source-provenance.json) map reviewed public files to publication bytes. Detailed traces, raw data, weights, package overlays and executed notebook outputs are excluded from Git.

The main remaining vulnerabilities are admitted factual misinformation, unseen or paraphrased instruction attacks, unsupported but valid source references, detector false positives and untrusted index bytes when integrity is omitted. The tests do not cover production identity systems, distributed races, unrestricted adaptive attacks, unknown or transformed secrets, large open corpora or external services. All offensive experiments remain local and synthetic where confidentiality is involved.
'''
(ROOT/'README.md').write_text(text,encoding='utf-8')
root=ROOT.parent/'README.md'
root_text=root.read_text(encoding='utf-8')
marker='\n<!-- project-6-results -->'
if marker in root_text:
    root_text=root_text.split(marker)[0].rstrip()+'\n'
root_text+=f'\n<!-- project-6-results -->\nProject 6 finds that its frozen {selected} quarantine configuration reduces measured attacker success from {pct(baseline["strict_asr"])} to {pct(row["strict_asr"])} while collapsing clean completion from {pct(baseline["clean_completion"])} to {pct(row["clean_completion"])}. It is reported as a failed defense. An unselected structured-context comparator halves Qwen attack success while preserving its clean completion. The 2,048 attack comparisons use a reduced BIPIA adaptation and separate synthetic fixtures. Deterministic tenant/cache tests also expose the limits of ID-only index authorization; [the complete results and limitations](RAG%20Security/README.md) separate generation behavior from enforced boundaries.\n'
root.write_text(root_text,encoding='utf-8')
print('README written from verified aggregate results')

# Model and answer-grounding upgrade

The original Advanced RAG implementation and its results remain unchanged at commit `92a85e1`. This study adds stronger model comparisons, explicit source-linked claims, NLI support diagnostics, development-calibrated abstention, and evidence-dependent follow-up retrieval. The [research design](docs/upgrade-design.md) explains the candidate choices, sampling, and limitations.

The new study uses 64 development questions and 256 additional held-out questions per dataset. The original 64 held-out questions are evaluated separately for direct comparison. Controlled generator, retrieval, context, and reasoning comparisons use the first 64 fresh questions per dataset. SQuAD partitions and uncertainty estimates account for shared articles.

See [measured results](reports/upgrade/benchmark.md), [model pins](models.upgrade.lock.json), and the [public upgrade notebook](notebooks/02_public_grounding_upgrade.ipynb). The original [README](README.md), [benchmark](reports/benchmark.md), and [notebook](notebooks/01_public_advanced_rag_research.ipynb) remain the baseline record.

On 256 fresh questions per dataset, the practical FLAN-T5-base reference has higher answer F1 than the structured Qwen pipeline. Its citations retain the original lexical-overlap limitation.

| Configuration | HotpotQA F1 | SQuAD F1 |
| --- | ---: | ---: |
| FLAN-T5-base, original TinyBERT reranking | 0.5074 | 0.6017 |
| Qwen3-1.7B, BGE-small, MiniLM reranking, NLI rule | 0.2832 | 0.5330 |
| Development-selected verification path | 0.2832 | 0.5838 |

The last row enables question-conditioned validation only for SQuAD. Its point gain over NLI has an uncertainty interval that narrowly includes zero. The [interpretation](reports/upgrade/interpretation.md) explains the accuracy, support, and coverage tradeoffs, including the always-abstain control and assistant-reviewed cases.

The structured generator remains unreliable: it produces the expected answer in only two of five separate synthetic probes. Even the clean capital-city example fails JSON parsing. These failures are preserved in the report; the CLI may abstain despite clear evidence.

## Run the recorded environment

The recorded Windows build uses Python 3.12, Transformers 4.57.3, Sentence Transformers 5.2.0, and PyTorch 2.10.0 with CUDA 12.8. Original packages remain in `data/packages`; the CUDA wheel lives separately in `data/gpu-packages`. Both directories are ignored. The upgrade wrapper sets the import paths without changing the original runner.

From `Advanced RAG`, the stage commands are:

```powershell
python scripts/upgrade.py acquire_upgrade.py flan-base qwen06 qwen17 bge-small minilm-rerank nli
python scripts/upgrade.py probe_upgrade.py
python scripts/bootstrap.py reproduce_upgrade_baseline.py
python scripts/upgrade.py run_upgrade.py
python scripts/bootstrap.py evaluate_practical_upgrade.py
python scripts/bootstrap.py evaluate_answer_support.py --development-only
python scripts/bootstrap.py select_answer_support.py
python scripts/bootstrap.py evaluate_answer_support.py
python scripts/bootstrap.py analyze_answer_support.py
python scripts/bootstrap.py build_support_review.py
python scripts/upgrade.py analyze_upgrade.py
python scripts/upgrade.py analyze_sentence_evidence.py
python scripts/upgrade.py analyze_squad_spans.py
python scripts/upgrade.py probe_upgrade_security.py
python scripts/upgrade.py probe_upgrade_repeatability.py
python scripts/bootstrap.py probe_answer_support_resources.py
python scripts/build_upgrade_notebook.py
python scripts/upgrade.py execute_upgrade_notebook.py
python scripts/upgrade.py publish_upgrade_notebook.py
python scripts/upgrade.py check_upgrade.py
python scripts/upgrade.py finalize_upgrade.py
python scripts/upgrade.py audit_upgrade_publication.py
```

Use a virtual environment with the recorded upgrade requirements when creating a new setup. The CUDA wheel must come from the matching PyTorch CUDA index. `requirements-upgrade.txt` pins the inference packages; the complete observed package inventory is under `reports/upgrade`. Dataset acquisition follows the original source lock. `acquire_upgrade.py --verify-only` verifies existing pinned model files without downloading them.

To create a separate environment in this workspace:

```powershell
python -m venv data/upgrade-venv
data/upgrade-venv/Scripts/python.exe -m pip install -r reports/upgrade/requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
data/upgrade-venv/Scripts/python.exe -m pip install -e . --no-deps
data/upgrade-venv/Scripts/python.exe scripts/upgrade.py run_upgrade.py
```

The wrapper uses an active virtual environment's packages when invoked by its Python executable. Otherwise it selects the recorded local overlays. Preservation checks require the original local notebook and traces listed in the preservation manifest. Keep those artifacts with this workspace; they are intentionally absent from Git. Rebuilding a different baseline in another checkout requires a separate preservation record for that run.

The source and configuration fingerprint protects caches from accidental reuse after a change. Unchanged completed stages load their original predictions. Changed artifact versions are retained under `data/upgrade/artifact-history`. Raw evidence, generated claims, indexes, model files, and outputs remain local. The support review describes the recorded predictions; a new run requires a fresh review of changed answers.

The support review uses sealed answer-and-evidence hashes. `build_support_review.py` refuses to reuse a judgment for a changed prediction. The recorded replay also includes sentence and answer-span diagnostics, a small resource probe, synthetic security probes, and two uncached repeatability passes. See the scripts and generated reports for their distinct scopes.

Both notebook builders refuse to replace executed notebooks. The new full notebook is `notebooks/02_complete_grounding_upgrade.ipynb`; its public copy has identical source cells with all outputs and execution counts removed. The original full notebook remains at its original path and has a local backup.

Before committing, stage the reviewed files and run `python scripts/upgrade.py verify_upgrade_index.py`. It checks the staged bytes against the working files and recorded provenance hashes. Upgrade reports preserve their saved byte endings; the public notebook uses LF explicitly.

## Answer against local documents

Provide a JSON object mapping source IDs to text:

```powershell
python scripts/upgrade.py answer_upgrade.py --documents data/my-documents.json --question "Your question" --calibration squad
```

The result contains the final answer, cited windows, support score, and an acceptance or failure flag. Malformed and fabricated citations are rejected. Multiple citations are allowed. A low NLI score is a reason to flag or abstain, not proof that an answer is false. A high score is not proof that the claims answer the question. Inspect the cited evidence before relying on an answer.

The CLI can use a question-conditioned check: FLAN-T5-base answers from the cited windows and compares that answer with the proposal. `--verification selected` uses it only where it improves development F1; `claims` keeps the original NLI path and `question` forces the comparison path. This is model agreement, not independent evidence of truth. The research reports preserve both paths, including cases where verification rejects a supported answer.

No paid APIs are used. The measured configurations use local unquantized models; Qwen3-4B and production injection resistance remain untested.

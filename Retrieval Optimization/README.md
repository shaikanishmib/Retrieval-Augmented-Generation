# Retrieval experiments on SciFact and FiQA

Does a more complicated retrieval pipeline actually find better evidence? I tested that question on the BEIR SciFact and FiQA-2018 datasets, comparing keyword search, two sentence-embedding models, rank fusion, reranking, and several ways of splitting documents.

This is a **retrieval study**, not an answer-generating RAG application. The evaluation asks whether relevant source documents appear near the top of the search results. It does not measure generated answers or factual accuracy.

## Results

The main comparison uses the full test splits: 300 queries from SciFact and 648 from FiQA. Chunk-level results are mapped back to their source document IDs before scoring.

| Retrieval method | SciFact nDCG@10 | FiQA nDCG@10 |
| --- | ---: | ---: |
| BM25 | 0.6618 | 0.2321 |
| all-MiniLM-L6-v2 | 0.6451 | 0.3687 |
| multi-qa-MiniLM-L6-cos-v1 | 0.5455 | 0.3599 |
| Reciprocal rank fusion (selected on validation data) | **0.7068** | **0.3817** |
| TinyBERT reranking, up to 20 candidates | 0.6762 | 0.3350 |
| TinyBERT reranking, up to 50 candidates | 0.6621 | 0.3070 |

Rank fusion did best on both test sets. The difference between the BM25 and dense baselines is also worth noting: BM25 was stronger than either encoder on SciFact, while dense retrieval was substantially stronger on FiQA.

Adding TinyBERT reranking did **not** help. Its scores fell below fusion on both datasets, and increasing the candidate pool from 20 to 50 made them worse. The corpus-derived pseudo-relevance expansion experiments also reduced retrieval quality. These are useful failures: more stages do not automatically make a retrieval system better.

Whole-document indexing performed best in the BM25 chunking validation comparisons. That finding should not be generalized to dense retrieval, where the effects of segmentation can differ. For per-query failures, candidate counts, paired uncertainty estimates, truncation checks, and resource measurements, see the [research notebook](notebooks/01_public_retrieval_research.ipynb) and [aggregate metrics](reports/research/metrics.csv).

## What I compared

The study includes four document representations: whole documents, fixed windows, recursive boundary splitting, and sentence boundary splitting. It evaluates BM25 alongside exact cosine similarity, exhaustive FAISS search, and persistent Qdrant in embedded mode, using two MiniLM encoders. I also tested reciprocal rank fusion, cross-encoder reranking, and query transformations derived from the corpus.

Chunking and fusion settings were chosen on validation data before the test comparisons. The reported metrics include nDCG@10, MRR@10, MAP@10, and Recall@1/3/5/10/20. Tests cover retrieval formulas, document-level aggregation, selection behavior, leakage prevention, and dataset integrity.

There are important boundaries to the measurements:

- Search latency comes from a sequential local pass with query embeddings precomputed; it excludes dense query encoding. Reranking time includes candidate retrieval and cross-encoder scoring.
- Memory observations include sampled RSS and a Windows process-lifetime peak working-set counter. They are not isolated per-index memory benchmarks.
- Qdrant ran locally in persistent embedded mode, not as a server. FiQA exceeds the client's recommendation of 20,000 points for embedded operation.
- Semantic chunking and generative HyDE were not run. No answer generation or faithfulness evaluation was performed.

## Run the study

Use Python 3.12 or newer. From the `rag-retrieval-optimization` directory, run these commands in PowerShell:

```powershell
python -m venv data/research-venv
data/research-venv/Scripts/python.exe -m pip install -e ".[research,dev]"
data/research-venv/Scripts/python.exe scripts/acquire.py
data/research-venv/Scripts/python.exe scripts/acquire_models.py
data/research-venv/Scripts/python.exe -m pytest
data/research-venv/Scripts/python.exe scripts/run_research.py
data/research-venv/Scripts/python.exe scripts/build_notebook.py
data/research-venv/Scripts/python.exe scripts/execute_notebook.py
data/research-venv/Scripts/python.exe scripts/publish_notebook.py
```

The acquisition script will not overwrite existing dataset extractions. To check data you already have, run `scripts/acquire.py --verify-only`. The model revisions and hashes are recorded in [models.lock.json](models.lock.json); the executed dependency environment is recorded in [reports/requirements-lock.txt](reports/requirements-lock.txt). For fresh encoding timing measurements, use a separate local experiment directory rather than reusing cached matrices.

The [public notebook](notebooks/01_public_retrieval_research.ipynb) contains the research code without saved outputs or execution counts. The executed local notebook, `notebooks/01_complete_retrieval_research.ipynb`, retains its actual results and figures but is excluded from Git. The publication script verifies that the two notebooks have identical cell sources and does not alter the local report. The notebook builder deliberately refuses to overwrite an executed report.

For an optional process-memory observation, run `scripts/monitor_process.py --pid <PID>` from another terminal with the project Python executable while the experiment runs. The runner prints its PID.

## Data and project files

[DATA.md](docs/DATA.md) documents the original sources, checksums, splits, attribution, and unresolved text-reuse rights. FiQA's 38 empty document bodies were left unchanged. Raw documents, model files, embeddings, indexes, per-query diagnostics, and the executed notebook remain local rather than being committed.

The reusable implementation is in [src/rag_data](src/rag_data); [tests](tests) contains the automated checks. Aggregate results and environment details are in [reports/research](reports/research), and [notebook-verification.json](reports/notebook-verification.json) records the public-copy checks. The small test fixtures are correctness examples, not benchmark data.

## Limitations and Future Improvements

The benchmark covers SciFact and FiQA, so the results do not show how the methods transfer to other domains. Relevant-document coverage and ranking both limit the pipeline: BM25 was stronger on SciFact, dense retrieval was stronger on FiQA, and TinyBERT reranking lowered nDCG@10 on both test sets. The study used MiniLM encoders and TinyBERT, and it does not establish that a larger model would perform better. FiQA's candidate pool and the domain gap between scientific claims and finance questions warrant further study.

The timing pass precomputed query embeddings, so reported search latency excludes online dense query encoding. Index and memory measurements came from local runs; embedded Qdrant exceeded its documented recommended point count for FiQA. The judgments score document relevance, which does not establish that an answer generated from a document is factually correct.

Future comparisons can measure candidate coverage and ranking together across more domains, test stronger retrieval representations and ranking models, and include query encoding, indexing, storage, and end-to-end latency in matched measurements. Those changes remain untested here.

## Phase 1 results
The objective was evidence ranking across BEIR SciFact (300 test queries) and FiQA (648), using BM25, two MiniLM encoders, reciprocal rank fusion, and TinyBERT. [Aggregate metrics](reports/research/metrics.csv) report fusion nDCG@10 of 0.7068 / 0.3817 for SciFact / FiQA versus BM25 0.6618 / 0.2321. Development-selected fusion is the strongest recorded reference. TinyBERT reranking and pseudo-relevance expansion failed to improve it. Whole-document BM25 validation does not establish the best dense segmentation policy.

Search latency excludes dense query encoding. Embedded Qdrant, sampled process memory, fixed local corpora, and untested generative query methods limit generalization. Model and source provenance remain in the original locks; the public notebook linked above retains the full experimental design.

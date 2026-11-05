# Dataset provenance and use

This project uses only the original BEIR SciFact and FiQA-2018 benchmark archives. Raw files remain local under `data/raw/`. The repository publishes acquisition code, checksums, validation summaries, aggregate experimental measurements, and an output-free notebook.

## Sources and immutable pins

The [BEIR dataset catalog](https://github.com/beir-cellar/beir/wiki/Datasets-available) links the original benchmark downloads. Its download paths have no release number. `sources.lock.json` pins each downloaded archive by byte size and SHA-256. `reports/dataset-validation.json` records the sizes and SHA-256 hashes of each extracted corpus, query file, and qrels file. `reports/fiqa-acquisition.json` records the FiQA download completion timestamp in UTC, derived from the downloaded file's modification time. SciFact was already present; its prior acquisition record supplies a date rather than an exact timestamp.

| Archive | Bytes | SHA-256 |
| --- | ---: | --- |
| `scifact.zip` | 2,816,079 | `536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165` |
| `fiqa.zip` | 17,948,027 | `32c7df99ed21252fdfb2cf3f5673502a8d245ee0c44c4a133570d92ce2b3ad02` |

The original SciFact release archive is retained as provenance evidence, with its own pin in `sources.lock.json`. It does not replace the BEIR-format files used in evaluation. The acquisition script verifies archive bytes before extraction, rejects traversal paths and symlinks, caps extraction size, and refuses to overwrite an existing extraction.

## Records and splits

| Dataset | Corpus documents | Query records | Train judged queries / pairs | Dev judged queries / pairs | Test judged queries / pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| SciFact | 5,183 | 1,109 | 809 / 919 | Not supplied by BEIR | 300 / 339 |
| FiQA-2018 | 57,638 | 6,648 | 5,500 / 14,166 | 500 / 1,238 | 648 / 1,706 |

Every qrels grade in these snapshots is 1. All referenced query and document IDs exist. No query IDs overlap between splits. The validator checks records, required field types, duplicate IDs, duplicate text, qrels pairs, relevance distributions, and references against the actual extracted files.

Each dataset has one shared corpus across its query splits. Train, development, and test define query and judgment partitions, not separate document collections.

`scripts/verify_scifact_origin.py` also verifies the original SciFact release. BEIR train corresponds to original `claims_train.jsonl`; BEIR test corresponds to original `claims_dev.jsonl`. Every query text matches. The BEIR positive pairs equal the original `cited_doc_ids` pairs, which include cited documents without annotated evidence; they are not limited to evidence-bearing documents. All corpus IDs and titles match, and concatenated abstracts match after whitespace normalization. `reports/scifact-original-mapping.json` records these checks.

FiQA contains 38 empty document bodies and 57,638 empty titles. Its duplicate-text excess is 39, including repeated empty strings. SciFact has four duplicate query texts beyond their first occurrences. No observations are filled in, merged, relabeled, or removed. Empty FiQA records remain in the corpus and receive the embedding model's empty-input representation. Text duplicates remain separate benchmark IDs. Length-filtering experiments change candidate eligibility only.

FiQA also has 621 identical ID strings across the query and document namespaces. Those namespaces are independent. Matching ID strings do not imply identical content, and the retrievers do not exclude such candidates.

Validation uses at most 150 queries in deterministic SHA-256 ID order. FiQA uses its original development labels; SciFact uses training labels because no BEIR development split exists. Test IDs and normalized exact text matches are excluded from validation selection. Local `validation-selection.json` files record the selected IDs. The project does not tune on final test judgments.

## Attribution and rights

SciFact is described by Wadden et al., *Fact or Fiction: Verifying Scientific Claims*, EMNLP 2020. The [authors' license statement](https://github.com/allenai/scifact/blob/68b98a56d93e0f9da0d2aab4e6c3294699a0f72e/LICENSE.md) assigns CC BY 4.0 to claims and evidence annotations and ODC-By 1.0 to the abstract database derived from S2ORC. The database license does not settle independent publisher rights in every abstract. Earlier S2ORC releases also carried noncommercial terms; the local evidence files are retained. This project does not redistribute titles, abstracts, claims, or judgments.

FiQA-2018 is described by Maia et al., *WWW'18 Open Challenge: Financial Opinion Mining and Question Answering*. The [original challenge site](https://sites.google.com/view/fiqa/home) and [challenge paper](https://andrefreitas.org/papers/www_fiqa_2018.pdf) document the task. Mirror license labels do not establish a blanket grant for underlying financial posts. Original item-level rights and any applicable noncommercial restrictions remain with their owners. This project uses the downloaded benchmark locally and publishes no raw financial posts, queries, judgments, or attribution-link exports. Complete underlying-text redistribution rights have not been established.

BEIR is described by Thakur et al., *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models*, NeurIPS Datasets and Benchmarks 2021. BEIR's maintainers explicitly distinguish format distribution from permission to use underlying datasets. Anyone rerunning this study must review the original terms for their intended use.

## Verification

Run `python scripts/acquire.py --verify-only` from the project directory to verify the archived bytes. The integration tests and research notebook validate the extracted records. Original IDs, grades, and split files are retained. All raw directories are ignored by Git; only aggregate summaries and file hashes are intended for publication.

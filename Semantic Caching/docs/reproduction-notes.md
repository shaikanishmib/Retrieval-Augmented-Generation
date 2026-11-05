# Reproduction notes

The recorded workspace reuses existing weights and Python packages through read-only paths. Public source and model locks stand on their own; a fresh environment acquires missing assets under `Semantic Caching/data`. Raw documents, weights, backend stores, signing keys, predictions and the executed notebook are ignored by Git.

The model lock stores workspace-relative bindings as well as immutable model revisions and file hashes. When acquisition cannot find those paths, it downloads into this project's data directory and updates the local bindings. That changes the run identity even when model bytes match. Begin a separate run and rerun development selection before evaluation. Do not reuse published freeze files as if they came from the new environment.

`prepare_data.py` reconstructs published question and pair IDs from the project's validation report. It checks full-row hashes. Prior-project internals are needed only for the original exclusion audit, not for reconstruction of the published sample.

The initial audit encountered Windows restrictions on hardware inspection, shared package imports and Git ownership. Read access outside the restricted runner was used for local computation; Git commands use a process-local `safe.directory` setting. No global ownership setting was changed. FAISS and Qdrant packages were downloaded into the ignored project package directory. Qdrant local mode required processing pywin32's package path file on Windows; the launcher does so. Redis was not installed.

The first PAWS URL was incorrect. The official URL then returned HTTP 403. Acquisition uses a pinned Hugging Face mirror and measured checksums, with the upstream dataset license linked in the lock. No checksum is described as independently supplied by the original authors.

Development found that the first hybrid path duplicated answer-cache behavior. Its selection artifacts and partial timings were preserved under `data/superseded-hybrid-v1`. The corrected policy adds evidence fallback with fresh generation. The final selection was recomputed before held-out trials. Extending source identity to cover experimental drivers caused a checkpoint assertion before final evaluation; the core selection hashes were verified and the identity extension was recorded. These superseded timings do not enter the reported aggregates.

A later baseline audit found whole-prompt truncation could remove questions placed after long evidence. All affected trials and development outputs were preserved under `data/superseded-prompt-v1`. Prompt version `qa-v2-question-preserved` reserves the instructions, complete question and answer suffix before packing evidence; it raises an error if the question alone exceeds the budget. Tests cover long evidence and oversized questions. Model selection and all final trials were rerun under this version. Publication also checks every generated request trace for the complete question and the token limit. The prompt revision report retains the affected artifact hashes.

Scope hardening then replaced model-alias-only bindings with exact generator and embedding revision identities. A same-alias revision-change integration test confirms misses after either revision changes. Early affected trials were retained under `data/superseded-scope-v1`; final development selection and evaluation were repeated after the change. The scope revision report records their hashes.

The final source identity includes source modules, configuration, dataset/model locks and the development/evaluation/ablation drivers. Runtime versions and device observations are separately recorded; a different machine or dependency environment requires a new run. A checkpoint represents one complete timed trial. Interrupted trials rerun. Each timed request still performs the real inference required by its policy.

The notebook executor refuses to replace an executed notebook. If a new run is needed, preserve the complete local notebook under a distinct ignored history directory first. The public copy must always be produced from the completed local notebook after writing review, with identical cell types, source and order.

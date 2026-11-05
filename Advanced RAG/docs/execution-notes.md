# Execution notes

The first package download attempt was blocked by the restricted network sandbox. Downloads then ran with approved network access and installed dependencies in ignored project storage. The global Python environment was not used as the installation target for the successful build.

The original HotpotQA CMU host failed DNS resolution. A guessed JSON path in the Hugging Face mirror returned 404. Inspection of the maintained mirror identified its Parquet validation file, which was pinned by commit and independently checked against the mirror's published LFS hash. Conversion preserves questions, answers, contexts, and supporting facts.

The first original-data integration check rejected out-of-range HotpotQA supporting-sentence indices. Validation now records those original annotation defects without changing labels. Document-level scoring uses their source titles. The counts and affected IDs are in `reports/hotpot-validation.json`.

Tests run before experiments and notebook publication skip the corresponding absent-artifact checks. The final validation must run those checks after artifacts exist. `reports/tests.xml` records the final pass.

The full executed notebook remains ignored in local storage. Failed notebook executions, if any, are retained separately under `data/failed-execution.ipynb`. No cleanup removes Project 1 or unrelated files. Bytecode and local test artifacts are ignored and can remain for inspection.

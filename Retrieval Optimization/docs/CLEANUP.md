# Project cleanup

The initial inventory measured 3,503 files totaling 1,524,078,043 bytes. Cleanup was limited to `rag-retrieval-optimization`. Existing changes in `sec-filing-intelligence` and all sibling projects were left untouched.

| Removed material | Files | Deleted file bytes | Recoverable backup bytes | Net reclaimed file bytes |
| --- | ---: | ---: | ---: | ---: |
| Project pytest copies and Python bytecode | 1,110 | 442,702,609 | 0 | 442,702,609 |
| Superseded experiments, datasets, archives, and local environment | 2,345 | 1,064,312,467 | 288,624 | 1,064,023,843 |
| Superseded aggregate reports and figures | 26 | 2,396,693 | 684,279 | 1,712,414 |
| Obsolete issue evidence | 2 | 5,603 | 2,083 | 3,520 |
| Total | 3,483 | 1,509,417,372 | 974,986 | 1,508,442,386 |

These measurements use actual file lengths. They do not measure filesystem allocation, compression, or changes to available volume capacity. New research dependencies, FiQA data, models, and experiment artifacts consume additional space; the table describes the cleanup itself.

Every removed file has an entry in a local manifest under `reports/cleanup/`, including size, reason, and regeneration or backup status. Unique source files, notebooks, documentation, and reports were preserved in `.local/legacy-work.zip` and `.local/legacy-reports.zip`. A later review identified two downloaded issue records as exclusively obsolete; they were preserved in `.local/obsolete-evidence.zip`. Archive integrity and file hashes were checked before removal. Those archives remain local and are ignored by Git. The cleanup also removed 373 empty legacy directories.

The active project retains SciFact archives, extracted records, and provenance evidence. It also retains the reusable acquisition safeguards, record validation, BM25 implementation, and evaluation logic, with corrections and new tests. Obsolete dataset adapters, dataset-specific tests, source locks, notebooks, reports, and downloaded artifacts were retired. No global or shared cache was cleaned.

After verification, cleanup removed another 106 temporary files containing 1,011,881 bytes. Logs and the preliminary EDA notebook were preserved in the 263,699-byte `.local/verification-artifacts.zip`; the remaining files were reproducible test fixtures and bytecode. This phase reclaimed another 748,182 bytes and removed 391 empty directory entries, including pytest links. One test directory required elevated filesystem access because the earlier elevated test run created it with a different Windows owner.

Across both phases, cleanup removed 3,589 file instances and 764 empty directory entries. Recoverable archives occupy 1,238,685 bytes, leaving 1,509,190,568 net reclaimed file bytes. Some paths were recreated during testing and removed again. `reports/cleanup/complete-cleanup-manifest.json` lists every deletion, its measured size, reason, and regeneration or backup status.

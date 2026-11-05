# Reference findings

Project 1 contains a 76-cell public notebook, seven reusable data and retrieval modules, acquisition and execution scripts, and tests for formulas, provenance, split separation, notebook equivalence, and report completeness. Its published comparison uses SciFact and FiQA. The local raw datasets, model snapshots, and executed notebook were absent in this checkout.

Its BM25 implementation uses postings and deterministic ties. Dense retrieval uses normalized embeddings and maps window hits back to source documents. Validation chooses chunking and fusion settings before final scoring. Per-query outputs support paired bootstrap comparisons, failure analysis, and audit checks. Reported reranking and expansion failures remain in the results.

The notebook builder refuses to replace executed outputs. Execution preserves prior runs, and publication clears a deep copy before checking source equivalence. Publication audits reject local artifacts and unrelated staged paths. These safeguards define Project 2's notebook and Git boundaries.

Advanced RAG independently implements the retrieval formulas in its own package. It uses exact cosine search because the candidate pools are small. It adds answer generation, attribution traces, selective prediction, and answer scoring. It has no runtime import from Project 1. The reference inventory records hashes for every original file and tests verify that those bytes remain unchanged.

The old reference reports contain historical directory names that differ from this checkout. Those reports remain unchanged. Reproduction instructions in Advanced RAG use its actual directory name.

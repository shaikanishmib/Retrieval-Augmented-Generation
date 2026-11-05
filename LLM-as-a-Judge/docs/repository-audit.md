# Lessons from the existing projects

The initial Git revision was `ef071ec6eb4ccbb944309b4cb4b4f230de6cc2cb`. The new target directory did not exist. No cleanup was required. The audit recorded earlier tracked-file hashes locally, verified reused model files, and inventoried cache and environment locations. Previous model weights and package overlays are read without modification.

| Existing component | What the implementation establishes | Consequence for this study |
| --- | --- | --- |
| Advanced RAG `upgrade/grounding.py` | Rejects malformed claim schemas and fabricated source IDs; a normalized answer anchor checks answer/claim consistency; NLI supplies separate support estimates | Citation validity, claim consistency and evidence support need separate endpoints |
| Advanced RAG `answer_support.py` | Compares a proposed answer with a second answer generated from cited evidence, alongside NLI | Model-to-model agreement is a diagnostic rather than independent truth |
| Advanced RAG `evaluation.py` | Compares citation source titles with dataset gold sources | Source-label precision is not claim-level attribution precision |
| Agentic RAG `models.py` | Reads NLI label names and records truncation at 512 tokens | Verify label ordering for each checkpoint and disclose context limits |
| Agentic RAG `controller.py` and `evaluation.py` | Assigns citations through lexical overlap and scores evidence recall independently of answer EM/F1 | More retrieved gold evidence does not establish a better supported answer |
| Semantic Caching `metrics.py` and cache study | Separates reuse equivalence from QA scoring, and reports false hits with request and hit denominators | High embedding similarity needs independent agreement checks and clear denominators |

The published Advanced RAG upgrade reports restrictive verification and structured-generation failures. Agentic RAG reports stronger evidence recall without consistent answer improvement. Semantic Caching reports incompatible semantic reuse despite high similarity. These findings motivate the hypotheses; they do not supply new human support labels.

Earlier local traces exist, but their assistant-authored qualitative reviews lack the independent annotation required for this project's main endpoint. They are excluded from the benchmark. The new experiments use pinned RAGTruth annotations and a separate model-annotated transfer subset. Fictional fixtures provide reproducible contradiction, citation and injection probes and are explicitly separated from human agreement.

The publication audit compares every earlier tracked file against its initial byte hash. It also verifies that staging contains only the new project and the root README. Earlier executed notebooks, caches, reports and model weights remain in their original locations.

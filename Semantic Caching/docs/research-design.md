# Research design

The primary question is whether semantic caching reduces end-to-end RAG latency and computation while keeping answer reuse, freshness and access errors acceptable. Exact caching is the first comparator. A cache can repeat an incorrect baseline answer perfectly; reuse precision and QA correctness are separate outcomes.

## Prespecified stages

1. Audit the host, dependencies, existing weights and datasets. Hash previous project artifacts before implementation. Preserve all existing work.
2. Acquire PAWS-Wiki Labeled Final and verify authentic SQuAD 2.0 and HotpotQA files. Use PAWS only for annotated sentence equivalence, not QA correctness. PAWS pairs are algorithmically proposed and human-labeled; they are not human-authored question paraphrases. QQP is excluded because its redistribution and reconstruction conditions add work without resolving QA equivalence.
3. Allocate QA by source groups, excluding prior experiment question IDs. SQuAD articles and selected disjoint HotpotQA source pools separate development from evaluation. Keep full record IDs and checksums locally and publish ID-only manifests. Synthetic transformations of genuine questions are explicitly synthetic and have no assumed QA label unless the transformation preserves the original question literally apart from a request wrapper.
4. Compare available MiniLM and BGE encoders with a lexical TF-IDF reference. TF-IDF is a third feasible embedding method, not a third pretrained neural model. Compare FLAN-small and FLAN-base on development QA. Keep the selected generator fixed across strategies.
5. Select a similarity threshold using development equivalence pairs only. Require zero observed false acceptances and maximize true acceptances; if none qualifies, disable semantic answer reuse. Also retain permissive threshold ablations to expose harm. Zero development errors cannot establish a production error bound.
6. Freeze configuration and source identity before final workloads. Run no-cache, exact, semantic answer, evidence and hybrid policies on identical sequential requests with fresh cache state per policy and trial. Three timing trials use deterministic orders. A separately measured model load and discarded warm-up precede requests. All misses perform real generation; checkpoints never replace timed inference within a trial.
7. Run controlled source updates, deletion, revocation, tenant/user permission changes, TTL, capacity, concurrent access, poisoning, serialization and restart probes. Use synthetic local documents and identities. Authorization and corpus compatibility precede returning an answer or evidence.
8. Aggregate traces, paired family bootstrap intervals and failure cases. Execute the notebook in a clean kernel; preserve executed output locally and publish an identical source-only copy. Audit normalized Git bytes and prior project hashes before publication.

## Workloads and estimands

QA workload families include unique queries, exact repeats, synthetic request-wrapper paraphrases, nearby different authentic questions, and explicit synthetic entity/number/date/negation stress tests. The last group tests rejection rather than annotated QA accuracy. SQuAD includes genuine unanswerable questions. Workload proportions are fixed before model selection. Cold-cache, warm-cache, unique-heavy and capacity-shift sequences are reported separately.

Annotated PAWS development and held-out subsets measure pair acceptance precision and false-positive rate at each threshold. Sentence overlap is removed across partitions; missing source-document mapping limits document-level leakage claims. QA family and source-document separation is independently enforced. PAWS sentences do not enter the QA RAG corpus.

Primary metrics are per-request end-to-end p50/p95, correct reuse precision, false hits per accepted hit and per request, EM/F1, and avoided generation calls. Retrieval reuse still incurs generation. Record embedding, lookup, search, retrieval, generation and insertion time separately. Report actual memory, serialized bytes, incremental insertion and restart costs. CUDA event timing estimates generation GPU elapsed time; it does not measure energy, billable cost or exclusive device utilization.

Paired bootstrap resamples HotpotQA question families and SQuAD articles with 2,000 replicates and fixed seeds. Shared HotpotQA evaluation source pages can make question-level intervals optimistic. Timing trials share a host and are not independent hardware replications. Confidence intervals are descriptive and pointwise. Zero observed security failures establish only the tested fixture properties.

## Eligibility and threat boundary

Cache keys include tenant, user, permission epoch, allowed-document set, corpus digest, prompt version and model revision. Each hit also checks current source existence, visibility, version and content hash. Unknown authorization fails closed. A full corpus digest deliberately invalidates entries after any corpus change, including newly added competing evidence. This is conservative and can lower hit rate.

Trusted application code alone inserts entries. An HMAC over persisted payloads detects modification using a separately held key; this cannot prevent compromised trusted writers or keys. SQLite uses transactions and WAL. An in-process lock covers lookup, eviction and insertion; generation outside the lock may duplicate concurrent misses. No claim of distributed single-flight or production security is made.

Semantic answer reuse remains experimental even with lexical guards: equal numbers and negation do not prove equivalence. Evidence reuse regenerates answers but can retrieve unsuitable context. Exact reuse with current authorization and corpus verification is the stricter correctness reference.

The corrected hybrid policy first attempts exact answer reuse, then semantic answer reuse at the development-selected threshold, then semantic evidence reuse at 0.95 with fresh generation. The evidence threshold is a prespecified experimental setting, not a validated equivalence boundary. The superseded implementation and partial traces are preserved locally; the development revision records this correction. Final held-out trials follow the corrected freeze. Corpus scopes are task-specific candidate pools. This deliberately blocks cross-task reuse when the supplied retrieval corpora differ.

# Threat model

The protected assets are answer correctness, evidence attribution, tenant-scoped documents, synthetic confidential canaries, the read-only tool registry, and the integrity of cached answers and retrieval metadata. A legitimate reader asks a factual question and expects a concise answer grounded in permitted evidence. A legitimate controller may request a document lookup. It has no export, shell, write, network or message-sending capability.

| Level | Attacker control | Boundary and measurement |
| --- | --- | --- |
| A | One registered passage in the retrieval fixture | Injection, factual replacement and fabricated citations; compare the changed bytes with the immutable registry |
| B | One submitted document in an isolated ingestion queue | Unapproved writes cannot replace registered IDs; a separate admitted-untrusted-document diagnostic tests the limits of content hashes |
| C | One local read-only lookup response | Model-proposed actions are logged; Python checks the schema, allowlist, current permissions and call budget |
| D | One proposed cache write | An untrusted candidate cannot become an approved answer; approved entries depend on tenant, user, current permission set, authorization epoch, corpus snapshot, question and model |
| E | One authenticated synthetic tenant | Try another tenant's overlapping topic on retrieval misses and cache hits; distinguish content exposure from blocked attempts |

The main D0 reference retains ordinary tenant permission checks and the read-only tool dispatcher. It receives ordinary task instructions and source IDs. The separate deliberately broken retrieval and shared-key cache comparators demonstrate application anti-patterns; their failures are never attributed to the ordinary D0 model configuration.

Retrieved text, source titles, uploaded documents, model answers and tool responses are untrusted. Identity, permissions, approved document IDs and versions, the canonical registry, model files and application code are trusted. An attacker cannot change a system instruction, approve an upload, alter the permission map or modify the registry. The admitted-document diagnostic deliberately registers an attacker's submission as an untrusted source; approval there means admission to retrieval, not certification of factual truth.

Hashes detect a mismatch against a trusted snapshot. They cannot recognize a false claim that was present when a document was registered. A provenance defense that drops the only useful document can lose the legitimate answer. Delimiters communicate boundaries to a model but do not enforce them. Injection detectors make errors on benign instructional text and on new attack families.

The application enforces permissions before ranking. Authorization reads canonical metadata. An index-ID substitution fixture demonstrates that ID-only authorization can still return attacker-mislabeled bytes; complete integrity checks are required when index entries are untrusted. The combined permission/integrity path blocks that case. Cache eligibility is checked before content is returned. Source revocation or version changes alter the scope. The queue rejects duplicate IDs. Tool schemas accept exactly two bounded string fields; only authorized `lookup` calls execute. Output validation rejects unregistered citations and supplied synthetic canaries, but valid source IDs do not establish evidential support.

Experiments use fixed local objects, at most 64 candidate documents, 12,000 characters per question or passage, four retrieved documents, two tool calls and 48 generated tokens. The recorded generator context budget is 340 tokenizer tokens, with tokenizer-specific input limits. Generation has a 15-second stopping target; Transformers checks time between decoding steps, so this is not a hard process deadline. No experiment opens a server or sends an offensive request to another system.

Residual risks include unknown detector pretraining overlap, forged facts inside admitted documents, paraphrased instructions, context truncation, unsupported answers with syntactically valid citations, and model-generated action proposals that fail parsing. The canary filter recognizes only the exact synthetic strings supplied to it. It does not demonstrate protection against transformed or unknown secrets. The local principal is a trusted test object, not a tested authentication service. Concurrency, distributed cache races, network services and production identity providers are outside this study.

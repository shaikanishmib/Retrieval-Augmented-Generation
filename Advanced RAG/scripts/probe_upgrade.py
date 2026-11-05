"""Benchmark model adapters on development examples before expanding the study."""
import gc
import argparse
import json
import time
import psutil
import torch
from advanced_rag.data import load
from advanced_rag.pipeline import answer_scores
from advanced_rag.upgrade.artifacts import ROOT, REPORTS, read, save
from advanced_rag.upgrade.models import Generator, Encoder, Reranker, Verifier
from advanced_rag.upgrade.retrieval import Index, context
from advanced_rag.upgrade.engine import plain_prompt, grounded_prompt
from advanced_rag.upgrade.grounding import parse, verify

torch.set_num_threads(4)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--generator", choices=["flan-small", "flan-base", "qwen06", "qwen17"])
args = parser.parse_args()
rows = [dict(row, dataset=name) for name in ("hotpot", "squad") for row in load(name)[0][:4]]
encoder = Encoder("minilm")
indexes = {r["id"]: Index(r["docs"], encoder) for r in rows}
verifier = Verifier()
prior = REPORTS / "feasibility.json"
if prior.exists() and not (REPORTS / "feasibility-initial.json").exists():
    (REPORTS / "feasibility-initial.json").write_bytes(prior.read_bytes())
report = read(prior) if prior.exists() else {"device": "cuda" if torch.cuda.is_available() else "cpu", "torch": torch.__version__, "generators": {}, "retrievers": {}}
for alias in ([args.generator] if args.generator else ("flan-small", "flan-base", "qwen06", "qwen17")):
    if alias != "flan-small" and not (ROOT / "data/upgrade/models" / alias / "config.json").exists():
        continue
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    generator = Generator(alias)
    measurements = []
    for row in rows:
        index = indexes[row["id"]]
        ranking, timing = index.search(row["question"])
        evidence, _ = context(ranking, index, generator, min(768, generator.limit - 220))
        for mode, prompt_fn in (("plain", plain_prompt), ("grounded", grounded_prompt)):
            while evidence and generator.count(prompt_fn(row["question"], evidence)) > generator.limit:
                evidence.popitem()
            result = generator.generate(prompt_fn(row["question"], evidence), limit=128 if mode == "grounded" else 32)
            parsed = parse(result["text"], evidence) if mode == "grounded" else {"answer": result["text"], "valid": True}
            em, f1 = answer_scores(parsed["answer"], row["answers"])
            measurements.append({"id": row["id"], "dataset": row["dataset"], "mode": mode, "em": em, "f1": f1,
                                 "valid": parsed["valid"], **{k: result[k] for k in ("generation_ms", "input_tokens", "output_tokens", "truncated")}})
            save(ROOT / f"data/upgrade/probes/{alias}-{row['id']}-{mode}.json", result)
    report["generators"][alias] = {"load_ms": generator.load_ms, "measurements": measurements,
                                    "rss_bytes": psutil.Process().memory_info().rss,
                                    "gpu_peak_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None}
    save(REPORTS / "feasibility.json", report)
    print(alias, generator.load_ms, report["generators"][alias]["gpu_peak_bytes"], flush=True)
    generator.close()
    del generator
for alias in ("minilm", "bge-small"):
    candidate = encoder if alias == "minilm" else Encoder(alias)
    times = []
    for row in rows:
        index = Index(row["docs"], candidate)
        _, elapsed = index.search(row["question"])
        times.append({"index_ms": index.index_ms, "document_encoding_ms": index.encoding_ms, **elapsed})
    report["retrievers"][alias] = {"load_ms": candidate.load_ms, "measurements": times}
for alias in ("tinybert", "minilm-rerank"):
    reranker = Reranker(alias)
    measurements = []
    for row in rows:
        _, elapsed = indexes[row["id"]].search(row["question"], "rerank", reranker)
        measurements.append(elapsed)
    report["retrievers"][alias] = {"load_ms": reranker.load_ms, "measurements": measurements}
    del reranker
    gc.collect()
save(REPORTS / "feasibility.json", report)
print(json.dumps({alias: {"load_ms": r['load_ms'], "examples": len(r['measurements'])} for alias, r in report['generators'].items()}, indent=2))

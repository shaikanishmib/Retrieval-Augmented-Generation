"""Evaluate the development-winning plain generator through the preserved citation baseline."""
import time
import hashlib
from pathlib import Path
import psutil
import torch
from advanced_rag.data import load
from advanced_rag.pipeline import Models, Pipeline
from advanced_rag.evaluation import score, aggregate
from advanced_rag.upgrade.artifacts import ROOT, LOCAL, REPORTS, read, save, fingerprint, preserved, sources
from advanced_rag.upgrade.models import Generator, Encoder
from sentence_transformers import CrossEncoder

preserved()
torch.set_num_threads(4)
selected = max(read(REPORTS / "selection.json")["plain_generators"], key=lambda r:r["f1"])["generator"]
generator = Generator(selected)
encoder = Encoder("minilm")
reranker = CrossEncoder(str(ROOT / "data/models/reranker"), device=generator.device, local_files_only=True)
models = Models(encoder.model, reranker, generator.tokenizer, generator.model, generator.device)
identity = fingerprint({"script": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "sources": sources(),
                        "models": read(ROOT / "models.upgrade.lock.json"), "torch": torch.__version__, "device": generator.device})
result = {"generator": selected, "selection": "Highest pooled development F1 in the prespecified plain-answer comparison",
          "scope": "Full-sample extension of the development-winning generator, using the unchanged baseline retrieval and post hoc lexical citations",
          "device": generator.device, "generator_dtype": str(generator.model.dtype), "model_loading_ms": generator.load_ms,
          "concurrent_with_structured_study": not (REPORTS / "resources.json").exists(),
          "timing_scope": "Sequential CPU baseline pass while the separate GPU study is active; shared host load is uncontrolled",
          "citation_limit": "This comparator retains the original lexical citation assignment and does not establish improved support verification", "datasets": {}}
for name in ("hotpot", "squad"):
    _, rows, _ = load(name, 64, 320)
    groups = {"original": {m: [] for m in ("hybrid", "rerank")}, "fresh": {m: [] for m in ("hybrid", "rerank")}}
    for i, row in enumerate(rows):
        path = LOCAL / "practical" / identity / (fingerprint({"row": row, "generator": selected}) + ".json")
        if path.exists():
            record = read(path)
        else:
            start = time.perf_counter()
            pipeline = Pipeline(models, row["docs"])
            indexing = (time.perf_counter() - start) * 1000
            record = {"id": row["id"], "indexing_ms": indexing, "predictions": {m:pipeline.run(row["question"], m) for m in ("hybrid", "rerank")}}
            save(path, record)
        split = "original" if i < 64 else "fresh"
        for method in groups[split]:
            groups[split][method].append(score(row, record["predictions"][method]))
        if (i + 1) % 64 == 0:
            print(name, i + 1, len(rows), flush=True)
    result["datasets"][name] = {split: {m:aggregate(values) for m, values in methods.items()} for split, methods in groups.items()}
    save(LOCAL / "practical" / f"{name}-scores.json", groups)
    save(REPORTS / "practical-baseline.json", result)
memory = psutil.Process().memory_info()
result["resources"] = {"observed_rss_bytes": memory.rss, "windows_process_peak_bytes": getattr(memory, "peak_wset", memory.rss),
                       "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None}
save(REPORTS / "practical-baseline.json", result)
preserved()

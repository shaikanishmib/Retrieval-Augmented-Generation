"""Measure encoder throughput without consulting relevance judgments."""

import json
import sys
import time
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]


def main():
    texts = []
    with (ROOT / "data/raw/scifact/corpus.jsonl").open(encoding="utf-8") as stream:
        for _, line in zip(range(64), stream):
            row = json.loads(line)
            texts.append(row["title"] + " " + row["text"])
    model = SentenceTransformer(str(ROOT / "data/models/all-MiniLM-L6-v2"), local_files_only=True, device="cpu")
    model.max_seq_length = 256
    results = []
    for threads in (4, 8, 12):
        torch.set_num_threads(threads)
        start = time.perf_counter()
        model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        results.append({"threads": threads, "seconds": time.perf_counter() - start})
    path = ROOT / "reports/encoding-thread-probe.json"
    path.write_text(json.dumps({"documents": 64, "relevance_labels_used": False, "concurrent_experiment": True, "measurements": results}, indent=2))
    print(json.dumps(results))


if __name__ == "__main__":
    main()

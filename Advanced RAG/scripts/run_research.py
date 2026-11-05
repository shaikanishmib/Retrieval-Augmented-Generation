"""Run the fixed local comparisons and retain per-question traces."""
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import psutil
from advanced_rag.data import load
from advanced_rag.evaluation import score, aggregate, paired
from advanced_rag.pipeline import Models, Pipeline

ROOT = Path(__file__).resolve().parents[1]

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")

def main():
    config = json.loads((ROOT / "configs/study.json").read_text())
    models = Models.load(ROOT)
    import torch
    environment = {"python": sys.version, "platform": platform.platform(), "device": models.device, "torch": torch.__version__,
                   "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "src").rglob("*.py")},
                   "config": config, "api_charges_usd": 0, "cost_scope": "No paid APIs. Electricity and hardware cost were not measured."}
    save(ROOT / "reports/environment.json", environment)
    summary, statistics = {}, {}
    for name in ("hotpot", "squad"):
        dev, test, validation = load(name, config["development_questions"], config["heldout_questions"])
        save(ROOT / f"reports/{name}-validation.json", validation)
        traces = {"development": {}, "heldout": {}}
        # Choose the abstention threshold on development predictions only.
        for row in dev:
            pipeline = Pipeline(models, row["docs"])
            traces["development"][row["id"]] = pipeline.run(row["question"], "hybrid", config["context_budget"])
        tuning = []
        for threshold in config["threshold_candidates"]:
            values = []
            for row in dev:
                p = dict(traces["development"][row["id"]])
                if p["support_overlap"] < threshold:
                    p.update(answer="", citations=[])
                values.append(score(row, p))
            tuning.append({"threshold": threshold, **aggregate(values)})
        chosen = max(tuning, key=lambda r: r["f1"])["threshold"]
        save(ROOT / f"reports/{name}-selection.json", {"candidates": tuning, "selected_threshold": chosen, "selection_metric": "development answer F1"})
        by_method = {m: [] for m in config["methods"] + ["abstention", "budget180"]}
        rss, builds = [], []
        for number, row in enumerate(test):
            started = time.perf_counter()
            pipeline = Pipeline(models, row["docs"])
            builds.append((time.perf_counter() - started) * 1000)
            per = {}
            for method in config["methods"]:
                if method == "corrective":
                    initial = per["hybrid"]
                    if initial["support_overlap"] < .5 or not initial["answer"]:
                        p = pipeline.run(row["question"], method, config["context_budget"])
                        p["total_ms"] += initial["total_ms"]
                        p["input_tokens"] += initial["input_tokens"]
                        p["output_tokens"] += initial["output_tokens"]
                        p["retried"] = True
                    else:
                        p = dict(initial, retried=False)
                else:
                    p = pipeline.run(row["question"], method, config["context_budget"], oracle=row["gold"] if method == "oracle" else None)
                per[method] = p
                by_method[method].append(score(row, p))
            p = dict(per["hybrid"])
            if p["support_overlap"] < chosen:
                p.update(answer="", citations=[])
            per["abstention"] = p
            by_method["abstention"].append(score(row, p))
            per["budget180"] = pipeline.run(row["question"], "hybrid", 180)
            by_method["budget180"].append(score(row, per["budget180"]))
            traces["heldout"][row["id"]] = {"question": row["question"], "references": row["answers"], "gold": row["gold"], "kind": row["kind"], "predictions": per}
            rss.append(psutil.Process().memory_info().rss)
            save(ROOT / f"data/experiments/{name}-traces.json", traces)
            print(name, number + 1, len(test), flush=True)
        save(ROOT / f"data/experiments/{name}-scores.json", by_method)
        summary[name] = {method: aggregate(values) for method, values in by_method.items()}
        statistics[name] = {method: paired([r["f1"] for r in by_method["hybrid"]], [r["f1"] for r in values], config["bootstrap_replicates"]) for method, values in by_method.items() if method != "hybrid"}
        save(ROOT / f"reports/{name}-resources.json", {"max_sampled_rss_bytes": max(rss), "build_p50_ms": float(np.median(builds)),
              "build_p95_ms": float(np.percentile(builds, 95)), "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated() if models.device == "cuda" else None,
              "scope": "RSS sampled after each question; model loading and document encoding excluded from per-answer latency. CPU/GPU peaks are process observations."})
        save(ROOT / "reports/metrics.json", summary)
        save(ROOT / "reports/paired-bootstrap.json", statistics)

if __name__ == "__main__":
    main()

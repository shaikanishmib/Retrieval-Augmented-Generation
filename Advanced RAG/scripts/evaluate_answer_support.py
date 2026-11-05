"""Evaluate a question-aware support replay with development-only threshold selection."""
import hashlib
import argparse
from pathlib import Path
import numpy as np
import torch
from advanced_rag.data import load
from advanced_rag.pipeline import answer_scores
from advanced_rag.answer_support import check, accept
from advanced_rag.upgrade.models import Generator, Verifier
from advanced_rag.upgrade.artifacts import ROOT, LOCAL, REPORTS, read, save, fingerprint, preserved
from advanced_rag.upgrade.evaluate import cluster_interval

def quality(rows, results, agreement, support):
    values = [accept(p, agreement, support) for p in results]
    em_f1 = [answer_scores(p["final_answer"], row["answers"]) for row, p in zip(rows, values)]
    positive = [i for i, r in enumerate(rows) if r["answers"]]
    negative = [i for i, r in enumerate(rows) if not r["answers"]]
    answered = [i for i, p in enumerate(values) if p["final_answer"]]
    blank = [i for i in range(len(rows)) if i not in answered]
    avg = lambda v: float(np.mean(v)) if len(v) else None
    return {"n": len(rows), "em": avg([v[0] for v in em_f1]), "f1": avg([v[1] for v in em_f1]),
            "coverage": len(answered) / len(rows), "answerable_f1": avg([em_f1[i][1] for i in positive]),
            "unanswerable_recall": avg([not values[i]["final_answer"] for i in negative]),
            "abstention_precision": avg([not rows[i]["answers"] for i in blank]),
            "selective_risk": avg([1 - em_f1[i][0] for i in answered]),
            "qa_checked_fraction": avg([p["qa_checked"] for p in results]),
            "mean_additional_validation_ms": avg([p["validation_ms"] for p in results]),
            "qa_output_tokens": avg([p["qa_output_tokens"] for p in results])}, [v[1] for v in em_f1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    preserved()
    qa = Generator("flan-base", device="cpu")
    verifier = Verifier()
    identity = fingerprint({"source": hashlib.sha256((ROOT / "src/advanced_rag/answer_support.py").read_bytes()).hexdigest(),
                            "script": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "models": read(ROOT / "models.upgrade.lock.json"),
                            "generation_identity": read(REPORTS / "experiment-identity.json")["fingerprint"], "torch": torch.__version__})
    def replay(name, split, rows):
        label = "selected" if split == "development" else "selected_raw"
        manifest = read(LOCAL / "manifests" / f"{label}-{name}-{split}.json")
        results, pointers = [], []
        for row, item in zip(rows, manifest["rows"]):
            assert row["id"] == item["id"]
            path = LOCAL / "answer-support" / identity / f"{name}-{row['id']}.json"
            if path.exists():
                result = read(path)
            else:
                result = check(row["question"], read(ROOT / item["path"])["result"], qa, verifier)
                save(path, result)
            results.append(result)
            pointers.append({"id": row["id"], "path": path.relative_to(ROOT).as_posix()})
        save(LOCAL / "answer-support" / f"{name}-{split}-manifest.json", {"identity": identity, "rows": pointers})
        return results
    development, selection, summaries = {}, {}, {}
    for name in ("hotpot", "squad"):
        dev, test, _ = load(name, 64, 320)
        development[name] = replay(name, "development", dev)
        tune_ids = set(read(REPORTS / "selection.json")["thresholds"][name]["threshold_ids"])
        indices = [i for i, row in enumerate(dev) if row["id"] in tune_ids]
        trials = []
        for agreement in (.5, .75, 1.):
            for support in (0., .1, .5):
                scores, _ = quality([dev[i] for i in indices], [development[name][i] for i in indices], agreement, support)
                trials.append({"agreement": agreement, "support": support, **scores})
        winner = max(trials, key=lambda r:r["f1"])
        selection[name] = {"agreement": winner["agreement"], "support": winner["support"], "trials": trials,
                           "threshold_ids": sorted(tune_ids), "criterion": "Development threshold-group answer F1"}
    selected_record = {"identity": identity, "selection": selection,
         "scope": "Exploratory question-conditioned validation follow-up; thresholds selected only on development groups",
         "source_sha256": hashlib.sha256((ROOT / "src/advanced_rag/answer_support.py").read_bytes()).hexdigest()}
    selected_path = REPORTS / "answer-support-selection.json"
    if not selected_path.exists() or read(selected_path) != selected_record:
        save(selected_path, selected_record)
    if args.development_only:
        print({name: {k:v[k] for k in ('agreement','support')} for name,v in selection.items()})
        return
    for name in ("hotpot", "squad"):
        dev, test, _ = load(name, 64, 320)
        cfg = selection[name]
        summaries[name] = {}
        for split, rows in (("fresh", test[64:]), ("original", test[:64])):
            results = replay(name, split, rows)
            scores, f1 = quality(rows, results, cfg["agreement"], cfg["support"])
            reference = read(LOCAL / "scores" / f"selected_raw-{name}-{split}.json")
            scores["paired_f1_vs_original_structured_parser"] = cluster_interval([p["f1"] for p in reference], f1, [r["group"] for r in rows])
            summaries[name][split] = scores
            save(REPORTS / "answer-support-results.json", {"identity": identity, "device": "cpu", "results": summaries,
                 "latency_scope": "Additional verification replay on retained generations; no new end-to-end latency claim",
                 "limitations": "QA-model agreement and NLI remain model judgments and can share entity or relevance errors"})
    preserved()

if __name__ == "__main__":
    main()

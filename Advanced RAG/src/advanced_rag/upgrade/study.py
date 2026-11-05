"""Run controlled stages with cached predictions and development-only choices."""
import gc
import time
import threading
from dataclasses import replace

import numpy as np
import psutil
import torch

from ..data import load, key
from ..pipeline import answer_scores, tokens
from .artifacts import ROOT, LOCAL, REPORTS, fingerprint, sources, read, save, preserved
from .engine import Engine
from .evaluate import aggregate, per_question, cluster_interval, calibration
from .models import Generator, Encoder, Reranker, Verifier
from .retrieval import Index

class Resources:
    def __init__(self):
        self.stop = threading.Event()
        self.rss = 0
        self.windows_peak = 0
        self.samples = 0
        def observe():
            while not self.stop.is_set():
                memory = psutil.Process().memory_info()
                self.rss = max(self.rss, memory.rss)
                self.windows_peak = max(self.windows_peak, getattr(memory, "peak_wset", memory.rss))
                self.samples += 1
                self.stop.wait(.05)
        self.thread = threading.Thread(target=observe, daemon=True)
        self.thread.start()

    def finish(self):
        self.stop.set()
        self.thread.join()
        return {"sampled_peak_rss_bytes": self.rss, "windows_process_peak_bytes": self.windows_peak,
                "sample_interval_ms": 50, "samples": self.samples,
                "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
                "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
                "scope": "Whole sequential study process; peaks are not isolated per method"}

class Study:
    def __init__(self):
        torch.set_num_threads(4)
        torch.manual_seed(42)
        self.config = read(ROOT / "configs/upgrade-study.json")
        self.identity = fingerprint({"sources": sources(), "models": read(ROOT / "models.upgrade.lock.json"),
                                     "original_models": read(ROOT / "models.lock.json"), "torch": torch.__version__, "config": self.config})
        self.encoders, self.rerankers, self.indexes = {}, {}, {}
        self.generator = None
        self.verifier = Verifier()
        self.loads = {"verifier": self.verifier.load_ms}
        self.rows = {}
        self.metrics, self.selections, self.intervals = {}, {}, {}
        self.resources = Resources()
        for name in ("hotpot", "squad"):
            dev, all_test, info = load(name, self.config["development_questions"], self.config["fresh_heldout_questions"] + 64)
            self.rows[name] = {"development": dev, "fresh": all_test[64:], "original": all_test[:64]}
            info.update(fresh_ids=[r["id"] for r in all_test[64:]], excluded_original_ids=[r["id"] for r in all_test[:64]])
            save(REPORTS / f"{name}-sampling.json", info)
        save(REPORTS / "experiment-identity.json", {"fingerprint": self.identity, "source_sha256": sources(), "config": self.config,
                                                    "torch": torch.__version__, "cuda": torch.cuda.is_available(), "cuda_version": torch.version.cuda})

    def adapters(self, cfg):
        alias = cfg["generator"]
        if self.generator is None or self.generator.alias != alias:
            if self.generator:
                self.generator.close()
            self.generator = Generator(alias)
            self.loads[alias] = self.generator.load_ms
        encoder = cfg.get("encoder", "minilm")
        if encoder not in self.encoders:
            self.encoders[encoder] = Encoder(encoder)
            self.loads[encoder] = self.encoders[encoder].load_ms
        reranker = cfg.get("reranker")
        if reranker and reranker not in self.rerankers:
            self.rerankers[reranker] = Reranker(reranker)
            self.loads[reranker] = self.rerankers[reranker].load_ms

    def run(self, name, split, rows, cfg, label):
        cfg = {k: v for k, v in cfg.items() if k != "name"}
        for setting, value in {"format": "grounded", "encoder": "minilm", "retrieval": "hybrid", "budget": 768,
                               "diversity": True, "reverse": False, "reasoning": "none", "threshold": 0.}.items():
            cfg.setdefault(setting, value)
        if cfg["reasoning"] != "corrective":
            cfg.pop("retry_threshold", None)
        self.adapters(cfg)
        results = []
        manifest = []
        for i, row in enumerate(rows):
            cache = fingerprint({"study": self.identity, "row": row, "configuration": cfg})
            path = LOCAL / "predictions" / self.identity / f"{cache}.json"
            if path.exists():
                result = read(path)["result"]
            else:
                encoder = cfg.get("encoder", "minilm")
                index_key = (encoder, fingerprint(row["docs"]))
                if index_key not in self.indexes:
                    self.indexes[index_key] = Index(row["docs"], self.encoders[encoder])
                index = self.indexes[index_key]
                engine = Engine(index, self.generator, self.verifier, self.rerankers.get(cfg.get("reranker")))
                result = engine.answer(row["question"], cfg)
                result["indexing"] = {"document_encoding_ms": index.encoding_ms, "index_ms": index.index_ms, "vector_bytes": index.vector_bytes}
                save(path, {"id": row["id"], "configuration": cfg, "result": result})
            results.append(result)
            manifest.append({"id": row["id"], "path": path.relative_to(ROOT).as_posix()})
            if (i + 1) % 32 == 0:
                print(label, name, split, i + 1, len(rows), flush=True)
        save(LOCAL / "manifests" / f"{label}-{name}-{split}.json", {"identity": self.identity, "configuration": cfg, "rows": manifest})
        values = [per_question(row, result) for row, result in zip(rows, results)]
        summary = aggregate(values)
        self.metrics.setdefault(split, {}).setdefault(name, {})[label] = summary
        save(REPORTS / "metrics.json", self.metrics)
        save(LOCAL / "scores" / f"{label}-{name}-{split}.json", values)
        return results, values

    def pooled_dev(self, cfg, label, count):
        pooled = []
        for name, splits in self.rows.items():
            _, values = self.run(name, "development", splits["development"][:count], cfg, label)
            pooled.extend(values)
        return aggregate(pooled)

    def main(self):
        preserved()
        candidates = []
        fixed = {"encoder": "minilm", "retrieval": "hybrid", "budget": 384, "diversity": True}
        for alias in self.config["generator_candidates"]:
            cfg = {**fixed, "generator": alias, "format": "plain"}
            values = self.pooled_dev(cfg, "generator_plain_" + alias, self.config["generator_selection_questions_per_dataset"])
            candidates.append({"generator": alias, **values})
        self.selections["plain_generators"] = candidates
        structured = []
        for alias in ("qwen06", "qwen17"):
            cfg = {**fixed, "generator": alias, "format": "grounded"}
            values = self.pooled_dev(cfg, "generator_grounded_" + alias, self.config["generator_selection_questions_per_dataset"])
            structured.append({"generator": alias, **values})
        winner = max(structured, key=lambda v: (v["answerable_f1"], v["f1"]))["generator"]
        self.selections["structured_generators"] = structured
        self.selections["selected_generator"] = winner
        self.selections["generator_objective"] = "Pooled development answerable F1, then overall F1; avoids choosing a citation generator primarily for abstaining"
        save(REPORTS / "selection.json", self.selections)
        retrieval = []
        for cfg in self.config["retrieval_candidates"]:
            values = self.pooled_dev({**cfg, "generator": winner, "diversity": True}, "retrieval_" + cfg["name"], self.config["retrieval_selection_questions_per_dataset"])
            retrieval.append({"configuration": cfg, **values})
        chosen = max(retrieval, key=lambda v: v["f1"])["configuration"]
        self.selections["retrieval"] = retrieval
        self.selections["selected_retrieval"] = chosen
        contexts = []
        for budget, diversity, reverse in ((384, True, False), (768, True, False), (1280, True, False), (768, False, False), (768, True, True)):
            cfg = {**chosen, "generator": winner, "budget": budget, "diversity": diversity, "reverse": reverse}
            label = f"context_{budget}_{int(diversity)}_{int(reverse)}"
            values = self.pooled_dev(cfg, label, self.config["retrieval_selection_questions_per_dataset"])
            contexts.append({"configuration": cfg, "label": label, **values})
        chosen_context = max(contexts, key=lambda v: v["f1"])["configuration"]
        self.selections["contexts"] = contexts
        self.selections["selected_configuration"] = chosen_context
        # Freeze model, retrieval, context, and threshold choices before fresh evaluation.
        self.selections["thresholds"] = {}
        for name, splits in self.rows.items():
            results, values = self.run(name, "development", splits["development"], chosen_context, "selected")
            groups = sorted({r["group"] for r in splits["development"]}, key=key)
            fit_groups = set(groups[:max(1, len(groups) // 2)])
            fit = [i for i, row in enumerate(splits["development"]) if row["group"] in fit_groups]
            tune = [i for i in range(len(results)) if i not in fit]
            if not tune:
                raise ValueError("Calibration requires separate development groups")
            trials = []
            for threshold in self.config["threshold_candidates"]:
                trial = aggregate([per_question(splits["development"][i], results[i], threshold) for i in tune])
                trials.append({"threshold": threshold, **trial})
            selected = max(trials, key=lambda v: v["f1"])["threshold"]
            lexical_trials = []
            for threshold in self.config["threshold_candidates"]:
                scores = []
                for i in tune:
                    p = dict(results[i], support=overlap(results[i]))
                    scores.append(per_question(splits["development"][i], p, threshold))
                lexical_trials.append({"threshold": threshold, **aggregate(scores)})
            lexical = max(lexical_trials, key=lambda v: v["f1"])["threshold"]
            from sklearn.linear_model import LogisticRegression
            x = [[results[i]["support"]] for i in fit]
            y = [values[i]["raw_em"] for i in fit]
            if len(set(y)) > 1:
                estimator = LogisticRegression(random_state=42).fit(x, y)
                coefficients = {"coefficient": float(estimator.coef_[0, 0]), "intercept": float(estimator.intercept_[0])}
            else:
                coefficients = {"coefficient": 0., "intercept": float(np.log((sum(y) + 1) / (len(y) - sum(y) + 1)))}
            self.selections["thresholds"][name] = {"selected": selected, "trials": trials, "lexical_selected": lexical,
                                                    "lexical_trials": lexical_trials, "calibrator": coefficients,
                                                    "fit_ids": [splits['development'][i]['id'] for i in fit],
                                                    "threshold_ids": [splits['development'][i]['id'] for i in tune]}
        save(REPORTS / "selection.json", self.selections)
        self.evaluate_frozen(chosen_context)
        save(REPORTS / "resources.json", {**self.resources.finish(), "model_loading_ms": self.loads,
                                           "verifier_pairs": self.verifier.total_pairs, "verifier_truncated_pairs": self.verifier.truncated_pairs})
        preserved()

    def evaluate_frozen(self, chosen):
        for name, splits in self.rows.items():
            results, raw_values = self.run(name, "fresh", splits["fresh"], chosen, "selected_raw")
            selection = self.selections["thresholds"][name]
            values = [per_question(r, p, selection["selected"]) for r, p in zip(splits["fresh"], results)]
            lexical_values = [per_question(r, dict(p, support=overlap(p)), selection["lexical_selected"]) for r, p in zip(splits["fresh"], results)]
            for label, scores in (("selected_verified", values), ("selected_lexical", lexical_values)):
                self.metrics["fresh"][name][label] = aggregate(scores)
                save(LOCAL / "scores" / f"{label}-{name}-fresh.json", scores)
            c = selection["calibrator"]
            probabilities = [1 / (1 + np.exp(-np.clip(c["coefficient"] * p["support"] + c["intercept"], -40, 40))) for p in results]
            save(REPORTS / f"{name}-calibration.json", {"raw_nli": calibration([p['support'] for p in results], [v['raw_em'] for v in raw_values]),
                                                       "platt": calibration(probabilities, [v['raw_em'] for v in raw_values]), "threshold": selection["selected"]})
            curve = [{"threshold": t, **aggregate([per_question(r, p, t) for r, p in zip(splits["fresh"], results)])} for t in self.config["threshold_candidates"]]
            save(REPORTS / f"{name}-risk-coverage.json", {"selection": "Development threshold fixed before held-out curves; no held-out tuning", "curve": curve})
            for label, cfg in (("balanced", {"generator": "qwen06", "encoder": "bge-small", "retrieval": "hybrid", "budget": 768, "diversity": True}),
                               ("strongest", {**chosen, "generator": "qwen17"})):
                self.run(name, "fresh", splits["fresh"], cfg, label)
            self.run(name, "original", splits["original"], chosen, "selected_raw")
        # Controlled comparisons use an explicitly smaller fixed subset.
        count = self.config["ablation_questions_per_dataset"]
        for alias in self.config["generator_candidates"]:
            cfg = {"generator": alias, "encoder": "minilm", "retrieval": "hybrid", "budget": 384, "diversity": True, "format": "plain"}
            for name, splits in self.rows.items():
                self.run(name, "controlled", splits["fresh"][:count], cfg, "generator_plain_" + alias)
        for cfg in self.config["retrieval_candidates"]:
            for name, splits in self.rows.items():
                self.run(name, "controlled", splits["fresh"][:count], {**cfg, "generator": chosen["generator"], "diversity": True}, "retrieval_" + cfg["name"])
        for entry in self.selections["contexts"]:
            for name, splits in self.rows.items():
                self.run(name, "controlled", splits["fresh"][:count], entry["configuration"], entry["label"])
        for reasoning in ("rewrite", "multistep", "corrective"):
            for name, splits in self.rows.items():
                cfg = {**chosen, "reasoning": reasoning, "retry_threshold": max(.5, self.selections["thresholds"][name]["selected"])}
                self.run(name, "controlled", splits["fresh"][:count], cfg, reasoning)
        for name, splits in self.rows.items():
            baseline = read(LOCAL / "scores" / f"selected_raw-{name}-fresh.json")[:count]
            comparisons = {}
            for label in self.metrics["controlled"][name]:
                values = read(LOCAL / "scores" / f"{label}-{name}-controlled.json")
                comparisons[label] = cluster_interval([r['f1'] for r in baseline], [r['f1'] for r in values], [r['group'] for r in values])
            self.intervals[name] = comparisons
        save(REPORTS / "paired-intervals.json", self.intervals)
        save(REPORTS / "metrics.json", self.metrics)

def overlap(result):
    answer = set(tokens(result["answer"]))
    return max((len(answer & set(tokens(e["text"]))) / max(1, len(answer)) for e in result["evidence"].values()), default=0.) if answer else 0.

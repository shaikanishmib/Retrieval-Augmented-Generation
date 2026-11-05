"""Answer against local evidence with explicit citations and support flags."""
import argparse
import json
from advanced_rag.upgrade.artifacts import ROOT, REPORTS, read
from advanced_rag.upgrade.engine import Engine
from advanced_rag.upgrade.models import Generator, Encoder, Reranker, Verifier
from advanced_rag.upgrade.retrieval import Index

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--documents", required=True, help="Local JSON mapping source IDs to document text")
parser.add_argument("--question", required=True)
parser.add_argument("--calibration", choices=["hotpot", "squad"], default="squad")
parser.add_argument("--threshold", type=float, help="Override the NLI support cutoff for the selected verification path")
parser.add_argument("--verification", choices=["selected", "claims", "question"], default="selected")
args = parser.parse_args()
documents = read(args.documents)
if not isinstance(documents, dict) or not documents or not all(isinstance(k, str) and isinstance(v, str) for k, v in documents.items()):
    raise ValueError("Documents must be a nonempty mapping of source IDs to text")
selection = read(REPORTS / "selection.json")
cfg = dict(selection["selected_configuration"])
cfg["threshold"] = args.threshold if args.threshold is not None else selection["thresholds"][args.calibration]["selected"]
generator = Generator(cfg["generator"])
index = Index(documents, Encoder(cfg["encoder"]))
reranker = Reranker(cfg["reranker"]) if cfg.get("reranker") else None
result = Engine(index, generator, Verifier(), reranker).answer(args.question, cfg)
adoption_path = REPORTS / "answer-support-adoption.json"
decision = read(adoption_path)["decisions"][args.calibration] if adoption_path.exists() else {"enabled": False}
if args.verification == "question" or (args.verification == "selected" and decision["enabled"]):
    from advanced_rag.answer_support import check, accept
    validation = read(REPORTS / "answer-support-selection.json")["selection"][args.calibration]
    qa = Generator("flan-base", device="cpu")
    cpu_verifier = Verifier()
    cpu_verifier.model.to("cpu")
    cpu_verifier.device = "cpu"
    checked = check(args.question, result, qa, cpu_verifier)
    support_cutoff = args.threshold if args.threshold is not None else validation["support"]
    result.update(accept(checked, validation["agreement"], support_cutoff))
    result["timings"]["question_validation_ms"] = checked["validation_ms"]
print(json.dumps({"answer": result["final_answer"], "flag": result["flag"], "support_score": result["support"],
                  "citations": [{"id": sid, **result["evidence"][sid]} for sid in result["final_citations"]],
                  "claims": result["claims"] if result["accepted"] else [], "timings": result["timings"]}, indent=2, ensure_ascii=False))

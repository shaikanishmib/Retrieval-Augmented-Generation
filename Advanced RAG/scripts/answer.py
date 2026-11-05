"""Answer a question against a local JSON mapping of source IDs to text."""
import argparse
import json
from pathlib import Path
from advanced_rag.pipeline import Models, Pipeline

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--documents", type=Path, required=True)
parser.add_argument("--question", required=True)
parser.add_argument("--method", choices=["bm25", "dense", "hybrid", "rerank", "multihop", "corrective"], default="hybrid")
parser.add_argument("--threshold", type=float, default=1.)
args = parser.parse_args()
documents = json.loads(args.documents.read_text(encoding="utf-8"))
if not isinstance(documents, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in documents.items()):
    raise ValueError("Documents must map string source IDs to text")
pipeline = Pipeline(Models.load(ROOT), documents)
if args.method == "corrective":
    initial = pipeline.run(args.question, "hybrid")
    result = initial
    if not initial["answer"] or initial["support_overlap"] < .5:
        result = pipeline.run(args.question, "corrective", threshold=args.threshold)
        for field in ("total_ms", "input_tokens", "output_tokens"):
            result[field] += initial[field]
else:
    result = pipeline.run(args.question, args.method, threshold=args.threshold)
print(json.dumps(result, indent=2, ensure_ascii=False))

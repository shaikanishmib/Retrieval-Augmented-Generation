"""Compare accuracy and timing across two completed retrieval passes."""

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="Directory containing the earlier dataset reports and environment.json.")
    args = parser.parse_args()
    current = ROOT / "reports/research"
    first_env = json.loads((args.reference / "environment.json").read_text())
    current_env = json.loads((current / "environment.json").read_text())
    if first_env["source_sha256"] != current_env["source_sha256"]:
        raise ValueError("The recorded retrieval sources differ between passes")
    rows = []
    for name in ("scifact", "fiqa"):
        first = json.loads((args.reference / f"{name}.json").read_text())
        latest = json.loads((current / f"{name}.json").read_text())
        if set(first["runs"]) != set(latest["runs"]):
            raise ValueError("The experiment matrices differ between passes")
        for method in first["runs"]:
            for label, result in (("initial", first["runs"][method]), ("verified", latest["runs"][method])):
                rows.append({"dataset": name, "method": method, "pass": label,
                             **{k: result[k] for k in ("ndcg@10", "p50_ms", "p95_ms", "query_count")}})
    path = current / "latency-repeatability.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} aggregate rows to {path}")


if __name__ == "__main__":
    main()

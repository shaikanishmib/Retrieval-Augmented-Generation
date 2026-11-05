"""Summarize completed experiment stages without loading models."""
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--all", action="store_true")
args = parser.parse_args()
path = ROOT / "reports/upgrade/metrics.json"
if path.exists():
    report = json.loads(path.read_text())
    for split, datasets in report.items():
        for name, methods in datasets.items():
            shown = list(methods.items()) if args.all else list(methods.items())[-4:]
            print(split, name, 'completed configurations:', len(methods), {m: {"n": r["n"], "f1": round(r["f1"], 4), "coverage": round(r["coverage"], 4), "p50_ms": round(r["p50_ms"], 1)} for m, r in shown})
selection = ROOT / "reports/upgrade/selection.json"
if selection.exists():
    values = json.loads(selection.read_text())
    print("Selected generator:", values.get("selected_generator"))
    print("Selected configuration:", values.get("selected_configuration"))
    print("Thresholds:", {n: r["selected"] for n, r in values.get("thresholds", {}).items()})
identity_path = ROOT / "reports/upgrade/experiment-identity.json"
if identity_path.exists():
    identity = json.loads(identity_path.read_text())["fingerprint"]
    files = list((ROOT / "data/upgrade/predictions" / identity).glob("*.json"))
    print('Completed prediction artifacts:', len(files))
    if files:
        latest = max(files, key=lambda p:p.stat().st_mtime)
        record = json.loads(latest.read_text(encoding='utf-8'))
        print('Latest configuration:', record['configuration'])

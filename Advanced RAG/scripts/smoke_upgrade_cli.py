"""Exercise the CLI and its threshold override with synthetic local evidence."""
import json
import subprocess
import sys
from advanced_rag.upgrade.artifacts import ROOT, LOCAL, REPORTS, save

fixture = LOCAL / "cli-smoke-documents.json"
save(fixture, {"reference": "The museum was not founded by Ada. It was founded by Bea."})
results = []
for threshold, expected in ((None, "Bea"), (1.1, "")):
    command = [sys.executable, str(ROOT / "scripts/answer_upgrade.py"), "--documents", str(fixture),
               "--question", "Who founded the museum?", "--calibration", "squad"]
    if threshold is not None:
        command += ["--threshold", str(threshold)]
    completed = subprocess.run(command, text=True, capture_output=True, check=True)
    result = json.loads(completed.stdout)
    results.append({"threshold": threshold, "expected": expected, "result": result, "passed": result["answer"] == expected})
save(REPORTS / "cli-smoke.json", {"scope": "Two real CLI invocations on synthetic evidence, separate from benchmark totals", "results": results})
assert all(r["passed"] for r in results), "Inspect the recorded CLI smoke failures"
print("Both CLI invocations returned the expected answer or forced abstention")

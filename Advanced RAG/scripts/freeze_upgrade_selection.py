"""Record the saved selection hash and its order relative to fresh predictions."""
import hashlib
from datetime import datetime, timezone
from advanced_rag.data import load
from advanced_rag.upgrade.artifacts import LOCAL, REPORTS, read, save

selection = REPORTS / "selection.json"
values = read(selection)
assert set(values["thresholds"]) == {"hotpot", "squad"}
identity = read(REPORTS / "experiment-identity.json")["fingerprint"]
fresh_ids = {row["id"] for name in ("hotpot", "squad") for row in load(name, 64, 320)[1][64:]}
times = []
for path in (LOCAL / "predictions" / identity).glob("*.json"):
    if read(path)["id"] in fresh_ids:
        times.append(path.stat().st_mtime)
assert times and selection.stat().st_mtime < min(times)
save(REPORTS / "selection-freeze.json", {"selection_sha256": hashlib.sha256(selection.read_bytes()).hexdigest(),
     "selection_saved_utc": datetime.fromtimestamp(selection.stat().st_mtime, timezone.utc).isoformat(),
     "first_fresh_prediction_utc": datetime.fromtimestamp(min(times), timezone.utc).isoformat(),
     "selection_precedes_fresh_predictions": True, "study_fingerprint": identity})
print("Selection was saved before the first fresh prediction")

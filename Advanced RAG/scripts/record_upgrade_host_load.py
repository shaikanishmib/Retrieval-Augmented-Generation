"""Record a host-load snapshot separately from benchmark resource measurements."""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import psutil

root = Path(__file__).resolve().parents[1]
battery = psutil.sensors_battery()
report = {
    "observed_utc": datetime.now(timezone.utc).isoformat(),
    "scope": "One host snapshot during the study; not an isolated benchmark or a causal explanation for latency",
    "memory": dict(psutil.virtual_memory()._asdict()),
    "cpu_percent_one_second": psutil.cpu_percent(interval=1),
    "battery": {"percent": battery.percent, "power_plugged": battery.power_plugged} if battery else None,
    "gpu_snapshot": subprocess.check_output(["nvidia-smi", "-q", "-d", "PERFORMANCE,CLOCK,POWER"], text=True),
}
destination = root / "reports/upgrade/host-load.json"
if destination.exists():
    raise FileExistsError("Keep the recorded host snapshot; use a new filename for another observation")
destination.write_bytes(json.dumps(report, indent=2).encode())
print("Recorded host-load snapshot separately from model benchmarks")

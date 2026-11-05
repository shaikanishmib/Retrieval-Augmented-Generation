"""Record observed memory use for a running retrieval experiment."""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    args = parser.parse_args()
    process = psutil.Process(args.pid)
    if Path(process.cwd()).resolve() != ROOT.resolve() or not any("run_research.py" in arg for arg in process.cmdline()):
        raise ValueError("The PID must belong to this project's research runner")
    started = datetime.now(timezone.utc).isoformat()
    samples, peak_rss, peak_counter = 0, 0, 0
    total_memory = psutil.virtual_memory().total
    while True:
        try:
            memory = process.memory_info()
            peak_rss = max(peak_rss, memory.rss)
            peak_counter = max(peak_counter, getattr(memory, "peak_wset", 0))
            samples += 1
        except psutil.NoSuchProcess:
            break
        time.sleep(1)
    result = {"started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
              "samples": samples, "sample_interval_seconds": 1, "host_physical_memory_bytes": total_memory,
              "maximum_sampled_rss_bytes": peak_rss, "maximum_observed_windows_peak_wset_bytes": peak_counter or None,
              "scope": "Whole research process, sampled from observer startup. The Windows peak counter includes earlier process lifetime; a peak after the last sample may be missed. These are not per-index memory costs.",
              "observer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    path = ROOT / "reports/research/memory-observation.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

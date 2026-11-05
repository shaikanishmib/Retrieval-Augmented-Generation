"""Acquire the public training release because prior work exhausted dev article groups."""
import urllib.request
import time
from agentic_rag.artifacts import ROOT,read,write,sha

if __name__ == "__main__":
    url="https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
    target=ROOT / "data/raw/squad-train.json"
    if not target.exists():
        with urllib.request.urlopen(url,timeout=120) as response:
            payload=response.read()
        target.write_bytes(payload)
    records=read(ROOT / "sources.lock.json")
    records=[r for r in records if r["name"]!="squad"]
    records.append(dict(name="squad",url=url,version="SQuAD 2.0 public training release",bytes=target.stat().st_size,
                        sha256=sha(target),license="CC BY-SA 4.0",acquired_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                        hash_provenance="SHA-256 measured at acquisition; not an independently supplied upstream checksum"))
    write(ROOT / "sources.lock.json",records)
    print(records[-1])

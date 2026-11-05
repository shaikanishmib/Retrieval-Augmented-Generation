"""Copy and verify pinned local data; reuse model files without writing to them."""
import shutil
import urllib.request
import argparse
from agentic_rag.artifacts import ROOT, REFERENCE, read, write, sha

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only",action="store_true")
    args=parser.parse_args()
    lock=ROOT / "sources.lock.json"
    sources = read(lock if lock.exists() else REFERENCE / "sources.lock.json")
    for record in sources:
        name = record["name"]
        src = REFERENCE / f"data/raw/{name}.json"
        dst = ROOT / ("data/raw/squad-train.json" if "train-v2.0" in record["url"] else f"data/raw/{name}.json")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            if args.verify_only:
                raise FileNotFoundError(dst)
            if src.exists() and sha(src)==record["sha256"]:
                shutil.copyfile(src, dst)
            else:
                with urllib.request.urlopen(record["url"],timeout=120) as response:
                    payload=response.read()
                if record["url"].endswith("parquet"):
                    parquet=dst.with_suffix(".parquet")
                    parquet.write_bytes(payload)
                    assert sha(parquet)==record["download_sha256"]
                    import pyarrow.parquet as pq
                    rows=pq.read_table(parquet).to_pylist()
                    converted=[]
                    for r in rows:
                        r["_id"]=r.pop("id")
                        r["context"]=list(zip(r["context"]["title"],r["context"]["sentences"]))
                        r["supporting_facts"]=list(zip(r["supporting_facts"]["title"],r["supporting_facts"]["sent_id"]))
                        converted.append(r)
                    # Original JSON serializer is part of the acquisition contract.
                    import json
                    dst.write_text(json.dumps(converted,ensure_ascii=False),encoding="utf-8")
                else:
                    dst.write_bytes(payload)
        assert sha(dst) == record["sha256"]
    write(ROOT / "sources.lock.json", sources)
    model_lock=ROOT / "models.lock.json"
    models = read(model_lock if model_lock.exists() else REFERENCE / "models.upgrade.lock.json")
    for alias, record in models.items():
        location = REFERENCE / "data/upgrade/models" / alias
        if not all((location / name).is_file() for name in record["files"]):
            location=ROOT / "data/models" / alias
            if not args.verify_only:
                location.mkdir(parents=True,exist_ok=True)
        for name, info in record["files"].items():
            target=location / name
            if not target.exists():
                if args.verify_only:
                    raise FileNotFoundError(target)
                target.parent.mkdir(parents=True,exist_ok=True)
                url=f'https://huggingface.co/{record["id"]}/resolve/{record["revision"]}/{name}'
                with urllib.request.urlopen(url,timeout=120) as response:
                    target.write_bytes(response.read())
            assert sha(location / name) == info["sha256"], (alias, name)
        record["local_path"] = str(location.relative_to(ROOT.parent))
    write(ROOT / "models.lock.json", models)
    print("Verified two dataset files and", len(models), "model snapshots. Prior projects opened read-only.")

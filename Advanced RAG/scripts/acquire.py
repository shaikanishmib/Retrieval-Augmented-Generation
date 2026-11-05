"""Acquire selected datasets and immutable local model snapshots."""
import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(ROOT / "data/hf")
os.environ["HF_HUB_DISABLE_XET"] = "1"

def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    specs = [("hotpot", "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/1908d6afbbead072334abe2965f91bd2709910ab/distractor/validation-00000-of-00001.parquet"),
             ("squad", "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json")]
    lock_path = ROOT / "sources.lock.json"
    previous = json.loads(lock_path.read_text()) if lock_path.exists() else []
    records = []
    for name, url in specs:
        path = ROOT / f"data/raw/{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            if args.verify_only:
                raise FileNotFoundError(path)
            partial = path.with_suffix(".parquet.part" if name == "hotpot" else ".part")
            subprocess.run(["curl.exe", "--fail", "--location", "--retry", "3", "--max-time", "600", url, "--output", str(partial)], check=True)
            if name == "hotpot":
                if digest(partial) != "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6":
                    raise ValueError("HotpotQA mirror differs from its published LFS hash")
                import pyarrow.parquet as pq
                rows = pq.read_table(partial).to_pylist()
                converted = []
                for row in rows:
                    row["_id"] = row.pop("id")
                    row["context"] = list(zip(row["context"]["title"], row["context"]["sentences"]))
                    row["supporting_facts"] = list(zip(row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"]))
                    converted.append(row)
                path.write_text(json.dumps(converted, ensure_ascii=False), encoding="utf-8")
                partial.replace(path.with_suffix(".parquet"))
            else:
                json.loads(partial.read_text(encoding="utf-8"))
                partial.replace(path)
        record = {"name": name, "url": url, "bytes": path.stat().st_size, "sha256": digest(path), "license": "CC BY-SA 4.0"}
        if name == "hotpot":
            parquet = path.with_suffix(".parquet")
            record["download_bytes"] = parquet.stat().st_size
            record["download_sha256"] = digest(parquet)
            if record["download_sha256"] != "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6":
                raise ValueError("HotpotQA Parquet hash differs from published LFS hash")
        prior = next((r for r in previous if r["name"] == name), None)
        if prior and any(prior[k] != record[k] for k in ("bytes", "sha256")):
            raise ValueError(f"Dataset differs from lock: {name}")
        record["acquired_utc"] = prior["acquired_utc"] if prior else datetime.now(timezone.utc).isoformat()
        records.append(record)
    lock_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    from huggingface_hub import HfApi, snapshot_download
    model_lock = ROOT / "models.lock.json"
    models = json.loads(model_lock.read_text()) if model_lock.exists() else []
    if not models:
        for role, name in [("generator", "google/flan-t5-small"), ("embedding", "sentence-transformers/all-MiniLM-L6-v2"), ("reranker", "cross-encoder/ms-marco-TinyBERT-L2-v2")]:
            models.append({"role": role, "id": name, "revision": HfApi().model_info(name).sha})
        model_lock.write_text(json.dumps(models, indent=2), encoding="utf-8")
    for spec in models:
        target = ROOT / "data/models" / spec["role"]
        if not args.verify_only:
            snapshot_download(spec["id"], revision=spec["revision"], local_dir=target,
                              allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors"],
                              ignore_patterns=["onnx/*", "openvino/*"])
        observed = {p.relative_to(target).as_posix(): digest(p) for p in target.rglob("*") if p.is_file() and ".cache" not in p.parts}
        if not observed or spec.get("files", observed) != observed:
            raise ValueError(f"Model snapshot differs from lock: {spec['id']}")
        spec["files"] = observed
        print(spec["role"], spec["revision"], len(observed), flush=True)
    model_lock.write_text(json.dumps(models, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()

"""Download only selected pinned upgrade snapshots with a storage guard."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = {
    "flan-base": ("google/flan-t5-base", "generator", "apache-2.0"),
    "qwen06": ("Qwen/Qwen3-0.6B", "generator", "apache-2.0"),
    "qwen17": ("Qwen/Qwen3-1.7B", "generator", "apache-2.0"),
    "bge-small": ("BAAI/bge-small-en-v1.5", "embedding", "mit"),
    "minilm-rerank": ("cross-encoder/ms-marco-MiniLM-L6-v2", "reranker", "apache-2.0"),
    "nli": ("cross-encoder/nli-deberta-v3-small", "verifier", "apache-2.0"),
}

def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", choices=CANDIDATES)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    lock = ROOT / "models.upgrade.lock.json"
    specs = json.loads(lock.read_text()) if lock.exists() else {}
    for alias in args.models:
        name, role, license_name = CANDIDATES[alias]
        if alias not in specs:
            info = HfApi().model_info(name, files_metadata=True)
            specs[alias] = {"id": name, "role": role, "revision": info.sha, "license": license_name,
                            "model_card": f"https://huggingface.co/{name}/blob/{info.sha}/README.md"}
            lock.write_text(json.dumps(specs, indent=2), encoding="utf-8")
        spec = specs[alias]
        target = ROOT / "data/upgrade/models" / alias
        if not args.verify_only:
            if shutil.disk_usage(ROOT).free < 15_000_000_000:
                raise OSError("Keep at least 15 GB free before acquiring another model")
            snapshot_download(name, revision=spec["revision"], local_dir=target,
                              allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors", "README.md", "LICENSE*"],
                              ignore_patterns=["onnx/*", "openvino/*", "onnx_O4/*"])
        observed = {p.relative_to(target).as_posix(): {"sha256": digest(p), "bytes": p.stat().st_size}
                    for p in target.rglob("*") if p.is_file() and ".cache" not in p.parts}
        if not observed or spec.get("files", observed) != observed:
            raise ValueError(f"Snapshot differs from lock: {alias}")
        spec["files"] = observed
        spec["bytes"] = sum(p["bytes"] for p in observed.values())
        lock.write_text(json.dumps(specs, indent=2), encoding="utf-8")
        print(alias, spec["revision"], spec["bytes"], flush=True)

if __name__ == "__main__":
    main()

"""Download immutable model snapshots into project-local storage."""

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(ROOT / "data/model-cache")
os.environ["HF_HUB_DISABLE_XET"] = "1"


def main():
    from huggingface_hub import HfApi, snapshot_download
    lock = ROOT / "models.lock.json"
    models = json.loads(lock.read_text()) if lock.exists() else []
    if not models:
        for role, name in [
            ("embedding", "sentence-transformers/all-MiniLM-L6-v2"),
            ("embedding", "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"),
            ("reranking", "cross-encoder/ms-marco-TinyBERT-L2-v2"),
        ]:
            info = HfApi().model_info(name)
            models.append({"role": role, "id": name, "revision": info.sha})
        lock.write_text(json.dumps(models, indent=2), encoding="utf-8")
    for model in models:
        target = ROOT / "data/models" / model["id"].split("/")[-1]
        info = HfApi().model_info(model["id"], revision=model["revision"])
        safe_weights = any(s.rfilename == "model.safetensors" for s in info.siblings)
        snapshot_download(model["id"], revision=model["revision"], local_dir=target,
                          allow_patterns=["*.json", "*.txt", "*.safetensors", "*.bin", "*.md"],
                          ignore_patterns=["onnx/*", "openvino/*", "rust_model.ot", "tf_model.h5", "flax_model.msgpack"] + (["pytorch_model.bin"] if safe_weights else []))
        observed = {p.relative_to(target).as_posix(): hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
                    for p in target.rglob("*") if p.is_file() and ".cache" not in p.parts}
        if model.get("files") and observed != model["files"]:
            raise ValueError(f"Model files differ from their pinned hashes: {model['id']}")
        model["files"] = observed
        print(f"Verified {model['id']} at {model['revision']}", flush=True)
    lock.write_text(json.dumps(models, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

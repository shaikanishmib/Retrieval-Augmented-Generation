import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_extract(archive: Path, destination: Path, name: str) -> None:
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            path = PurePosixPath(member.filename)
            if (path.is_absolute() or ".." in path.parts or "\\" in member.orig_filename
                    or ":" in member.filename or not path.parts or path.parts[0] != name
                    or (member.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError("Unsafe archive member")
        if sum(member.file_size for member in bundle.infolist()) > 200_000_000:
            raise ValueError("Archive exceeds extraction size limit")
        bundle.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire checksum-pinned dataset archives.")
    parser.add_argument("--verify-only", action="store_true", help="Verify local archives without downloading or extracting.")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "sources.lock.json").read_text(encoding="utf-8"))
    destination = ROOT / "data" / "raw"
    destination.mkdir(parents=True, exist_ok=True)
    for item in manifest:
        archive = destination / item["filename"]
        if not archive.exists():
            if args.verify_only:
                raise FileNotFoundError(archive)
            partial = archive.with_suffix(".part")
            try:
                if sys.platform == "win32":
                    subprocess.run(["curl.exe", "--fail", "--location", "--silent", "--show-error",
                                    "--proto", "=https", "--max-time", "900",
                                    item["url"], "--output", str(partial)], check=True)
                else:
                    with urllib.request.urlopen(item["url"], timeout=120) as response, partial.open("wb") as stream:
                        shutil.copyfileobj(response, stream)
                if partial.stat().st_size != item["bytes"] or sha256(partial) != item["sha256"]:
                    raise ValueError(f"Download differs from pin: {item['filename']}")
                partial.replace(archive)
            finally:
                partial.unlink(missing_ok=True)
        if archive.stat().st_size != item["bytes"] or sha256(archive) != item["sha256"]:
            raise ValueError(f"Archive differs from pin: {item['filename']}")
        if not args.verify_only and item["name"] in ("scifact", "fiqa"):
            target = destination / item["name"]
            if target.exists():
                raise FileExistsError(f"Move the existing extraction before reacquiring: {target}")
            with tempfile.TemporaryDirectory(dir=destination) as temporary:
                safe_extract(archive, Path(temporary), item["name"])
                (Path(temporary) / item["name"]).replace(target)
        print(f"Verified {item['filename']}: {item['bytes']} bytes, {item['sha256']}")


if __name__ == "__main__":
    main()

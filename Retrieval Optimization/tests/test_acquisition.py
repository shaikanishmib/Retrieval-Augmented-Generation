import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("acquisition", ROOT / "scripts/acquire.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("member", ["../escape", "/absolute", "scifact/../../escape", "scifact/evil\\escape", "scifact/C:escape", "wrong/file"])
def test_unsafe_extraction_rejected(tmp_path, member):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(member.replace("\\", "/"), "x")
    if "\\" in member:
        archive.write_bytes(archive.read_bytes().replace(member.replace("\\", "/").encode(), member.encode()))
    with pytest.raises(ValueError, match="Unsafe"):
        module.safe_extract(archive, tmp_path / "extracted", "scifact")
    assert not (tmp_path / "extracted").exists()


def test_symlink_rejected(tmp_path):
    archive=tmp_path/"link.zip"
    info=zipfile.ZipInfo("scifact/link")
    info.create_system=3
    info.external_attr=(0o120777 << 16)
    with zipfile.ZipFile(archive,"w") as bundle:
        bundle.writestr(info,"../outside")
    with pytest.raises(ValueError,match="Unsafe"):
        module.safe_extract(archive,tmp_path/"extracted","scifact")


def test_original_archives_and_extracted_bytes_match():
    for item in json.loads((ROOT/"sources.lock.json").read_text()):
        archive=ROOT/"data/raw"/item["filename"]
        if not archive.exists():
            pytest.skip("Acquire the checksum-pinned original archives")
        assert archive.stat().st_size==item["bytes"]
        assert module.sha256(archive)==item["sha256"]
        if item["name"] in {"scifact","fiqa"}:
            with zipfile.ZipFile(archive) as bundle:
                for member in bundle.infolist():
                    if not member.is_dir():
                        assert (ROOT/"data/raw"/member.filename).read_bytes()==bundle.read(member)

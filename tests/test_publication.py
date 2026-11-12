"""Release integrity checks use disposable local files and never contact GitHub."""
from pathlib import Path
import importlib.util
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("newslens_publisher", ROOT / "scripts" / "publish_release.py")
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


def bundle_fixture(tmp_path):
    run = tmp_path / "run"
    files = {"adapter/adapter_model.safetensors": b"verified-adapter",
        "protocol.json": b"{}", "execution_manifest.json": b"{}", "smoke/gate.json": b"{}",
        "smoke/requirements-gpu.lock.txt": b"torch==2.8.0", "training/summary.json": b"{}"}
    for name, content in files.items():
        path = run / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    bundle = tmp_path / "adapter.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
        archive.writestr("checksums.json", "{}")
        archive.writestr("BASE_MODEL.txt", "fixed-base-revision")
    return bundle, run


def test_release_bundle_matches_evaluated_adapter(tmp_path):
    bundle, run = bundle_fixture(tmp_path)
    publisher.check_bundle(bundle, run)
    (run / "adapter" / "adapter_model.safetensors").write_bytes(b"a-different-adapter")
    with pytest.raises(ValueError, match="differs from the verified run"):
        publisher.check_bundle(bundle, run)


@pytest.mark.parametrize("unexpected", ["model.safetensors", ".env"])
def test_release_rejects_unexpected_payloads(tmp_path, unexpected):
    bundle, run = bundle_fixture(tmp_path)
    with zipfile.ZipFile(bundle, "a") as archive:
        archive.writestr(unexpected, "unapproved-content")
    with pytest.raises(ValueError, match="unexpected or missing"):
        publisher.check_bundle(bundle, run)


def test_release_verifies_remote_digest(tmp_path):
    path = tmp_path / "asset.zip"
    path.write_bytes(b"measured-adapter")
    asset = {"state": "uploaded", "size": path.stat().st_size, "digest": "sha256:" + publisher.file_digest(path)}
    publisher.check_asset(asset, path)
    asset["digest"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="integrity mismatch"):
        publisher.check_asset(asset, path)

"""Release integrity checks use disposable local files and never contact GitHub."""
from pathlib import Path
import importlib.util
import json
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
        "smoke/requirements-gpu.lock.txt": b"torch==2.8.0", "training/summary.json": b"{}", "stage_sessions.jsonl": b""}
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


def test_session_timing_sums_each_session_once():
    from verify_results import session_summary
    rows = [{"session": "first", "stage": "train", "event": "start", "elapsed_seconds": 0},
        {"session": "first", "stage": "train", "event": "progress", "elapsed_seconds": 100},
        {"session": "first", "stage": "train", "event": "progress", "elapsed_seconds": 200},
        {"session": "resumed", "stage": "train", "event": "start", "elapsed_seconds": 0},
        {"session": "resumed", "stage": "train", "event": "completed", "elapsed_seconds": 300}]
    assert session_summary(rows)["train"] == {"recorded_active_seconds": 500, "sessions": 2, "completed": 1, "censored": 1}


def test_final_report_includes_resources_and_matrix_links(tmp_path):
    from newslens import FINAL_CONDITIONS
    from verify_results import enrich_report
    bundle, run = bundle_fixture(tmp_path)
    (run / "training" / "summary.json").write_text(json.dumps({"hardware": {
        "parameter_dtypes": {"torch.float16": 100, "torch.float32": 10}, "unsloth_precision_guard": True}}))
    (run / "stage_sessions.jsonl").write_text(json.dumps({"session": "full", "stage": "train", "event": "completed", "elapsed_seconds": 3600}) + "\n")
    (run / "predictions").mkdir()
    for condition in FINAL_CONDITIONS:
        (run / "predictions" / f"{condition}.jsonl").write_text('{"seconds": 2}\n')
    output = tmp_path / "report"
    output.mkdir()
    (output / "REPORT.md").write_text("Canonical training prompt, FP16 weights, response-only loss.\n")
    enrich_report(run, output)
    resources = json.loads((output / "resource_summary.json").read_text())
    assert resources["stages"]["train"]["recorded_active_seconds"] == 3600
    assert resources["inference"]["concise_zero"]["examples_per_second"] == 0.5
    assert resources["adapter_weights_bytes"] == len(b"verified-adapter")
    text = (output / "REPORT.md").read_text(encoding="utf-8")
    assert text.count("![") == 14
    assert "FP16 weights" not in text and "precision safeguards" in text

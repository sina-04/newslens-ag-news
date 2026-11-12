"""Verify a downloaded Colab run and assemble publication assets.

No GitHub writes occur here. A missing/incomplete experiment fails closed.
"""
from pathlib import Path
import argparse
import json
import math
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from newslens.common import file_digest, folder_digest, load_config, read_json, run_identity, write_json
from newslens.data import load_prepared
from newslens.reporting import report
from newslens import FINAL_CONDITIONS
from newslens.evaluation import validate_predictions
from newslens.prompts import parse_response


def verify_run(run, config, data):
    splits, _, dataset = load_prepared(data)
    identity = run_identity(config, dataset)
    gate = read_json(run / "smoke" / "gate.json")
    if not gate["passed"] or gate["identity"] != identity or gate["optimizer_steps"] != config["smoke_steps"]:
        raise ValueError("No matching completed smoke gate.")
    if gate.get("gradient_checks") != config["smoke_steps"]:
        raise ValueError("The gate did not verify every optimizer step's gradients.")
    for field in ("frozen_base_unchanged", "adapter_changed", "checkpoint_resume_verified", "reload_predictions_match"):
        if gate.get(field) is not True:
            raise ValueError(f"Required GPU verification missing: {field}")
    if gate["memory"]["estimated_peak_headroom_bytes"] < config["minimum_headroom_gib"] * 2**30:
        raise ValueError("GPU headroom requirement was not met.")
    if gate["reload_memory"]["estimated_peak_headroom_bytes"] < config["minimum_headroom_gib"] * 2**30:
        raise ValueError("Reload inference headroom requirement was not met.")
    training = read_json(run / "training" / "summary.json")
    if training["identity"] != identity or training["epochs"] < config["epochs"] - 0.001:
        raise ValueError("Incomplete or mismatched full training.")
    adapter = run / "adapter"
    adapter_config = read_json(adapter / "adapter_config.json")
    if adapter_config["base_model_name_or_path"] != config["model_id"] or adapter_config.get("revision") != config["model_revision"]:
        raise ValueError("Adapter does not reference the exact upstream base revision.")
    if not (adapter / "adapter_model.safetensors").is_file():
        raise ValueError("Adapter weights missing.")
    adapter_sha = folder_digest(adapter)
    metadata = read_json(run / "predictions" / "lora_concise_zero.manifest.json")
    if metadata["adapter_sha256"] != adapter_sha:
        raise ValueError("The adapter differs from the one evaluated.")
    for condition in FINAL_CONDITIONS:
        rows = validate_predictions(run / "predictions" / f"{condition}.jsonl", splits["test"], identity, condition)
        for row in rows:
            if row["prediction"] != parse_response(row["raw_response"], row["output_truncated"]):
                raise ValueError(f"Stored parsed label disagrees with raw response: {condition}/{row['id']}")
            if not 0 < row["prompt_tokens"] <= config["context_tokens"] - config["max_new_tokens"]:
                raise ValueError("Prediction prompt exceeds the frozen context budget.")
            if not 0 < row["generated_tokens"] <= config["max_new_tokens"]:
                raise ValueError("Prediction generation exceeds the frozen token budget.")
            if not math.isfinite(row["seconds"]) or row["seconds"] < 0:
                raise ValueError("Invalid measured generation time.")
    return identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "prepared")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "colab_t4.yaml")
    args = parser.parse_args()
    run = ROOT / "artifacts" / "verified-run"
    if run.exists():
        raise SystemExit("artifacts/verified-run already exists; use a new clean import directory explicitly.")
    run.mkdir(parents=True)
    with zipfile.ZipFile(args.bundle) as archive:
        for name in archive.namelist():
            if not (run / name).resolve().is_relative_to(run.resolve()):
                raise ValueError("Unsafe result bundle entry")
        archive.extractall(run)
    config = load_config(args.config)
    identity = verify_run(run, config, args.data_dir)
    # Regenerate all metrics/plots locally from the full saved prediction tables.
    report(config, args.data_dir, run, ROOT / "docs" / "results")
    predictions = ROOT / "docs" / "results" / "predictions"
    predictions.mkdir(exist_ok=True)
    for path in (run / "predictions").glob("*"):
        if path.is_file():
            shutil.copy2(path, predictions / path.name)
    release = ROOT / "artifacts" / "newslens-qwen35-4b-lora-v1.0.0.zip"
    with zipfile.ZipFile(release, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((run / "adapter").rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(run))
        for relative in ("protocol.json", "execution_manifest.json", "smoke/gate.json", "smoke/requirements-gpu.lock.txt", "training/summary.json"):
            archive.write(run / relative, relative)
        archive.writestr("checksums.json", json.dumps({str(p.relative_to(run)): file_digest(p) for p in sorted((run / "adapter").rglob("*")) if p.is_file()}, indent=2))
        archive.writestr("BASE_MODEL.txt", f"{config['model_id']}\nrevision={config['model_revision']}\nLoad this exact base separately; weights are not included.\n")
    write_json(release.with_suffix(".sha256.json"), {release.name: file_digest(release), "run_id": identity["run_id"]})
    print("Verified full experiment. Publication asset:", release)


if __name__ == "__main__":
    main()

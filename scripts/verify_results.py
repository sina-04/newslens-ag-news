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
from newslens.common import file_digest, folder_digest, load_config, read_json, read_rows, run_identity, write_json
from newslens.data import load_prepared
from newslens.reporting import report
from newslens import FINAL_CONDITIONS
from newslens.evaluation import validate_predictions
from newslens.prompts import parse_response


def session_summary(events):
    sessions = {}
    for event in events:
        key = event["session"]
        elapsed = event["elapsed_seconds"]
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Invalid session timing event.")
        prior = sessions.get(key)
        if prior and (prior["stage"] != event["stage"] or elapsed < prior["elapsed_seconds"]):
            raise ValueError("Inconsistent session timing events.")
        sessions[key] = {"stage": event["stage"], "elapsed_seconds": elapsed, "last_event": event["event"]}
    stages = {}
    for value in sessions.values():
        stage = stages.setdefault(value["stage"], {"recorded_active_seconds": 0, "sessions": 0, "completed": 0, "censored": 0})
        stage["recorded_active_seconds"] += value["elapsed_seconds"]
        stage["sessions"] += 1
        stage["completed"] += value["last_event"] == "completed"
        stage["censored"] += value["last_event"] not in ("completed", "interrupted", "failed")
    return stages


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
    timing = session_summary(read_rows(run / "stage_sessions.jsonl"))
    if timing.get("train", {}).get("completed", 0) < 1:
        raise ValueError("No completed full-training session is recorded.")
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


def enrich_report(run, output):
    training = read_json(run / "training" / "summary.json")
    stages = session_summary(read_rows(run / "stage_sessions.jsonl"))
    resources = {"stages": stages, "parameter_dtypes": training["hardware"]["parameter_dtypes"],
        "precision_guard": training["hardware"]["unsloth_precision_guard"],
        "adapter_weights_bytes": (run / "adapter" / "adapter_model.safetensors").stat().st_size,
        "adapter_folder_bytes": sum(p.stat().st_size for p in (run / "adapter").rglob("*") if p.is_file()), "inference": {}}
    table = []
    for condition in FINAL_CONDITIONS:
        rows = read_rows(run / "predictions" / f"{condition}.jsonl")
        seconds = sum(row["seconds"] for row in rows)
        throughput = len(rows) / seconds if seconds > 0 else 0
        resources["inference"][condition] = {"examples": len(rows), "generation_seconds": seconds, "examples_per_second": throughput}
        table.append(f"| {condition} | {seconds / 60:.2f} | {throughput:.3f} |")
    write_json(output / "resource_summary.json", resources)
    shutil.copy2(run / "stage_sessions.jsonl", output / "stage_sessions.jsonl")
    text = (output / "REPORT.md").read_text(encoding="utf-8")
    text = text.replace("Canonical training prompt, FP16 weights,", "Canonical training prompt, unquantized weights with Unsloth's Qwen3.5 precision safeguards,")
    text += f"\n## Measured timing and adapter size\n\nRecorded active training wall time: **{stages['train']['recorded_active_seconds'] / 3600:.2f} hours** across {stages['train']['sessions']} sessions. This includes loading, preparation, compilation, validation, checkpoint writing and any repeated work. It excludes gaps between Colab sessions. Censored sessions: {stages['train']['censored']}; for a runtime terminated without a final event, elapsed time ends at its last durable heartbeat, making the aggregate a lower bound.\n\n"
    text += f"Adapter weights: **{resources['adapter_weights_bytes'] / 2**20:.2f} MiB**; complete adapter/tokenizer folder: {resources['adapter_folder_bytes'] / 2**20:.2f} MiB. Actual parameter dtype counts: `{resources['parameter_dtypes']}`. Precision guard active: `{resources['precision_guard']}`.\n\n"
    text += "Generation times below sum the measured example-level calls across resumptions; throughput excludes model loading and prediction-file checkpoint writes. The full resource summary and session events are included with the report.\n\n| Condition | Generation minutes | Examples/second |\n| --- | ---: | ---: |\n" + "\n".join(table) + "\n\n## Confusion matrices\n"
    for condition in FINAL_CONDITIONS:
        text += f"\n### {condition}\n\n![Raw confusion matrix for {condition}]({condition}_confusion.png)\n\n![Row-normalized confusion matrix for {condition}]({condition}_confusion_normalized.png)\n"
    (output / "REPORT.md").write_text(text, encoding="utf-8")


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
    output = ROOT / "docs" / "results"
    report(config, args.data_dir, run, output)
    enrich_report(run, output)
    predictions = output / "predictions"
    predictions.mkdir(exist_ok=True)
    for path in (run / "predictions").glob("*"):
        if path.is_file():
            shutil.copy2(path, predictions / path.name)
    write_json(output / "checksums.json", {p.relative_to(output).as_posix(): file_digest(p) for p in sorted(output.rglob("*")) if p.is_file() and p != output / "checksums.json"})
    release = ROOT / "artifacts" / "newslens-qwen35-4b-lora-v1.0.0.zip"
    with zipfile.ZipFile(release, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((run / "adapter").rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(run))
        for relative in ("protocol.json", "execution_manifest.json", "smoke/gate.json", "smoke/requirements-gpu.lock.txt", "training/summary.json", "stage_sessions.jsonl"):
            archive.write(run / relative, relative)
        archive.writestr("checksums.json", json.dumps({p.relative_to(run).as_posix(): file_digest(p) for p in sorted((run / "adapter").rglob("*")) if p.is_file()}, indent=2))
        archive.writestr("BASE_MODEL.txt", f"{config['model_id']}\nrevision={config['model_revision']}\nLoad this exact base separately; weights are not included.\n")
    write_json(release.with_suffix(".sha256.json"), {release.name: file_digest(release), "run_id": identity["run_id"]})
    print("Verified full experiment. Publication asset:", release)


if __name__ == "__main__":
    main()

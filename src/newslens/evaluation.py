from __future__ import annotations

from pathlib import Path
import time

from . import CONDITIONS
from .common import digest, environment, folder_digest, read_json, read_rows, require_match, run_identity, source_manifest, write_json, write_rows
from .data import load_prepared
from .runtime import generate_one, load_model, memory_snapshot


def freeze_protocol(run_dir, identity):
    path = Path(run_dir) / "protocol.json"
    require_match(path, identity)
    write_json(path, identity)


def validate_predictions(path, expected, identity, condition, complete=True):
    path = Path(path)
    metadata = read_json(path.with_suffix(".manifest.json"))
    if metadata["identity"] != identity or metadata["condition"] != condition:
        raise ValueError(f"Prediction provenance mismatch: {path}")
    rows = read_rows(path) if path.exists() else []
    expected_ids = {r["id"]: r["label"] for r in expected}
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)) or not set(ids).issubset(expected_ids):
        raise ValueError(f"Unexpected or duplicate predictions: {path}")
    if any(r["label"] != expected_ids[r["id"]] or r["run_id"] != identity["run_id"] or r["condition"] != condition or r["prediction"] not in (-1, 0, 1, 2, 3) for r in rows):
        raise ValueError(f"Prediction labels/identity mismatch: {path}")
    if complete and set(ids) != set(expected_ids):
        raise ValueError(f"Incomplete predictions: {len(ids)}/{len(expected_ids)} for {condition}")
    return rows


def evaluate(config, data_dir, model_dir, run_dir, adapter=None, conditions=None):
    from .training import require_gate
    splits, demos, manifest = load_prepared(data_dir)
    identity = run_identity(config, manifest)
    require_gate(run_dir, identity)
    freeze_protocol(run_dir, identity)
    conditions = conditions or (["lora_concise_zero"] if adapter else list(CONDITIONS))
    if adapter and conditions != ["lora_concise_zero"]:
        raise ValueError("LoRA is evaluated only with the canonical concise zero-shot prompt.")
    if not adapter and any(c not in CONDITIONS for c in conditions):
        raise ValueError("Unknown unchanged-model condition.")
    if adapter:
        summary = read_json(Path(run_dir) / "training" / "summary.json")
        if summary["identity"] != identity:
            raise ValueError("Adapter training provenance mismatch.")
    adapter_sha = folder_digest(adapter) if adapter else None
    pending = []
    for condition in conditions:
        path = Path(run_dir) / "predictions" / f"{condition}.jsonl"
        metadata = {"identity": identity, "condition": condition, "adapter_sha256": adapter_sha, "split": "test"}
        require_match(path.with_suffix(".manifest.json"), metadata)
        write_json(path.with_suffix(".manifest.json"), metadata)
        prior = validate_predictions(path, splits["test"], identity, condition, complete=False)
        if len(prior) != len(splits["test"]):
            pending.append((condition, path, prior))
    if not pending:
        print("All requested predictions already complete.", flush=True)
        return
    model, tokenizer, hardware = load_model(model_dir, config, adapter=adapter)
    import torch
    torch.cuda.reset_peak_memory_stats()
    write_json(Path(run_dir) / "execution_manifest.json", {"identity": identity, "source": source_manifest(), "environment": environment(), "hardware": hardware})
    for condition, path, prior in pending:
        predictions = list(prior)
        seen = {r["id"] for r in prior}
        prompt_condition = condition.removeprefix("lora_")
        started = time.perf_counter()
        for row in splits["test"]:
            if row["id"] in seen:
                continue
            value = generate_one(model, tokenizer, row["text"], prompt_condition, demos, config)
            predictions.append({"id": row["id"], "label": row["label"], "run_id": identity["run_id"], "condition": condition, **value})
            if len(predictions) % config["prediction_checkpoint_examples"] == 0:
                write_rows(path, predictions)
                print({"condition": condition, "completed": len(predictions), "total": len(splits["test"])}, flush=True)
        write_rows(path, predictions)
        validate_predictions(path, splits["test"], identity, condition)
        write_json(path.with_suffix(".timing.json"), {"current_session_seconds": time.perf_counter() - started, "total_generation_seconds": sum(p["seconds"] for p in predictions), "memory": memory_snapshot(), "hardware": hardware})


def predict(config, model_dir, text, adapter=None):
    from . import LABELS
    model, tokenizer, hardware = load_model(model_dir, config, adapter=adapter)
    result = generate_one(model, tokenizer, text, "concise_zero", [], config)
    return {"category": LABELS[result["prediction"]] if result["prediction"] >= 0 else None, **result}

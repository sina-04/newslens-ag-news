from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import re
import unicodedata

from . import LABELS
from .common import digest, read_json, read_rows, write_json, write_rows


def duplicate_key(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def split_records(train, test, validation_fraction=0.1, seed=42):
    from sklearn.model_selection import train_test_split
    test_keys = {duplicate_key(r["text"]) for r in test}
    groups = defaultdict(list)
    exclusions = []
    for row in train:
        key = duplicate_key(row["text"])
        if key in test_keys:
            exclusions.append({"id": row["id"], "reason": "test_overlap"})
        elif not key:
            exclusions.append({"id": row["id"], "reason": "empty"})
        else:
            groups[key].append(row)
    pool = []
    for key, rows in sorted(groups.items()):
        if len({r["label"] for r in rows}) != 1:
            exclusions.extend({"id": r["id"], "reason": "conflicting_labels"} for r in rows)
            continue
        rows = sorted(rows, key=lambda r: r["id"])
        pool.append({**rows[0], "duplicate_sha256": digest(key)})
        exclusions.extend({"id": r["id"], "reason": "training_duplicate"} for r in rows[1:])
    training, validation = train_test_split(pool, test_size=validation_fraction, stratify=[r["label"] for r in pool], random_state=seed)
    return {"train": sorted(training, key=lambda r: r["id"]), "validation": sorted(validation, key=lambda r: r["id"]), "test": test}, exclusions


def prepare(config, output_dir):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    output = Path(output_dir)
    request = {"dataset_id": config["dataset_id"], "dataset_revision": config.get("dataset_revision"), "seed": config["seed"], "validation_fraction": config["validation_fraction"], "normalization": "NFKC/casefold/whitespace-v1"}
    if (output / "manifest.json").exists():
        manifest = read_json(output / "manifest.json")
        if manifest["preparation"] != request:
            raise ValueError("Existing data uses different preparation settings.")
        load_prepared(output)
        return manifest
    revision = config.get("dataset_revision") or HfApi().dataset_info(config["dataset_id"]).sha
    raw = load_dataset(config["dataset_id"], revision=revision)
    if raw["train"].features["label"].names != list(LABELS):
        raise ValueError("Unexpected AG News label order.")
    records = {split: [{"id": f"{split}:{i:06d}", "text": row["text"], "label": int(row["label"])} for i, row in enumerate(raw[split])] for split in ("train", "test")}
    if len(records["train"]) != 120000 or len(records["test"]) != 7600:
        raise ValueError("Unexpected official dataset size.")
    splits, exclusions = split_records(**records, validation_fraction=config["validation_fraction"], seed=config["seed"])
    for name, rows in splits.items():
        write_rows(output / f"{name}.jsonl", rows)
    write_rows(output / "exclusions.jsonl", exclusions)
    demos = [next(r for r in splits["train"] if r["label"] == label) for label in range(4)]
    write_json(output / "demonstrations.json", demos)
    identity = {"revision": revision, "splits": {name: digest(rows) for name, rows in splits.items()}, "demonstrations": digest(demos)}
    manifest = {"preparation": request, "dataset_revision": revision, "labels": list(LABELS), "content_sha256": digest(identity), "identity": identity, "counts": {name: len(rows) for name, rows in splits.items()}, "exclusion_counts": dict(Counter(r["reason"] for r in exclusions)), "class_counts": {name: {LABELS[label]: sum(r["label"] == label for r in rows) for label in range(4)} for name, rows in splits.items()}, "character_lengths": {name: {"min": min(len(r["text"]) for r in rows), "max": max(len(r["text"]) for r in rows), "mean": sum(len(r["text"]) for r in rows) / len(rows)} for name, rows in splits.items()}, "examples": demos}
    write_json(output / "manifest.json", manifest)
    return manifest


def load_prepared(directory):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    splits = {name: read_rows(directory / f"{name}.jsonl") for name in ("train", "validation", "test")}
    demos = read_json(directory / "demonstrations.json")
    identity = {"revision": manifest["dataset_revision"], "splits": {name: digest(rows) for name, rows in splits.items()}, "demonstrations": digest(demos)}
    if digest(identity) != manifest["content_sha256"]:
        raise ValueError("Prepared dataset or demonstrations changed.")
    return splits, demos, manifest

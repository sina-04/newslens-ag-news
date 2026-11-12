from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def read_rows(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def load_config(path):
    import yaml
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if config["microbatch"] != 1 or config["gradient_accumulation"] != 32:
        raise ValueError("The frozen T4 protocol requires effective batch size 32 (1 x 32).")
    return config


def source_manifest():
    root = Path(__file__).resolve().parents[2]
    files = [*root.glob("src/newslens/*.py"), *root.glob("configs/*.yaml"), root / "pyproject.toml"]
    hashes = {str(p.relative_to(root)).replace("\\", "/"): file_digest(p) for p in sorted(files) if p.exists()}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        bundle = root / "SOURCE_MANIFEST.json"
        commit = read_json(bundle).get("git_commit") if bundle.exists() else None
    return {"git_commit": commit, "files": hashes, "content_sha256": digest(hashes)}


def environment():
    packages = {}
    for name in ("torch", "torchvision", "torchaudio", "torchao", "torchcodec", "xformers", "transformers", "tokenizers", "accelerate", "safetensors", "peft", "trl", "unsloth", "unsloth_zoo", "triton", "bitsandbytes", "datasets", "huggingface_hub", "numpy", "scikit-learn", "pillow"):
        try:
            dist = importlib.metadata.distribution(name)
            packages[name] = {"version": dist.version, "direct_url": json.loads(dist.read_text("direct_url.json") or "null")}
        except importlib.metadata.PackageNotFoundError:
            continue
    return {"python": sys.version, "packages": packages, "captured_at": datetime.now(timezone.utc).isoformat()}


def run_identity(config, data_manifest):
    source = source_manifest()
    identity = {"config": config, "data_sha256": data_manifest["content_sha256"], "source_sha256": source["content_sha256"]}
    return {**identity, "run_id": digest(identity)}


def require_match(path, identity):
    if Path(path).exists() and read_json(path) != identity:
        raise ValueError(f"Refusing to reuse mismatched artifacts: {path}")


def folder_digest(path):
    root = Path(path)
    files = {str(p.relative_to(root)): file_digest(p) for p in sorted(root.rglob("*")) if p.is_file() and not p.name.endswith(".tmp")}
    return digest(files)

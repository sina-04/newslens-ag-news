from __future__ import annotations

from pathlib import Path
import os
import tarfile
import tempfile

from .common import file_digest, read_json, write_json


def archive_checkpoint(checkpoint, destination, identity):
    checkpoint, destination = Path(checkpoint), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"{checkpoint.name}.tar.gz"
    temporary = target.with_suffix(".partial")
    with tarfile.open(temporary, "w:gz", compresslevel=1) as archive:
        archive.add(checkpoint, arcname=checkpoint.name)
    os.replace(temporary, target)
    write_json(target.with_suffix(".json"), {"identity": identity, "sha256": file_digest(target), "checkpoint": checkpoint.name})
    return target


def restore_checkpoint(archive, output, identity):
    archive, output = Path(archive), Path(output)
    manifest = read_json(archive.with_suffix(".json"))
    if manifest["identity"] != identity or manifest["sha256"] != file_digest(archive):
        raise ValueError("Checkpoint identity/checksum mismatch.")
    output.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as bundle:
        root = output.resolve()
        for member in bundle.getmembers():
            resolved = (root / member.name).resolve()
            if not resolved.is_relative_to(root) or member.issym() or member.islnk():
                raise ValueError("Unsafe archive entry.")
        bundle.extractall(output, filter="data")
    return output / manifest["checkpoint"]


def latest_archive(directory):
    candidates = [p for p in Path(directory).glob("checkpoint-*.tar.gz") if p.with_suffix(".json").exists()]
    return max(candidates, key=lambda p: int(p.name.split("-")[1].split(".")[0]), default=None)


def prune_archives(directory, best_checkpoint=None):
    """Retain two latest recoveries plus best, after a complete save."""
    directory = Path(directory)
    archives = sorted((p for p in directory.glob("checkpoint-*.tar.gz") if p.with_suffix(".json").exists()), key=lambda p: int(p.name.split("-")[1].split(".")[0]))
    keep = set(archives[-2:])
    if best_checkpoint:
        keep.add(directory / f"{Path(best_checkpoint).name}.tar.gz")
    for path in archives:
        if path not in keep:
            path.unlink()
            path.with_suffix(".json").unlink()

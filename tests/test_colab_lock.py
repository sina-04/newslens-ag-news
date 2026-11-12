"""Kernel restoration is tested without installing the CUDA stack locally."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from colab_lock import locked_requirements


def test_restore_keeps_upstream_commits_and_external_kernels():
    gate = {"environment": {"packages": {
        "torch": {"version": "2.8.0+cu128", "direct_url": None},
        "unsloth": {"version": "2026.11.4", "direct_url": {"url": "https://github.com/unslothai/unsloth", "vcs_info": {"commit_id": "fixed-commit"}}}}}}
    freeze = "causal-conv1d @ https://example.com/fixed-causal.whl\nmamba_ssm @ https://example.com/fixed-mamba.whl\ncut-cross-entropy==25.1.1\n"
    packages, kernels = locked_requirements(gate, freeze)
    assert "torch==2.8.0" in packages
    assert "unsloth[base] @ git+https://github.com/unslothai/unsloth@fixed-commit" in packages
    assert kernels == freeze.splitlines()


def test_restore_rejects_missing_kernel_lock():
    gate = {"environment": {"packages": {}}}
    with pytest.raises(ValueError, match="missing required recorded kernel"):
        locked_requirements(gate, "cut-cross-entropy==25.1.1\n")

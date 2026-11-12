"""Reconstruct the verified GPU install, including separately installed kernels."""


def locked_requirements(gate, freeze_text):
    locked = []
    for name, record in gate["environment"]["packages"].items():
        direct = record.get("direct_url") or {}
        vcs = direct.get("vcs_info", {})
        if vcs.get("commit_id"):
            extras = "[base]" if name in ("unsloth", "unsloth_zoo") else ""
            locked.append(f"{name}{extras} @ git+{direct['url']}@{vcs['commit_id']}")
        else:
            locked.append(f"{name}=={record['version'].split('+')[0]}")
    required = {"causal-conv1d", "mamba-ssm", "cut-cross-entropy"}
    kernels = {}
    for line in freeze_text.splitlines():
        name = line.split(" @", 1)[0].split("==", 1)[0].replace("_", "-").lower()
        if name in required:
            kernels[name] = line
    if set(kernels) != required:
        raise ValueError("The passing GPU lock is missing required recorded kernel packages.")
    return locked, list(kernels.values())

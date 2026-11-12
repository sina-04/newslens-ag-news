"""Install the upstream T4-compatible stack. Run in a fresh Colab runtime."""
import os
from pathlib import Path
import subprocess
import sys
import json

if not Path("/content").exists():
    raise SystemExit("This installer is intended for a disposable Colab Linux runtime.")

def pip(*args):
    with subprocess.Popen([sys.executable, "-m", "pip", "install", *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, process.args)

pip("uv")
# Resume an existing verified run with its recorded library versions and Git commits.
gate_path = Path("/content/drive/MyDrive/NewsLens/runs/qwen35-4b-v1/smoke/gate.json")
if gate_path.exists():
    gate = json.loads(gate_path.read_text())
    if gate.get("passed"):
        locked = []
        for name, record in gate["environment"]["packages"].items():
            direct = record.get("direct_url") or {}
            vcs = direct.get("vcs_info", {})
            if vcs.get("commit_id"):
                locked.append(f"{name} @ git+{direct['url']}@{vcs['commit_id']}")
            else:
                locked.append(f"{name}=={record['version'].split('+')[0]}")
        pip(*locked)
        pip("-e", ".", "--no-deps")
        print("Recorded GPU stack restored. Restart the session before running NewsLens.")
        raise SystemExit(0)
# Match the published Qwen3.5-4B T4 notebook; no moving Git refs in saved provenance.
import urllib.request
def head(repo):
    req = urllib.request.Request(f"https://api.github.com/repos/{repo}/commits/main", headers={"User-Agent": "NewsLens"})
    with urllib.request.urlopen(req) as response:
        return json.load(response)["sha"]

revisions = {repo: head(repo) for repo in ("unslothai/unsloth", "unslothai/unsloth-zoo")}
pip("torch==2.8.0", "torchaudio==2.8.0", "torchvision==0.23.0", "triton==3.4.0", "xformers==0.0.32.post2", "bitsandbytes", f"unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo@{revisions['unslothai/unsloth-zoo']}", f"unsloth[base] @ git+https://github.com/unslothai/unsloth@{revisions['unslothai/unsloth']}")
pip("--no-deps", "torchcodec==0.7.0", "trl==0.22.2", "tokenizers==0.22.2")
pip("transformers==5.2.0")
subprocess.run(["unsloth", "install-kernels"], check=True)
pip("--no-deps", "torchao>=0.16.0")
pip("-e", ".", "pytest")
Path("INSTALL_REVISIONS.json").write_text(json.dumps(revisions, indent=2))
print("Installation complete. Restart the Colab session before importing the GPU stack.")

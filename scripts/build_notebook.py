"""Extend the user's downloaded notebook without replacing the original cells."""
from pathlib import Path
import json

root = Path(__file__).resolve().parents[1]
path = root / "notebooks" / "newslens_ag_news.ipynb"
notebook = json.loads(path.read_text(encoding="utf-8"))
notebook["cells"] = notebook["cells"][:3]
notebook["metadata"].pop("widgets", None)

def cell(source, kind="code"):
    value = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    notebook["cells"].append(value)

cell("""# NewsLens: controlled Qwen3.5-4B experiment

The three original preparation cells above are preserved. Do not rerun the download cell when the fixed snapshot already exists. Use the T4 runtime. Execute each stage in order and inspect failures before continuing. Model weights and datasets stay outside Git.

## Upload the source bundle

Build `artifacts/newslens-source.zip` locally with `python scripts/build_bundle.py`, then select it below. Resume using the same source bundle and configuration.
""", "markdown")
cell('''from google.colab import files
from pathlib import Path
import zipfile, os
uploaded = files.upload()
bundle_name = next(name for name in uploaded if name.endswith(".zip"))
workspace = Path("/content/newslens")
workspace.mkdir(exist_ok=True)
with zipfile.ZipFile(bundle_name) as archive:
    for name in archive.namelist():
        if not (workspace / name).resolve().is_relative_to(workspace.resolve()):
            raise ValueError("Unsafe source bundle entry")
    archive.extractall(workspace)
os.chdir(workspace)
print("Source ready:", workspace)
''')
cell("""## Install the compatible GPU stack

Run once in a fresh runtime, then choose **Runtime → Restart session** before continuing. On later sessions, reinstall the package versions and Git revisions captured by the passing smoke test. The installer starts from the official Unsloth T4 recipe.
""", "markdown")
cell('''import subprocess, sys
subprocess.run([sys.executable, "scripts/install_colab.py"], check=True)
''')
cell('''from google.colab import drive
drive.mount("/content/drive")
from pathlib import Path
import os, subprocess, sys
os.chdir("/content/newslens")
ROOT = Path("/content/drive/MyDrive/NewsLens")
DATA = ROOT / "data" / "prepared-v1"
RUN = ROOT / "runs" / "qwen35-4b-v1"
MODEL = Path("/content/models/Qwen3.5-4B")
def run(stage, *extra):
    subprocess.run([sys.executable, "-u", "-m", "newslens", stage,
        "--data-dir", str(DATA), "--run-dir", str(RUN), "--model-dir", str(MODEL), *extra], check=True)
run("stage-model")
run("prepare")
''')
cell("""## Disposable compatibility gate

This performs 50 optimizer steps with a deliberate checkpoint stop/resume at step 25. It verifies frozen weights, adapter updates, finite gradients, reload equivalence, and GPU headroom. Its adapter is discarded from the actual experiment. Read the measured duration estimate before the full run; free Colab sessions may require multiple resumptions.
""", "markdown")
cell('''run("smoke-test")
import json
gate = json.loads((RUN / "smoke" / "gate.json").read_text())
assert gate["passed"]
print({k: gate[k] for k in ("estimated_training_hours_excluding_validation", "estimated_evaluation_hours", "memory")})
''')
cell("""## Unchanged-model baseline and prompt experiments

Runs all six frozen prompt conditions against the full official test set. Predictions persist every 100 examples. Repeating this stage resumes matching incomplete files.
""", "markdown")
cell('''run("evaluate")
''')
cell("""## Full LoRA training

Starts from the original snapshot. All six unchanged-model evaluations must be complete. Recovery archives include optimizer/scheduler/RNG state. Repeating this stage restores the latest complete checkpoint. Full validation loss selects the best epoch checkpoint.
""", "markdown")
cell('''run("train")
''')
cell("""## LoRA test evaluation and measured report""", "markdown")
cell('''run("evaluate", "--adapter", str(RUN / "adapter"))
run("report", "--output-dir", str(RUN / "results"))
from IPython.display import Markdown, display
display(Markdown((RUN / "results" / "REPORT.md").read_text()))
''')
cell('''examples = [
    "Diplomats from several nations agreed to resume peace negotiations next week.",
    "The national team won the final after a dramatic penalty shootout.",
    "The retailer reported higher quarterly profits as online sales grew.",
    "Researchers developed a new chip design that reduces computing energy use.",
]
for text in examples:
    run("predict", "--adapter", str(RUN / "adapter"), "--text", text)
''')
cell("""## Export artifacts

Keep the base snapshot and recovery archives in Drive. Export the adapter, measured predictions, run metadata, GPU lock file, and generated report for final verification and GitHub publication. Download this executed notebook using File → Download → Download .ipynb.
""", "markdown")
cell('''import zipfile
from google.colab import files
export = Path("/content/newslens-results.zip")
with zipfile.ZipFile(export, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for relative in ("adapter", "results", "predictions"):
        for path in (RUN / relative).rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(RUN))
    for relative in ("protocol.json", "execution_manifest.json", "smoke/gate.json", "smoke/requirements-gpu.lock.txt", "training/summary.json"):
        archive.write(RUN / relative, relative)
files.download(str(export))
''')
notebook["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(path)

"""Create a source-only upload bundle; model/data/credentials never enter it."""
from pathlib import Path
import json
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from newslens.common import source_manifest

target = ROOT / "artifacts" / "newslens-source.zip"
target.parent.mkdir(exist_ok=True)
files = [*ROOT.glob("src/newslens/*.py"), *ROOT.glob("configs/*.yaml"), *ROOT.glob("scripts/*.py"), *ROOT.glob("tests/*.py"), *ROOT.glob("notebooks/*.ipynb"), ROOT / "pyproject.toml", ROOT / "README.md", ROOT / "LICENSE", ROOT / "IMPLEMENTATION_PLAN.md"]
with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(files):
        if path.is_file():
            bundle.write(path, str(path.relative_to(ROOT)).replace("\\", "/"))
    bundle.writestr("SOURCE_MANIFEST.json", json.dumps(source_manifest(), indent=2))
print(target)

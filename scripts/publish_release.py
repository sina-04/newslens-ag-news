"""Publish verified NewsLens artifacts; default mode performs local checks only.

API references:
https://docs.github.com/en/rest/releases/releases#create-a-release
https://docs.github.com/en/rest/releases/assets#upload-a-release-asset
"""
from pathlib import Path
import argparse
import hashlib
import json
import mimetypes
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from newslens.common import file_digest, load_config, read_json, write_json
from verify_results import verify_run

REPO = "sina-04/newslens-ag-news"
TAG = "v1.0.0"
API = f"https://api.github.com/repos/{REPO}"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def credential():
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    # Use the already configured Git credential helper, without printing secrets.
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "Never"}
    result = subprocess.run(["git", "-c", "credential.interactive=never", "credential", "fill"],
        cwd=ROOT, input=f"protocol=https\nhost=github.com\npath={REPO}.git\n\n",
        capture_output=True, text=True, env=env)
    if result.returncode:
        raise RuntimeError("GitHub credentials are unavailable to the release API.")
    fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    if not fields.get("password"):
        raise RuntimeError("Git credential helper returned no access token.")
    return fields["password"]


class Client:
    def __init__(self, token):
        self.token = token

    def request(self, url, method="GET", payload=None, binary=None, content_type=None, missing_ok=False):
        # Never send the token to URLs supplied by artifact contents or redirects.
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https" or parsed.hostname not in ("api.github.com", "uploads.github.com"):
            raise ValueError("Unexpected GitHub API host.")
        headers = {"Accept": "application/vnd.github+json", "Authorization": f"Bearer {self.token}",
                   "X-GitHub-Api-Version": "2026-03-10", "User-Agent": "NewsLens-release"}
        data = binary
        if payload is not None:
            data = json.dumps(payload).encode()
            content_type = "application/json"
        if content_type:
            headers["Content-Type"] = content_type
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                raise RuntimeError("Authenticated API redirect refused.")
        try:
            with urllib.request.build_opener(NoRedirect()).open(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if missing_ok and error.code == 404:
                return None
            raise RuntimeError(f"GitHub API {method} failed with HTTP {error.code}.") from None


def check_asset(asset, path):
    expected = "sha256:" + file_digest(path)
    if asset["state"] != "uploaded" or asset["size"] != path.stat().st_size or asset.get("digest") != expected:
        raise ValueError(f"Remote asset integrity mismatch: {path.name}")


def check_bundle(bundle, run):
    expected = {str(p.relative_to(run)).replace("\\", "/"): p for p in (run / "adapter").rglob("*") if p.is_file()}
    for name in ("protocol.json", "execution_manifest.json", "smoke/gate.json", "smoke/requirements-gpu.lock.txt", "training/summary.json", "stage_sessions.jsonl"):
        expected[name] = run / name
    with zipfile.ZipFile(bundle) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())) or set(archive.namelist()) != set(expected) | {"checksums.json", "BASE_MODEL.txt"}:
            raise ValueError("Release bundle contains unexpected or missing entries.")
        for name, path in expected.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != file_digest(path):
                raise ValueError(f"Release bundle differs from the verified run: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=ROOT / "artifacts" / "verified-run")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "prepared")
    parser.add_argument("--notes", type=Path, default=ROOT / "docs" / "RELEASE_v1.0.0.md")
    parser.add_argument("--publish", action="store_true", help="Create/upload/publish the verified release.")
    args = parser.parse_args()
    config = load_config(ROOT / "configs" / "colab_t4.yaml")
    identity = verify_run(args.run_dir, config, args.data_dir)
    asset = ROOT / "artifacts" / "newslens-qwen35-4b-lora-v1.0.0.zip"
    checksum = asset.with_suffix(".sha256.json")
    checks = read_json(checksum)
    if checks.get(asset.name) != file_digest(asset) or checks.get("run_id") != identity["run_id"]:
        raise ValueError("Release bundle checksum mismatch.")
    check_bundle(asset, args.run_dir)
    notes = args.notes.read_text(encoding="utf-8")
    if git("symbolic-ref", "--short", "HEAD") != "main" or git("status", "--porcelain"):
        raise ValueError("Publication requires the clean verified main branch.")
    origin = git("remote", "get-url", "origin")
    if origin not in (f"https://github.com/{REPO}", f"https://github.com/{REPO}.git"):
        raise ValueError("Origin differs from the authorized NewsLens repository.")
    commit = git("rev-parse", "HEAD")
    if not args.publish:
        print(json.dumps({"ready": True, "commit": commit, "run_id": identity["run_id"], "assets": [asset.name, checksum.name]}))
        return
    client = Client(credential())
    remote = client.request(API + "/commits/main")
    if remote["sha"] != commit:
        raise ValueError("Remote main does not match the verified local commit.")
    tag = client.request(API + "/git/ref/tags/" + TAG, missing_ok=True)
    if tag:
        obj = tag["object"]
        for _ in range(5):
            if obj["type"] != "tag":
                break
            obj = client.request(API + "/git/tags/" + obj["sha"])["object"]
        if obj["type"] != "commit" or obj["sha"] != commit:
            raise ValueError("Existing release tag differs from the verified commit.")
    release = client.request(API + "/releases/tags/" + TAG, missing_ok=True)
    if release is None:
        # Draft tags may not be discoverable through the tag endpoint.
        drafts = client.request(API + "/releases?per_page=100")
        release = next((r for r in drafts if r["tag_name"] == TAG), None)
    if release is None:
        release = client.request(API + "/releases", "POST", payload={"tag_name": TAG,
            "target_commitish": commit, "name": "NewsLens — Qwen3.5-4B AG News LoRA", "body": notes, "draft": True})
    elif not release["draft"]:
        raise ValueError("The release is already published; refusing to modify an existing release.")
    elif release["target_commitish"] not in (commit, "main"):
        raise ValueError("Existing draft release targets a different commit.")
    assets = client.request(API + f"/releases/{release['id']}/assets?per_page=100")
    for path in (asset, checksum):
        uploaded = next((a for a in assets if a["name"] == path.name), None)
        if uploaded is None:
            url = release["upload_url"].split("{", 1)[0] + "?" + urllib.parse.urlencode({"name": path.name})
            uploaded = client.request(url, "POST", binary=path.read_bytes(),
                content_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            assets.append(uploaded)
        check_asset(uploaded, path)
    release = client.request(API + f"/releases/{release['id']}", "PATCH", payload={"draft": False, "body": notes, "make_latest": "true"})
    receipt = {"release_url": release["html_url"], "commit": commit, "run_id": identity["run_id"], "assets": {}}
    for path in (asset, checksum):
        uploaded = next(a for a in assets if a["name"] == path.name)
        # Public download uses no Authorization header, including on redirects.
        url = uploaded["browser_download_url"]
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https" or parsed.hostname != "github.com" or not parsed.path.startswith(f"/{REPO}/releases/download/{TAG}/"):
            raise ValueError("Unexpected public release URL.")
        h = hashlib.sha256()
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "NewsLens-release-check"}), timeout=60) as response:
            for chunk in iter(lambda: response.read(2**20), b""):
                h.update(chunk)
        if h.hexdigest() != file_digest(path):
            raise ValueError("Downloaded release checksum differs from local artifact.")
        receipt["assets"][path.name] = {"url": url, "sha256": h.hexdigest()}
    write_json(ROOT / "artifacts" / "publication_receipt.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()

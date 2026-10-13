"""Download and verify the released, training-ready EBV-Eye caches."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.parse
import urllib.request

from code.paths import PROJECT_ROOT, CACHE_ROOT


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class StripRedirectCredentials(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, url):
        redirected = super().redirect_request(request, fp, code, message, headers, url)
        if redirected and urllib.parse.urlsplit(url).netloc != urllib.parse.urlsplit(request.full_url).netloc:
            redirected.remove_header("Authorization")
        return redirected


def validate_record(directory: Path, record: dict):
    for name, expected in record["files"].items():
        path = directory / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"cache file missing or SHA-256 differs: {path}")


def install_archive(archive: Path, output: Path, record: dict):
    if sha256(archive) != record["sha256"]:
        raise ValueError(f"archive SHA-256 differs: {archive.name}")
    relative = Path(record["relative_cache_dir"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("invalid cache directory in catalog")
    destination = output / relative
    if destination.exists():
        validate_record(destination, record)
        return
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cache-download-", dir=output) as temporary:
        staging = Path(temporary)
        allowed = {str(relative / name) for name in record["files"]}
        with tarfile.open(archive, "r:") as source:
            members = source.getmembers()
            if {m.name for m in members} != allowed or len(members) != len(allowed):
                raise ValueError("cache archive contains unexpected members")
            for member in members:
                if not member.isfile():
                    raise ValueError("cache archive must contain only regular files")
            source.extractall(staging, members=members, filter="data")
        validate_record(staging / relative, record)
        destination.parent.mkdir(parents=True, exist_ok=True)
        (staging / relative).rename(destination)


def download_record(record: dict, output: Path, *, archive_root: Path | None = None,
                    base_url: str | None = None):
    destination = output / record["relative_cache_dir"]
    if destination.exists():
        validate_record(destination, record)
        return
    if archive_root:
        install_archive(archive_root / record["asset_name"], output, record)
        return
    url = (base_url.rstrip("/") + "/" + record["asset_name"]) if base_url else record.get("url")
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not base_url and token and record.get("asset_api_url"):
        url = record["asset_api_url"]
    if not url:
        raise RuntimeError("Training-cache download URLs have not been published. Use a published catalog, --base-url, or --archive-root.")
    headers = {"User-Agent": "EBV-Eye-training-cache", "Accept": "application/octet-stream"}
    if token and urllib.parse.urlsplit(url).netloc == "api.github.com":
        headers["Authorization"] = f"Bearer {token}"
    opener = urllib.request.build_opener(StripRedirectCredentials())
    with tempfile.TemporaryDirectory(prefix="ebv-cache-transfer-") as temporary:
        archive = Path(temporary) / record["asset_name"]
        with opener.open(urllib.request.Request(url, headers=headers), timeout=120) as response, archive.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        install_archive(archive, output, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=PROJECT_ROOT / "datasets/formal_eye0.json")
    parser.add_argument("--output", type=Path, default=CACHE_ROOT)
    parser.add_argument("--users", nargs="+", type=int, default=[15])
    parser.add_argument("--tasks", nargs="+", choices=("pursuit", "saccade"), default=["pursuit", "saccade"])
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument("--base-url")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text())
    selected = [r for r in catalog["records"] if r["user"] in args.users and r["task"] in args.tasks]
    expected = {(task, user) for task in args.tasks for user in args.users}
    if {(r["task"], r["user"]) for r in selected} != expected:
        raise ValueError("requested user/task is absent from the cache catalog")
    if args.dry_run:
        print(json.dumps({"records": len(selected), "bytes": sum(r["bytes"] for r in selected),
                          "publication_status": catalog["publication_status"]}, indent=2))
        return
    for record in selected:
        download_record(record, args.output, archive_root=args.archive_root, base_url=args.base_url)
        print(f"verified {record['task']} user{record['user']} eye0", flush=True)


if __name__ == "__main__":
    main()

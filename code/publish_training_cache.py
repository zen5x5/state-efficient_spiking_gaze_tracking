"""Upload prepared cache archives to a GitHub Release and write a live catalog.

Requires GH_TOKEN or GITHUB_TOKEN with repository Contents write access.
No raw-data preprocessing is performed. Existing assets must have matching
GitHub SHA-256 digests; uploads are never silently replaced.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
from pathlib import Path
import urllib.parse
import urllib.request
import urllib.error

from code.download_dataset import sha256

API_VERSION = "2026-03-10"


def api(token: str, url: str, method: str = "GET", data=None):
    body = None if data is None else json.dumps(data).encode()
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
               "X-GitHub-Api-Version": API_VERSION, "User-Agent": "EBV-Eye-cache-publisher"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, method=method, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def upload(token: str, upload_url: str, path: Path):
    url = urllib.parse.urlsplit(upload_url.split("{")[0] + "?" + urllib.parse.urlencode({"name": path.name}))
    connection = http.client.HTTPSConnection(url.netloc, timeout=300)
    connection.putrequest("POST", url.path + "?" + url.query)
    for key, value in {"Authorization": f"Bearer {token}", "Content-Type": "application/octet-stream",
                       "Content-Length": str(path.stat().st_size), "Accept": "application/vnd.github+json",
                       "X-GitHub-Api-Version": API_VERSION, "User-Agent": "EBV-Eye-cache-publisher"}.items():
        connection.putheader(key, value)
    connection.endheaders()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            connection.send(block)
    response = connection.getresponse()
    result = json.loads(response.read())
    connection.close()
    if response.status != 201:
        raise RuntimeError(f"GitHub asset upload failed with HTTP {response.status}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--repository")
    parser.add_argument("--target-branch")
    parser.add_argument("--output-catalog", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text())
    for record in catalog["records"]:
        path = args.archive_root / record["asset_name"]
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"prepared archive differs: {path.name}")
    repository = args.repository or catalog["repository"]
    branch = args.target_branch or catalog["target_branch"]
    if args.dry_run:
        print(json.dumps({"repository": repository, "branch": branch, "release_tag": catalog["release_tag"],
                          "assets": len(catalog["records"]), "bytes": sum(r["bytes"] for r in catalog["records"])}, indent=2))
        return
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("No GitHub upload credential. Set GH_TOKEN or GITHUB_TOKEN in your terminal; do not put it in the repository.")
    root = f"https://api.github.com/repos/{repository}"
    api(token, root + "/branches/" + urllib.parse.quote(branch, safe=""))
    releases = api(token, root + "/releases?per_page=100")
    matches = [r for r in releases if r["tag_name"] == catalog["release_tag"]]
    if len(matches) > 1:
        raise ValueError("multiple releases use the requested cache tag")
    if matches:
        release = matches[0]
        if release["target_commitish"] != branch:
            raise ValueError("existing release targets a different branch")
    else:
        release = api(token, root + "/releases", "POST", {
            "tag_name": catalog["release_tag"], "target_commitish": branch,
            "name": "Formal eye0 training caches", "draft": True,
            "body": "Frozen training, validation and test count-frame caches for users 4–27 eye0. Archive and file checksums are supplied in the data catalog."})
    existing = {a["name"]: a for a in api(token, release["assets_url"] + "?per_page=100")}
    for record in catalog["records"]:
        asset = existing.get(record["asset_name"])
        if asset is None:
            asset = upload(token, release["upload_url"], args.archive_root / record["asset_name"])
        if asset["size"] != record["bytes"] or asset["digest"] != "sha256:" + record["sha256"]:
            raise ValueError(f"uploaded asset SHA-256 differs: {record['asset_name']}")
        record["url"] = asset["browser_download_url"]
        record["asset_api_url"] = asset["url"]
        print(f"verified upload {record['asset_name']}", flush=True)
    if release["draft"]:
        release = api(token, release["url"], "PATCH", {"draft": False, "make_latest": "false"})
    # Draft assets use temporary untagged download URLs. Fetch the final URLs
    # after publication instead of retaining the values returned by upload.
    published_assets = {asset["name"]: asset for asset in
                        api(token, release["assets_url"] + "?per_page=100")}
    for record in catalog["records"]:
        asset = published_assets[record["asset_name"]]
        if (asset["state"] != "uploaded" or asset["size"] != record["bytes"]
                or asset["digest"] != "sha256:" + record["sha256"]):
            raise ValueError(f"published asset differs: {record['asset_name']}")
        record["url"] = asset["browser_download_url"]
        record["asset_api_url"] = asset["url"]
    catalog["repository"] = repository
    catalog["publication_status"] = "published"
    catalog["release_url"] = release["html_url"]
    args.output_catalog.write_text(json.dumps(catalog, indent=2) + "\n")
    print(release["html_url"])


if __name__ == "__main__":
    main()

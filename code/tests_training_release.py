"""Verify archive integrity checks and the release's training entrypoints."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import importlib

from code.download_dataset import install_archive, validate_record
from code.retrain import commands


class TrainingReleaseTests(unittest.TestCase):
    def test_archive_install_and_corruption_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "cache.tar"
            relative = "saccade/fixed_event_window_400_stride200_96x128/user15/eye0"
            data = b"trusted cache payload"
            with tarfile.open(archive, "w") as tf:
                item = tarfile.TarInfo(relative + "/train.npz")
                item.size = len(data)
                tf.addfile(item, io.BytesIO(data))
            record = {"relative_cache_dir": relative,
                      "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                      "files": {"train.npz": hashlib.sha256(data).hexdigest()}}
            install_archive(archive, root / "installed", record)
            validate_record(root / "installed" / relative, record)
            (root / "installed" / relative / "train.npz").write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                install_archive(archive, root / "installed", record)

    def test_archive_cannot_extract_outside_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "cache.tar"
            with tarfile.open(archive, "w") as tf:
                item = tarfile.TarInfo("../escape")
                item.size = 3
                tf.addfile(item, io.BytesIO(b"bad"))
            record = {"relative_cache_dir": "saccade/user15/eye0",
                      "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                      "files": {"train.npz": hashlib.sha256(b"bad").hexdigest()}}
            with self.assertRaises(ValueError):
                install_archive(archive, root / "installed", record)
            self.assertFalse((root / "escape").exists())

    def test_reproduction_commands_parse_in_actual_training_modules(self):
        for task in ("pursuit", "saccade"):
            for family in ("ann", "snn"):
                plan = commands(task, family, 15, 2021, Path("/tmp/cache"), Path("/tmp/result"), "cuda:0")
                module = importlib.import_module(f"code.{task}.fixed_event_window.{family}.training")
                for command in plan:
                    with patch("sys.argv", [command[2], *command[3:]]):
                        args = module.parse_args()
                    self.assertEqual(args.seed, 2021)
                    self.assertEqual(args.cache_dir, Path(f"/tmp/cache/{task}/fixed_event_window_400_stride200_96x128/user15/eye0"))
                    self.assertIn(args.epochs, (100, 200))
                self.assertEqual(len(plan), 1 if family == "ann" else 2)


    def test_publisher_uses_final_urls_after_draft_publication(self):
        from code import publish_training_cache as publisher
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "cache.tar"
            archive.write_bytes(b"prepared cache")
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            catalog = root / "catalog.json"
            catalog.write_text(json.dumps({"repository": "owner/repo", "target_branch": "release/models-training",
                                          "release_tag": "formal-eye0-training-cache", "records": [
                {"asset_name": archive.name, "bytes": archive.stat().st_size, "sha256": digest}]}))
            release = {"url": "https://api.github.com/releases/1", "assets_url": "https://api.github.com/assets",
                       "upload_url": "https://uploads.github.com/assets", "html_url": "https://github.com/owner/repo/releases/tag/formal-eye0-training-cache",
                       "draft": True}
            draft_asset = {"name": archive.name, "size": archive.stat().st_size, "digest": "sha256:" + digest,
                           "state": "uploaded", "url": "https://api.github.com/assets/1",
                           "browser_download_url": "https://github.com/owner/repo/releases/download/untagged-temporary/cache.tar"}
            final_asset = dict(draft_asset, browser_download_url="https://github.com/owner/repo/releases/download/formal-eye0-training-cache/cache.tar")
            published = False

            def fake_api(token, url, method="GET", data=None):
                nonlocal published
                if "/branches/" in url:
                    return {}
                if url.endswith("/releases?per_page=100"):
                    return []
                if url.endswith("/releases") and method == "POST":
                    return release
                if url == release["url"] and method == "PATCH":
                    published = True
                    return dict(release, draft=False)
                if url == release["assets_url"] + "?per_page=100":
                    return [final_asset] if published else []
                raise AssertionError("unexpected API call")

            output = root / "published.json"
            with patch.dict("os.environ", {"GH_TOKEN": "test-credential"}), patch("sys.argv", [
                    "publisher", "--catalog", str(catalog), "--archive-root", str(root),
                    "--output-catalog", str(output)]), patch.object(publisher, "api", side_effect=fake_api), \
                    patch.object(publisher, "upload", return_value=draft_asset) as upload, patch("sys.stdout", io.StringIO()):
                publisher.main()
            result = json.loads(output.read_text())
            self.assertEqual(result["records"][0]["url"], final_asset["browser_download_url"])
            self.assertEqual(result["publication_status"], "published")
            upload.assert_called_once()



if __name__ == "__main__":
    unittest.main()

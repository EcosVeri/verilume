from __future__ import annotations

import os
import stat
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from verilume.core.web_search import (
    CustomJsonSearch,
    UnsafeURLError,
    _is_http_url,
    validate_public_http_url,
)
from verilume.ingest import UploadTooLargeError, save_uploaded_file
from verilume.settings import AppSettings, save_user_config


class UploadLimitTests(unittest.TestCase):
    def test_rejects_file_over_limit_before_writing(self) -> None:
        with TemporaryDirectory() as tmp:
            docs = Path(tmp) / "docs"
            with self.assertRaises(UploadTooLargeError):
                save_uploaded_file("big.pdf", b"x" * 2048, docs, max_bytes=1024)
            # Nothing should have been written.
            self.assertFalse((docs / "big.pdf").exists())

    def test_accepts_file_within_limit(self) -> None:
        with TemporaryDirectory() as tmp:
            docs = Path(tmp) / "docs"
            target = save_uploaded_file("ok.txt", b"hello", docs, max_bytes=1024)
            self.assertTrue(target.exists())

    def test_strips_path_traversal_from_name(self) -> None:
        with TemporaryDirectory() as tmp:
            docs = Path(tmp) / "docs"
            target = save_uploaded_file("../../etc/evil.txt", b"data", docs)
            self.assertEqual(target.parent, docs)
            self.assertEqual(target.name, "evil.txt")


class ConfigPermissionTests(unittest.TestCase):
    def test_saved_config_is_owner_only(self) -> None:
        if os.name != "posix":
            self.skipTest("POSIX permissions only")
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.env"
            save_user_config(AppSettings(hf_token="secret-token"), path=path)
            mode = stat.S_IMODE(path.stat().st_mode)
            self.assertEqual(mode, 0o600)


class SSRFValidationTests(unittest.TestCase):
    def test_allows_public_https(self) -> None:
        validate_public_http_url("https://api.example.com/search", resolve_dns=False)

    def test_rejects_http_by_default(self) -> None:
        with self.assertRaises(UnsafeURLError):
            validate_public_http_url("http://api.example.com", resolve_dns=False)

    def test_rejects_loopback_and_private(self) -> None:
        for url in (
            "https://localhost/x",
            "https://127.0.0.1/x",
            "https://10.1.2.3/x",
            "https://192.168.0.5/x",
            "https://169.254.169.254/latest/meta-data",  # cloud metadata
        ):
            with self.assertRaises(UnsafeURLError):
                validate_public_http_url(url, resolve_dns=False)

    def test_rejects_embedded_credentials(self) -> None:
        with self.assertRaises(UnsafeURLError):
            validate_public_http_url("https://user:pass@example.com", resolve_dns=False)

    def test_custom_search_blocks_private_endpoint(self) -> None:
        provider = CustomJsonSearch(
            provider_name="x",
            api_key="",
            endpoint="https://127.0.0.1/search?q={query}",
        )
        with self.assertRaises(UnsafeURLError):
            provider.search("hello")

    def test_is_http_url_rejects_dangerous_schemes(self) -> None:
        self.assertFalse(_is_http_url("javascript:alert(1)"))
        self.assertFalse(_is_http_url("data:text/html,evil"))
        self.assertFalse(_is_http_url("file:///etc/passwd"))
        self.assertTrue(_is_http_url("https://example.com"))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from unittest.mock import patch

from verilume.app import ACTIVE_SETTINGS_KEY, _clear_rag_cache_if_settings_changed
from verilume.settings import AppSettings


class FakeCachedService:
    def __init__(self) -> None:
        self.clear_calls = 0

    def cache_clear(self) -> None:
        self.clear_calls += 1


class AppCacheTests(unittest.TestCase):
    def test_rag_cache_is_cleared_when_sidebar_settings_change(self) -> None:
        session_state = {}
        cached_service = FakeCachedService()
        first_settings = AppSettings(hf_token="first")
        same_settings = AppSettings(hf_token="first")
        changed_settings = AppSettings(hf_token="second")

        with (
            patch("verilume.app.st.session_state", new=session_state),
            patch("verilume.app.get_rag_service", new=cached_service),
        ):
            _clear_rag_cache_if_settings_changed(first_settings)
            _clear_rag_cache_if_settings_changed(same_settings)
            _clear_rag_cache_if_settings_changed(changed_settings)

        self.assertEqual(cached_service.clear_calls, 2)
        self.assertEqual(session_state[ACTIVE_SETTINGS_KEY], changed_settings)


if __name__ == "__main__":
    unittest.main()

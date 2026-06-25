from __future__ import annotations

import unittest

from verilume.ui.styles import _theme_tokens


class StyleThemeTests(unittest.TestCase):
    def test_theme_tokens_default_to_dark(self) -> None:
        tokens = _theme_tokens("unknown")

        self.assertIn("--veri-bg: #0b0d10;", tokens["variables"])
        self.assertIn("rgba(255, 200, 87", tokens["app_gradient"])

    def test_light_theme_tokens_use_light_surfaces_and_dark_text(self) -> None:
        tokens = _theme_tokens("light")

        self.assertIn("--veri-bg: #f8f9fb;", tokens["variables"])
        self.assertIn("--veri-input-text: #1c2430;", tokens["variables"])
        self.assertIn("rgba(248, 249, 251", tokens["app_gradient"])


if __name__ == "__main__":
    unittest.main()

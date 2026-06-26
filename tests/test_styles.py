from __future__ import annotations

import unittest

from verilume.ui.styles import _BASE_CSS, _theme_tokens


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

    def test_command_bar_tokens_are_available_in_both_themes(self) -> None:
        for appearance in ("dark", "light"):
            with self.subTest(appearance=appearance):
                tokens = _theme_tokens(appearance)

                self.assertIn("--veri-sidebar-width: 320px;", tokens["variables"])
                self.assertIn("--veri-command-width:", tokens["variables"])
                self.assertIn("--veri-command-bottom: 28px;", tokens["variables"])
                self.assertIn("--veri-command-height: 46px;", tokens["variables"])

    def test_secondary_buttons_force_readable_nested_text(self) -> None:
        self.assertIn('button[kind="secondary"][data-testid="baseButton-secondary"]', _BASE_CSS)
        self.assertIn("-webkit-text-fill-color: var(--veri-text) !important;", _BASE_CSS)
        self.assertIn(".stButton > button [data-testid=\"stMarkdownContainer\"] p", _BASE_CSS)
        self.assertIn(".veri-dark-button-anchor", _BASE_CSS)
        self.assertIn("div:has(.veri-dark-button-anchor) + div button", _BASE_CSS)
        self.assertIn('[data-testid="stButton"] button', _BASE_CSS)
        self.assertIn("background-color: var(--veri-panel-2) !important;", _BASE_CSS)


if __name__ == "__main__":
    unittest.main()

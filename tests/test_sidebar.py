from __future__ import annotations

import unittest

from verilume.ui.sidebar import _active_model_html
from verilume.ui.sidebar import (
    _answer_mode_value,
    _benchmark_compare_html,
    _field_label_html,
    _search_source_help,
    _search_source_label,
    _search_source_value,
)
from verilume.settings import AppSettings


class SidebarRenderingTests(unittest.TestCase):
    def test_active_model_html_is_high_contrast_and_escaped(self) -> None:
        html = _active_model_html(
            "Active Hugging Face model",
            "org/model-with-very-long-name<&>",
        )

        self.assertIn("veri-active-model", html)
        self.assertIn("Active Hugging Face model", html)
        self.assertIn("org/model-with-very-long-name&lt;&amp;&gt;", html)
        self.assertNotIn("`", html)

    def test_search_source_labels_use_release_wording(self) -> None:
        self.assertEqual(_search_source_label("Auto"), "Auto (Recommended)")
        self.assertEqual(_search_source_label("Local Only"), "Local")
        self.assertEqual(_search_source_label("Local + AI + Web"), "Hybrid (Local + AI + Web)")
        self.assertEqual(_search_source_label("Web Only"), "Web")

    def test_search_source_help_describes_each_mode(self) -> None:
        help_text = _search_source_help()

        self.assertIn("Auto (Recommended): Searches local documents first", help_text)
        self.assertIn("Local: Uses only indexed local documents.", help_text)
        self.assertIn("Hybrid (Local + AI + Web): Searches local documents", help_text)
        self.assertIn("Web: Uses web sources only.", help_text)

    def test_legacy_research_search_mode_maps_to_answer_mode(self) -> None:
        settings = AppSettings(search_mode="Research Mode", answer_style="Standard")

        self.assertEqual(_search_source_value(settings), "Local + AI + Web")
        self.assertEqual(_answer_mode_value(settings), "Research")

    def test_benchmark_compare_html_uses_single_compact_group(self) -> None:
        html = _benchmark_compare_html()

        self.assertIn("veri-benchmark-compare", html)
        self.assertIn("<strong>Compare</strong>", html)
        self.assertIn("<span>Full</span>", html)
        self.assertIn("<span>Local</span>", html)
        self.assertIn("<span>AI</span>", html)
        self.assertIn("<span>Web</span>", html)

    def test_field_label_help_html_is_escaped_and_themeable(self) -> None:
        html = _field_label_html("HF provider", "Use <auto> & stay safe.")

        self.assertIn("veri-field-help", html)
        self.assertIn("HF provider", html)
        self.assertIn("Use &lt;auto&gt; &amp; stay safe.", html)
        self.assertNotIn("<auto>", html)


if __name__ == "__main__":
    unittest.main()

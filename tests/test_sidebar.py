from __future__ import annotations

import unittest

from verilume.ui.sidebar import _active_model_html


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


if __name__ == "__main__":
    unittest.main()

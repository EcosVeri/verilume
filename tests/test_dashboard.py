from __future__ import annotations

import unittest
from unittest.mock import patch

from verilume.core.prompt_suggestions import PromptSuggestion
from verilume.ui.dashboard import (
    DEFAULT_DASHBOARD_COLLAPSED,
    recent_activity_from_messages,
    render_suggested_prompts,
)


class DashboardTests(unittest.TestCase):
    def test_dashboard_defaults_to_collapsed_expand_action(self) -> None:
        self.assertTrue(DEFAULT_DASHBOARD_COLLAPSED)

    def test_recent_activity_uses_latest_user_messages(self) -> None:
        messages = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Answer"},
            {"role": "user", "content": "Second"},
            {"role": "user", "content": "Third"},
            {"role": "assistant", "content": "Answer"},
        ]

        self.assertEqual(recent_activity_from_messages(messages, limit=2), ["Third", "Second"])

    def test_suggested_prompt_button_only_sets_pending_prompt(self) -> None:
        session_state: dict[str, str] = {}
        suggestions = [
            PromptSuggestion(
                title="Summarise uploaded documents",
                prompt="Summarise uploaded documents",
                category="collection",
                priority=0.9,
                document_id=None,
                document_type=None,
            )
        ]

        def fake_button(label: str, **_: object) -> bool:
            return label == "Summarise uploaded documents"

        with (
            patch("verilume.ui.dashboard.st.session_state", session_state),
            patch("verilume.ui.dashboard.st.markdown"),
            patch("verilume.ui.dashboard.st.button", side_effect=fake_button),
            patch("verilume.ui.dashboard.st.rerun", side_effect=RuntimeError("rerun")),
            self.assertRaises(RuntimeError),
        ):
            render_suggested_prompts(suggestions)

        self.assertEqual(session_state["pending_prompt"], "Summarise uploaded documents")


if __name__ == "__main__":
    unittest.main()

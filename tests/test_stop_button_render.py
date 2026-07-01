"""Real-rendering checks (via Streamlit AppTest) for the Stop button.

Guards the reported regression: the toolbar Stop button was frozen `disabled`
during a query because the toolbar renders in the main script, which does not
re-execute during the polling fragment's run_every ticks. The live Stop now
lives inside the fragment (which does re-render every tick).
"""

from __future__ import annotations

import unittest

from streamlit.testing.v1 import AppTest


def _app_script() -> None:
    import threading

    import streamlit as st

    from verilume.settings import AppSettings
    from verilume.ui.chat import (
        _GEN_STATE_KEY,
        _poll_generation_impl,
        _render_toolbar,
        init_chat_state,
    )

    init_chat_state()
    settings = AppSettings()

    if st.session_state.get("_test_mode") == "generating":
        st.session_state.generating = True
        st.session_state[_GEN_STATE_KEY] = {
            "thread": None,
            "stop_event": threading.Event(),
            "result": {"done": False, "stage": "Searching web evidence...", "request_id": "x"},
            "prompt": "Q",
            "request_id": "x",
        }
        _render_toolbar(settings)
        _poll_generation_impl(settings)
    else:
        _render_toolbar(settings)


class StopButtonRenderTests(unittest.TestCase):
    def test_idle_toolbar_has_no_stop_button(self) -> None:
        at = AppTest.from_function(_app_script)
        at.session_state["_test_mode"] = "idle"
        at.run(timeout=30)
        labels = [b.label for b in at.button]
        self.assertFalse(any("Stop" in lbl for lbl in labels), labels)

    def test_generating_shows_enabled_live_stop_button(self) -> None:
        at = AppTest.from_function(_app_script)
        at.session_state["_test_mode"] = "generating"
        at.run(timeout=30)
        stop_buttons = [b for b in at.button if "Stop generating" in b.label]
        self.assertEqual(len(stop_buttons), 1, [b.label for b in at.button])
        self.assertFalse(stop_buttons[0].disabled, "Live Stop button must be enabled")


if __name__ == "__main__":
    unittest.main()

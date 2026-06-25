from __future__ import annotations

import unittest
from pathlib import Path

from verilume.cli import (
    STREAMLIT_BROWSER_ADDRESS,
    STREAMLIT_HOST,
    STREAMLIT_PORT,
    _streamlit_cli_args,
)


class StreamlitLaunchTests(unittest.TestCase):
    def test_streamlit_args_pin_localhost_port(self) -> None:
        args = _streamlit_cli_args(Path("src/verilume/app.py"))

        self.assertEqual(args[:2], ["run", "src/verilume/app.py"])
        self.assertEqual(args[args.index("--server.address") + 1], STREAMLIT_HOST)
        self.assertEqual(args[args.index("--server.port") + 1], str(STREAMLIT_PORT))
        self.assertEqual(
            args[args.index("--browser.serverAddress") + 1],
            STREAMLIT_BROWSER_ADDRESS,
        )
        self.assertEqual(args[args.index("--server.fileWatcherType") + 1], "none")
        self.assertEqual(args[args.index("--browser.gatherUsageStats") + 1], "false")


if __name__ == "__main__":
    unittest.main()

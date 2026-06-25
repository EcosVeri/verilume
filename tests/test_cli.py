from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from verilume.cli import (
    STREAMLIT_BROWSER_ADDRESS,
    STREAMLIT_HOST,
    STREAMLIT_PORT,
    STREAMLIT_PORT_ENV,
    _first_available_port,
    _resolve_streamlit_port,
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

    def test_streamlit_args_accept_selected_port(self) -> None:
        args = _streamlit_cli_args(Path("src/verilume/app.py"), port=8512)

        self.assertEqual(args[args.index("--server.port") + 1], "8512")

    def test_first_available_port_skips_busy_ports(self) -> None:
        with mock.patch("verilume.cli._port_is_available", side_effect=[False, True]):
            self.assertEqual(_first_available_port(STREAMLIT_HOST, STREAMLIT_PORT), STREAMLIT_PORT + 1)

    def test_resolve_streamlit_port_uses_environment_override(self) -> None:
        with mock.patch.dict("os.environ", {STREAMLIT_PORT_ENV: "8525"}):
            self.assertEqual(_resolve_streamlit_port(), 8525)

    def test_resolve_streamlit_port_reclaims_default_when_preferred_ports_are_busy(self) -> None:
        with (
            mock.patch.dict("os.environ", {}, clear=True),
            mock.patch("verilume.cli._first_available_port", return_value=None),
            mock.patch("verilume.cli._force_reclaim_streamlit_port", return_value=STREAMLIT_PORT) as reclaim,
        ):
            self.assertEqual(_resolve_streamlit_port(), STREAMLIT_PORT)
            reclaim.assert_called_once_with(STREAMLIT_HOST, STREAMLIT_PORT)


if __name__ == "__main__":
    unittest.main()

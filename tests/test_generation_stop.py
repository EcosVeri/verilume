from __future__ import annotations

import threading
import time
import unittest

from verilume.core.generation import BaseGenerator, GenerationStopped
from verilume.settings import AppSettings


class _StubGenerator(BaseGenerator):
    """Concrete BaseGenerator so we can exercise _call_with_stop directly."""

    def chat(self, messages: list[dict[str, str]]) -> str:  # pragma: no cover - unused
        return ""


class CallWithStopTests(unittest.TestCase):
    def test_returns_result_when_no_stop_hook(self) -> None:
        gen = _StubGenerator(AppSettings())
        self.assertEqual(gen._call_with_stop(lambda: "ok"), "ok")

    def test_returns_result_when_not_stopped(self) -> None:
        gen = _StubGenerator(AppSettings())
        gen._should_stop = lambda: False
        self.assertEqual(gen._call_with_stop(lambda: 42), 42)

    def test_propagates_exception_from_call(self) -> None:
        gen = _StubGenerator(AppSettings())
        gen._should_stop = lambda: False

        def boom() -> str:
            raise ValueError("nope")

        with self.assertRaises(ValueError):
            gen._call_with_stop(boom)

    def test_stop_returns_quickly_while_call_blocks(self) -> None:
        gen = _StubGenerator(AppSettings())
        stop = threading.Event()
        gen._should_stop = stop.is_set

        def slow_call() -> str:
            time.sleep(5.0)  # simulate a blocking, non-cancellable network request
            return "late"

        # Request the stop before the call would ever finish.
        stop.set()
        started = time.perf_counter()
        with self.assertRaises(GenerationStopped):
            gen._call_with_stop(slow_call)
        elapsed = time.perf_counter() - started
        # Must abandon the blocked call, not wait ~5s for it.
        self.assertLess(elapsed, 1.0)


if __name__ == "__main__":
    unittest.main()

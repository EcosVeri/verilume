from __future__ import annotations

import unittest

from verilume.core.schemas import LocalSource, RAGResponse, WebSource
from verilume.rag import _STABLE_DIAGNOSTIC_FLAGS, _ensure_stable_diagnostics


def _response(**kwargs) -> RAGResponse:
    base = dict(
        answer="a",
        local_sources=[],
        web_sources=[],
        used_web=False,
        confidence="medium",
        diagnostics={},
    )
    base.update(kwargs)
    return RAGResponse(**base)


class StableDiagnosticsTests(unittest.TestCase):
    def test_all_flags_present_as_booleans_when_diagnostics_empty(self) -> None:
        response = _ensure_stable_diagnostics(_response(diagnostics={}))
        for key in _STABLE_DIAGNOSTIC_FLAGS:
            self.assertIn(key, response.diagnostics)
            self.assertIsInstance(response.diagnostics[key], bool)

    def test_existing_values_are_preserved_and_coerced(self) -> None:
        response = _ensure_stable_diagnostics(
            _response(diagnostics={"used_local": 1, "used_web": 0})
        )
        self.assertIs(response.diagnostics["used_local"], True)
        self.assertIs(response.diagnostics["used_web"], False)

    def test_flags_inferred_from_response_when_missing(self) -> None:
        response = _ensure_stable_diagnostics(
            _response(
                local_sources=[
                    LocalSource(
                        label="S1", document="d", page=1, chunk_id="c1", text="t", score=0.9
                    )
                ],
                web_sources=[WebSource(label="W1", title="t", url="https://x", content="c")],
                used_web=True,
                diagnostics={},
            )
        )
        self.assertTrue(response.diagnostics["used_local"])
        self.assertTrue(response.diagnostics["used_web"])

    def test_none_values_are_replaced_not_kept(self) -> None:
        response = _ensure_stable_diagnostics(
            _response(diagnostics={"local_sufficient": None})
        )
        self.assertIsInstance(response.diagnostics["local_sufficient"], bool)


if __name__ == "__main__":
    unittest.main()

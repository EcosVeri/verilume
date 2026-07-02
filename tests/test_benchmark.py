from __future__ import annotations

from verilume.core.benchmark import (
    AI_ONLY,
    FULL,
    LOCAL_ONLY,
    WEB_ONLY,
    BenchmarkReport,
    choose_best_mode,
    make_benchmark_result,
)
from verilume.core.schemas import LocalSource, RAGResponse, WebSource
from verilume.rag import _benchmark_mode_settings, _search_mode_allows_local, _search_mode_key
from verilume.settings import AppSettings


def test_benchmark_result_counts_sources_and_latency() -> None:
    response = RAGResponse(
        answer="Local and web answer [S1] [W1]",
        local_sources=[
            LocalSource("S1", "doc.pdf", 1, "chunk-1", "Local text", 0.92),
        ],
        web_sources=[
            WebSource("W1", "Web", "https://example.com", "Web text", 0.88),
        ],
        used_web=True,
        confidence="high",
        diagnostics={"answer_verification_score": 0.84},
    )

    result = make_benchmark_result(FULL, response, 1.25)

    assert result.source_count == 2
    assert result.local_source_count == 1
    assert result.web_source_count == 1
    assert result.latency_seconds == 1.25
    assert result.faithfulness_score == 0.84


def test_benchmark_report_converts_to_rag_response() -> None:
    local = make_benchmark_result(
        LOCAL_ONLY,
        RAGResponse("Local answer [S1]", [LocalSource("S1", "doc.pdf", 1, "c1", "x", 1.0)], [], False, "local-grounded"),
        0.2,
    )
    ai = make_benchmark_result(
        AI_ONLY,
        RAGResponse("AI answer", [], [], False, "model-only"),
        0.1,
    )
    report = BenchmarkReport("Question?", [local, ai], choose_best_mode([local, ai]))

    response = report.to_rag_response()

    assert "Benchmark Results" in response.answer
    assert response.diagnostics["benchmark_mode"] is True
    assert response.diagnostics["benchmark_report"]["best_mode"] == LOCAL_ONLY


def test_choose_best_mode_declares_no_winner_when_all_modes_fail() -> None:
    # The screenshot bug: "best: Local" shown under "I could not answer..." —
    # with every mode low-confidence and sourceless, the latency tiebreak was
    # crowning the fastest failure.
    failed = [
        make_benchmark_result(
            mode,
            RAGResponse("I could not answer from local files.", [], [], False, "low"),
            0.1 * (index + 1),
        )
        for index, mode in enumerate((FULL, LOCAL_ONLY, AI_ONLY, WEB_ONLY))
    ]

    assert choose_best_mode(failed) == ""

    report = BenchmarkReport("Question?", failed, "", notes=["No mode produced a grounded answer for this question."])
    answer = report.to_rag_response().answer
    assert "No mode produced a grounded answer" in answer
    assert "Best mode:" not in answer


def test_choose_best_mode_picks_only_among_grounded_results() -> None:
    failed_local = make_benchmark_result(
        LOCAL_ONLY,
        RAGResponse("I could not answer.", [], [], False, "low"),
        0.01,  # fastest — must not win on latency
    )
    grounded_web = make_benchmark_result(
        WEB_ONLY,
        RAGResponse(
            "Web answer [W1]",
            [],
            [WebSource("W1", "Web", "https://example.com", "text", 0.8)],
            True,
            "web-assisted",
        ),
        2.0,
    )

    assert choose_best_mode([failed_local, grounded_web]) == WEB_ONLY


def test_benchmark_mode_settings_isolate_strategies() -> None:
    settings = AppSettings(benchmark_mode=True, semantic_cache_enabled=True, enable_web_search=True)
    modes = dict(_benchmark_mode_settings(settings))

    assert set(modes) == {FULL, LOCAL_ONLY, AI_ONLY, WEB_ONLY}
    assert all(not mode_settings.benchmark_mode for mode_settings in modes.values())
    assert all(not mode_settings.semantic_cache_enabled for mode_settings in modes.values())
    assert _search_mode_key(modes[AI_ONLY]) == "ai_only"
    assert _search_mode_allows_local("ai_only") is False
    assert modes[LOCAL_ONLY].enable_web_search is False

# Tests for the Phase D.2 LLM narrative generator.
# No real API calls: a fake client verifies the happy path; the no-key path and
# the failure path verify graceful skipping.

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.report import llm_narrative
from src.report.llm_narrative import (
    _system_prompt,
    _user_content,
    generate_narrative,
)
from src.report.structured_report import DiagnosisEntry, ECGReport


def _make_report() -> ECGReport:
    return ECGReport(
        heart_rate_bpm=72.0,
        rhythm_classification="Sinus rhythm",
        rhythm_basis="sinus",
        rate_category="normal",
        ectopy_present=False,
        pvc_count=0,
        pr_ms=150.0,
        qrs_ms=90.0,
        qt_ms=380.0,
        qtc_ms=400.0,
        qtc_formula="bazett",
        interval_flags={"pr": "normal", "qrs": "normal", "qtc": "normal"},
        top_diagnoses=[
            DiagnosisEntry(label="NORMAL ECG", probability=0.91, above_threshold=True),
        ],
        is_normal=True,
        overall_assessment="Normal ECG",
        abnormal_reasons=[],
        normal_criteria={"sinus_rhythm": True},
    )


@dataclass
class _FakeTextBlock:
    type: str
    text: str


@dataclass
class _FakeResponse:
    content: list


class _FakeMessages:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.last_kwargs: dict | None = None

    def create(self, **kwargs: object) -> _FakeResponse:
        self.last_kwargs = kwargs
        return self._response


class _FakeClient:
    def __init__(self, response: _FakeResponse) -> None:
        self.messages = _FakeMessages(response)


def test_system_prompt_localizes_language() -> None:
    assert "Turkish" in _system_prompt("tr")
    assert "English" in _system_prompt("en")
    # Unknown code falls back to English in generate_narrative, but the prompt
    # builder defaults unknown names to English too.
    assert "English" in _system_prompt("xx")


def test_user_content_contains_findings() -> None:
    content = _user_content(_make_report())
    assert "NORMAL ECG" in content
    assert "Sinus rhythm" in content


def test_generate_narrative_with_fake_client() -> None:
    response = _FakeResponse(
        content=[_FakeTextBlock(type="text", text="Sinüs ritmi, hız 72/dk. Normal EKG.")]
    )
    client = _FakeClient(response)

    result = generate_narrative(_make_report(), language="tr", client=client)

    assert result == "Sinüs ritmi, hız 72/dk. Normal EKG."
    assert client.messages.last_kwargs is not None
    assert client.messages.last_kwargs["model"] == "claude-opus-4-8"


def test_generate_narrative_skips_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_narrative, "load_dotenv", lambda: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert generate_narrative(_make_report(), language="tr") is None


def test_generate_narrative_returns_none_on_api_error() -> None:
    class _BrokenClient:
        class messages:  # noqa: N801
            @staticmethod
            def create(**kwargs: object) -> object:
                raise RuntimeError("boom")

    result = generate_narrative(_make_report(), language="en", client=_BrokenClient())
    assert result is None


def test_generate_narrative_handles_empty_content() -> None:
    client = _FakeClient(_FakeResponse(content=[]))
    assert generate_narrative(_make_report(), language="en", client=client) is None

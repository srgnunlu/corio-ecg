# LLM natural-language narrative for the structured ECG report (Phase D.2).
#
# Turns the deterministic ECGReport (HR, rhythm, intervals, AI diagnoses) into a
# short, readable clinical summary via the Claude API. This is a SUMMARIZER, not
# a diagnostic engine: the system prompt forbids inventing findings, adding
# diagnoses not already in the report, or making clinical decisions. If the API
# key is missing or the call fails, generation is skipped (returns None) and the
# rest of the report still works — the narrative is always optional.
#
# Security: only the structured findings (rates, intervals, diagnosis labels) are
# sent to the API. No patient identifiers / PHI are included.

from __future__ import annotations

import json
import logging
import os

from dotenv import load_dotenv

from src.report.structured_report import ECGReport, report_to_dict

logger = logging.getLogger(__name__)

# Sonnet 4.6 — fast and cost-efficient. Adaptive thinking is unnecessary here:
# this is short, grounded summarization (not reasoning) that runs automatically
# on every analysis, so we favor Sonnet's speed/cost and leave thinking off.
NARRATIVE_MODEL: str = "claude-sonnet-4-6"
_MAX_TOKENS: int = 1024

SUPPORTED_LANGUAGES: dict[str, str] = {"tr": "Turkish", "en": "English"}

_SYSTEM_PROMPT = """\
You are a careful medical scribe summarizing a finished ECG analysis for a \
physician. You are NOT diagnosing — every finding has already been produced by a \
deterministic measurement pipeline and a diagnosis model. Your only job is to turn \
the provided structured findings into a clear, professional prose summary.

Hard rules:
- Use ONLY the findings in the provided JSON. Never introduce a diagnosis, rhythm, \
or measurement that is not present.
- If a value is null or marked unmeasurable, say it could not be measured — do not \
guess or infer it.
- Do not give treatment advice, management plans, or a final clinical decision.
- Be concise and factual. No hedging filler, no invented numbers.
- Write the entire response in {language}.

Structure the summary as 3 short paragraphs:
1. Overall verdict and heart rate / rhythm.
2. Interval measurements (PR, QRS, QT, QTc) and any flagged abnormalities.
3. The model's top diagnoses, noting which are above the reporting threshold.

End with one sentence: this is an AI-generated research summary, not a clinical \
decision, and a clinician must verify it. Output plain text only — no markdown \
headers, no bullet symbols."""


def _system_prompt(language: str) -> str:
    lang_name = SUPPORTED_LANGUAGES.get(language, "English")
    return _SYSTEM_PROMPT.format(language=lang_name)


def _user_content(report: ECGReport) -> str:
    """The structured findings the model is allowed to describe (JSON)."""
    return (
        "Summarize this ECG analysis. Findings JSON:\n\n"
        + json.dumps(report_to_dict(report), ensure_ascii=False, indent=2)
    )


def narrative_available() -> bool:
    """True if an Anthropic API key is configured (so generation can run)."""
    load_dotenv()
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def generate_narrative(
    report: ECGReport,
    *,
    language: str = "tr",
    client: object | None = None,
) -> str | None:
    """Generate a natural-language summary of the report, or None on failure.

    Best-effort by design: a missing API key, a network error, or any API
    failure returns None instead of raising, so the report path never breaks.

    Args:
        report: the assembled structured report.
        language: "tr" or "en" (falls back to English for unknown codes).
        client: optional pre-built Anthropic client (used in tests); when None a
            client is created from the environment.
    """
    load_dotenv()
    if not os.getenv("ANTHROPIC_API_KEY") and client is None:
        logger.info("ANTHROPIC_API_KEY not set — skipping LLM narrative.")
        return None

    if language not in SUPPORTED_LANGUAGES:
        language = "en"

    try:
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        response = client.messages.create(  # type: ignore[attr-defined]
            model=NARRATIVE_MODEL,
            max_tokens=_MAX_TOKENS,
            system=_system_prompt(language),
            messages=[{"role": "user", "content": _user_content(report)}],
        )
    except Exception:  # noqa: BLE001 — narrative must never break the report path
        logger.exception("LLM narrative generation failed")
        return None

    return _extract_text(response)


def _extract_text(response: object) -> str | None:
    """Pull the concatenated text blocks out of a Messages API response."""
    content = getattr(response, "content", None)
    if not content:
        return None
    parts = [
        block.text
        for block in content
        if getattr(block, "type", None) == "text" and getattr(block, "text", None)
    ]
    text = "\n".join(parts).strip()
    return text or None

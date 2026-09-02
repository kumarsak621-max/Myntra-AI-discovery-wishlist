"""Pipeline step semantics. Insights cannot be SUCCESS when analysis is PARTIAL/FAILED."""

from __future__ import annotations

from typing import Any


def insights_status_for_analyze(analyze_step: str, analyzed: int) -> str:
    """Map analyze outcome + stored analyzed count to an Insights stage."""
    analyzed = int(analyzed or 0)
    step = (analyze_step or "").strip()
    if step == "partial":
        return "partial" if analyzed > 0 else "insufficient"
    if step == "failed":
        return "failed" if analyzed <= 0 else "partial"
    if step == "insufficient":
        return "insufficient"
    if step == "no_new":
        return "done" if analyzed > 0 else "insufficient"
    if step == "pending":
        return "pending"
    if analyzed <= 0:
        return "insufficient"
    return "done"


def derive_failed_reason(
    *,
    steps: dict[str, Any] | None,
    last_analysis: dict[str, Any] | None = None,
    last_collection: dict[str, Any] | None = None,
    step4_error: dict[str, Any] | None = None,
    pipeline_result: dict[str, Any] | None = None,
) -> str | None:
    """Always defined. None when there is no stage failure to display."""
    if pipeline_result and pipeline_result.get("failed_reason"):
        text = str(pipeline_result.get("failed_reason") or "").strip()
        return text or None
    steps = steps or {}
    analyze = steps.get("analyze")
    if analyze in {"failed", "partial"}:
        err = (step4_error or {}).get("error") or (last_analysis or {}).get("message")
        if err:
            return str(err).strip() or None
    if steps.get("play") == "failed" or steps.get("apple") == "failed":
        errors = ((last_collection or {}).get("stats") or {}).get("errors")
        if isinstance(errors, list) and errors:
            return "; ".join(str(item) for item in errors[:3] if str(item).strip())
        if isinstance(errors, str) and errors.strip():
            return errors.strip()
    return None

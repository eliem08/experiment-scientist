"""Pluggable, auditable evaluators for Experiment Scientist.

Built-in deterministic evaluators are transparent scoring rules. The judge adapter
is provider-agnostic and requires an injected callable; this module never performs
network requests or loads a model by itself.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Protocol


EVALUATOR_API_VERSION = "experiment-scientist.evaluator.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EvaluationInput:
    """One item to score. Keep case and pairing IDs stable across conditions."""

    case_id: str
    response: str
    reference: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluatorDefinition:
    evaluator_id: str
    version: str
    kind: str
    direction: str
    description: str
    limitations: tuple[str, ...] = ()


@dataclass
class EvaluationResult:
    schema_version: str
    evaluator_id: str
    evaluator_version: str
    evaluator_kind: str
    case_id: str
    score: float | None
    label: str | None
    status: str
    direction: str
    rationale: str | None = None
    raw_decision: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    created_at_utc: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe data; reject non-finite scores rather than emitting NaN."""
        payload = asdict(self)
        if self.score is not None and not math.isfinite(self.score):
            raise ValueError("Evaluation score must be finite")
        json.dumps(payload, ensure_ascii=False, allow_nan=False)
        return payload


class Evaluator(Protocol):
    """Structural interface implemented by all evaluator plugins."""

    definition: EvaluatorDefinition

    def evaluate(self, item: EvaluationInput) -> EvaluationResult:
        """Score a single response and return an auditable result."""


class _BaseEvaluator:
    definition: EvaluatorDefinition

    def _result(
        self, item: EvaluationInput, score: float, *, label: str | None = None,
        rationale: str | None = None, metadata: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        if not math.isfinite(score):
            raise ValueError("Evaluator returned a non-finite score")
        return EvaluationResult(
            schema_version=EVALUATOR_API_VERSION,
            evaluator_id=self.definition.evaluator_id,
            evaluator_version=self.definition.version,
            evaluator_kind=self.definition.kind,
            case_id=item.case_id,
            score=float(score),
            label=label,
            status="completed",
            direction=self.definition.direction,
            rationale=rationale,
            metadata=metadata or {},
        )


class ExactMatchEvaluator(_BaseEvaluator):
    """Score exact string equality after optional whitespace trimming."""

    definition = EvaluatorDefinition(
        "exact-match", "1.0.0", "rule_based", "higher",
        "Exact string equality against a reference.",
        ("Does not allow paraphrases or semantic equivalence.",),
    )

    def __init__(self, *, strip_whitespace: bool = True, case_sensitive: bool = True):
        self.strip_whitespace = strip_whitespace
        self.case_sensitive = case_sensitive

    def evaluate(self, item: EvaluationInput) -> EvaluationResult:
        if not isinstance(item.reference, str):
            raise ValueError("exact-match requires a string reference")
        actual, expected = item.response, item.reference
        if self.strip_whitespace:
            actual, expected = actual.strip(), expected.strip()
        if not self.case_sensitive:
            actual, expected = actual.casefold(), expected.casefold()
        matched = actual == expected
        return self._result(item, float(matched), label="match" if matched else "mismatch")


class ContainsEvaluator(_BaseEvaluator):
    """Score whether a declared substring appears in the response."""

    definition = EvaluatorDefinition(
        "contains", "1.0.0", "rule_based", "higher",
        "Case-sensitive or case-insensitive substring presence.",
        ("Lexical presence is not semantic correctness.",),
    )

    def __init__(self, expected: str, *, case_sensitive: bool = False):
        if not isinstance(expected, str) or not expected:
            raise ValueError("expected substring must be a non-empty string")
        self.expected = expected
        self.case_sensitive = case_sensitive

    def evaluate(self, item: EvaluationInput) -> EvaluationResult:
        actual, expected = item.response, self.expected
        if not self.case_sensitive:
            actual, expected = actual.casefold(), expected.casefold()
        matched = expected in actual
        return self._result(item, float(matched), label="contains" if matched else "missing",
                            metadata={"expected_sha256": hashlib.sha256(self.expected.encode("utf-8")).hexdigest()})


class RegexMatchEvaluator(_BaseEvaluator):
    """Score whether a trusted, predeclared regular expression matches."""

    definition = EvaluatorDefinition(
        "regex-match", "1.0.0", "rule_based", "higher",
        "Whether a predeclared regular expression matches a response.",
        ("A regex match does not establish factual validity.",),
    )

    def __init__(self, pattern: str, *, flags: int = re.MULTILINE):
        try:
            self.pattern = re.compile(pattern, flags)
        except re.error as exc:
            raise ValueError(f"Invalid evaluator regex: {exc}") from exc
        self.pattern_text = pattern

    def evaluate(self, item: EvaluationInput) -> EvaluationResult:
        matched = self.pattern.search(item.response) is not None
        return self._result(item, float(matched), label="match" if matched else "no_match",
                            metadata={"pattern_sha256": hashlib.sha256(self.pattern_text.encode("utf-8")).hexdigest()})


class JSONFieldEqualsEvaluator(_BaseEvaluator):
    """Compare a JSON object's selected field to a declared reference value."""

    definition = EvaluatorDefinition(
        "json-field-equals", "1.0.0", "task_specific", "higher",
        "Parses a JSON response and compares one field with a reference value.",
        ("Only validates the selected field and not other response properties.",),
    )

    def __init__(self, field_path: str):
        parts = tuple(part for part in field_path.split(".") if part)
        if not parts:
            raise ValueError("field_path must contain at least one non-empty segment")
        self.field_path = parts

    def evaluate(self, item: EvaluationInput) -> EvaluationResult:
        if not isinstance(item.reference, Mapping):
            raise ValueError("json-field-equals requires a mapping reference")
        try:
            actual: Any = json.loads(item.response)
            expected: Any = item.reference
            for part in self.field_path:
                actual = actual[part]
                expected = expected[part]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            return self._result(item, 0.0, label="invalid_or_missing",
                                rationale=f"Could not parse response or resolve field: {type(exc).__name__}",
                                metadata={"field_path": ".".join(self.field_path)})
        matched = actual == expected
        return self._result(item, float(matched), label="match" if matched else "mismatch",
                            metadata={"field_path": ".".join(self.field_path)})


JudgeCallable = Callable[[str, Mapping[str, Any]], str | Mapping[str, Any]]


class LLMJudgeEvaluator(_BaseEvaluator):
    """Provider-neutral judge adapter with bounded retries and auditable decisions.

    The caller supplies a callable that accepts (rendered_prompt, context) and
    returns JSON text or a mapping. Network access, credentials, rate limits, and
    provider-specific behavior remain outside this module.
    """

    def __init__(
        self, *, judge_id: str, judge_revision: str | None, rubric: str,
        judge_call: JudgeCallable, max_retries: int = 2, minimum_score: float = 0.0,
        maximum_score: float = 1.0, direction: str = "higher",
    ):
        if not judge_id.strip():
            raise ValueError("judge_id must be non-empty")
        if not rubric.strip():
            raise ValueError("rubric must be non-empty")
        if max_retries < 0 or max_retries > 10:
            raise ValueError("max_retries must be between 0 and 10")
        if not math.isfinite(minimum_score) or not math.isfinite(maximum_score) or minimum_score >= maximum_score:
            raise ValueError("score bounds must be finite and minimum_score < maximum_score")
        if direction not in {"higher", "lower", "neutral"}:
            raise ValueError("direction must be higher, lower, or neutral")
        self.judge_id = judge_id
        self.judge_revision = judge_revision
        self.rubric = rubric
        self.judge_call = judge_call
        self.max_retries = max_retries
        self.minimum_score = minimum_score
        self.maximum_score = maximum_score
        self.direction = direction
        self.definition = EvaluatorDefinition(
            "llm-judge", "1.0.0", "llm_judge", direction,
            "Provider-neutral rubric-based model judgment.",
            ("Judgments may be biased, prompt-sensitive, and disagree with human raters.",
             "A judge score is not ground truth without external validation."),
        )

    def evaluate(self, item: EvaluationInput) -> EvaluationResult:
        prompt = (
            f"Rubric:\n{self.rubric}\n\n"
            "Evaluate the response using only the rubric. Return one JSON object with "
            "keys score (number within the declared range), label (string), and rationale (string).\n"
            f"Case ID: {item.case_id}\nResponse:\n{item.response}\n"
            f"Reference (JSON): {json.dumps(item.reference, ensure_ascii=False, default=str)}"
        )
        context = {
            "case_id": item.case_id,
            "judge_id": self.judge_id,
            "judge_revision": self.judge_revision,
            "rubric_sha256": hashlib.sha256(self.rubric.encode("utf-8")).hexdigest(),
        }
        last_error = "judge call did not run"
        raw: Any = None
        for attempt in range(self.max_retries + 1):
            try:
                raw = self.judge_call(prompt, {**context, "attempt": attempt + 1})
                decision = json.loads(raw) if isinstance(raw, str) else raw
                if not isinstance(decision, Mapping):
                    raise ValueError("judge decision must be a JSON object")
                score = decision.get("score")
                if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
                    raise ValueError("score must be a finite number")
                if not self.minimum_score <= float(score) <= self.maximum_score:
                    raise ValueError("score is outside the declared range")
                label, rationale = decision.get("label"), decision.get("rationale")
                if not isinstance(label, str) or not isinstance(rationale, str):
                    raise ValueError("label and rationale must be strings")
                result = self._result(
                    item, float(score), label=label, rationale=rationale,
                    metadata={
                        **context, "score_min": self.minimum_score, "score_max": self.maximum_score,
                        "judge_decision_sha256": _canonical_hash(decision),
                    },
                )
                result.raw_decision = dict(decision)
                return result
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                last_error = str(exc)
            except Exception as exc:  # provider adapter failures become explicit failed rows
                last_error = f"judge adapter error: {type(exc).__name__}: {exc}"
        return EvaluationResult(
            schema_version=EVALUATOR_API_VERSION,
            evaluator_id=self.definition.evaluator_id,
            evaluator_version=self.definition.version,
            evaluator_kind=self.definition.kind,
            case_id=item.case_id,
            score=None,
            label=None,
            status="failed",
            direction=self.direction,
            rationale=last_error,
            raw_decision=raw,
            metadata={**context, "max_attempts": self.max_retries + 1},
            warnings=["Judge failure is missing data; it must not be converted to a zero score."],
        )


def validate_human_annotation(annotation: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one human annotation without assuming it is correct."""
    required = ("case_id", "annotator_id", "label", "rubric_version")
    missing = [key for key in required if not isinstance(annotation.get(key), str) or not annotation[key].strip()]
    if missing:
        raise ValueError("Human annotation missing non-empty fields: " + ", ".join(missing))
    score = annotation.get("score")
    if score is not None and (isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score))):
        raise ValueError("annotation score must be a finite number or null")
    result = dict(annotation)
    result["schema_version"] = "experiment-scientist.human-annotation.v1"
    result["created_at_utc"] = result.get("created_at_utc") or _utc_now()
    result["score"] = float(score) if score is not None else None
    json.dumps(result, ensure_ascii=False, allow_nan=False)
    return result

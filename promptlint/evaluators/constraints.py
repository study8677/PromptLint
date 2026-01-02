from __future__ import annotations

import json
import re
from typing import Any, Dict, Tuple

from promptlint.core.types import Constraint
from promptlint.utils.text import non_empty_lines


class ConstraintResult:
    def __init__(self, score: float, details: Dict[str, Any]) -> None:
        self.score = score
        self.details = details


def evaluate_constraint(text: str, constraint: Constraint) -> ConstraintResult:
    rules = constraint.rules or {}
    if rules:
        return _evaluate_rules(text, rules)
    return _heuristic_constraint(text, constraint)


def _evaluate_rules(text: str, rules: Dict[str, Any]) -> ConstraintResult:
    rule_type = rules.get("type")
    if rule_type == "count":
        return _rule_count(text, rules)
    if rule_type == "all_lines_match":
        return _rule_all_lines_match(text, rules)
    if rule_type == "regex":
        return _rule_regex(text, rules)
    if rule_type == "json":
        return _rule_json(text, rules)
    if rule_type == "length":
        return _rule_length(text, rules)
    if rule_type == "contains":
        return _rule_contains(text, rules, negate=False)
    if rule_type == "not_contains":
        return _rule_contains(text, rules, negate=True)

    return ConstraintResult(
        0.0,
        {"status": "unknown_rule", "rule_type": rule_type},
    )


def _rule_count(text: str, rules: Dict[str, Any]) -> ConstraintResult:
    pattern = rules.get("pattern", ".*")
    regex = re.compile(pattern)
    lines = non_empty_lines(text)
    count = sum(1 for line in lines if regex.search(line))

    exact = rules.get("exact")
    minimum = rules.get("min")
    maximum = rules.get("max")

    passed = True
    if exact is not None:
        passed = count == int(exact)
    if minimum is not None:
        passed = passed and count >= int(minimum)
    if maximum is not None:
        passed = passed and count <= int(maximum)

    score = 1.0 if passed else 0.0
    return ConstraintResult(
        score,
        {"status": "ok" if passed else "fail", "count": count},
    )


def _rule_all_lines_match(text: str, rules: Dict[str, Any]) -> ConstraintResult:
    pattern = rules.get("pattern", ".*")
    regex = re.compile(pattern)
    lines = non_empty_lines(text)
    if not lines:
        return ConstraintResult(0.0, {"status": "empty"})
    matched = sum(1 for line in lines if regex.search(line))
    passed = matched == len(lines)
    score = 1.0 if passed else matched / len(lines)
    return ConstraintResult(
        score,
        {"status": "ok" if passed else "partial", "matched": matched, "total": len(lines)},
    )


def _rule_regex(text: str, rules: Dict[str, Any]) -> ConstraintResult:
    pattern = rules.get("pattern", "")
    if not pattern:
        return ConstraintResult(0.0, {"status": "missing_pattern"})
    regex = re.compile(pattern, re.DOTALL)
    passed = bool(regex.search(text))
    return ConstraintResult(1.0 if passed else 0.0, {"status": "ok" if passed else "fail"})


def _rule_json(text: str, rules: Dict[str, Any]) -> ConstraintResult:
    try:
        payload = json.loads(text)
    except Exception:
        return ConstraintResult(0.0, {"status": "invalid_json"})

    expect = rules.get("expect")
    if expect == "object" and not isinstance(payload, dict):
        return ConstraintResult(0.0, {"status": "not_object"})
    if expect == "array" and not isinstance(payload, list):
        return ConstraintResult(0.0, {"status": "not_array"})

    return ConstraintResult(1.0, {"status": "ok"})


def _rule_length(text: str, rules: Dict[str, Any]) -> ConstraintResult:
    unit = rules.get("unit", "chars")
    if unit == "words":
        value = len(re.findall(r"\b\w+\b", text))
    elif unit == "lines":
        value = len(non_empty_lines(text))
    else:
        value = len(text)

    minimum = rules.get("min")
    maximum = rules.get("max")
    passed = True
    if minimum is not None:
        passed = passed and value >= int(minimum)
    if maximum is not None:
        passed = passed and value <= int(maximum)

    score = 1.0 if passed else 0.0
    return ConstraintResult(score, {"status": "ok" if passed else "fail", "value": value})


def _rule_contains(text: str, rules: Dict[str, Any], negate: bool) -> ConstraintResult:
    terms = rules.get("terms")
    if not terms:
        term = rules.get("term")
        terms = [term] if term else []
    matches = [term for term in terms if term and term in text]
    passed = bool(matches)
    if negate:
        passed = not passed
    score = 1.0 if passed else 0.0
    return ConstraintResult(
        score,
        {
            "status": "ok" if passed else "fail",
            "matches": matches,
            "negate": negate,
        },
    )


def _heuristic_constraint(text: str, constraint: Constraint) -> ConstraintResult:
    desc = (constraint.description or "").lower()
    if "json" in desc:
        return _rule_json(text, {"expect": "object"})
    if "bullet" in desc or "list" in desc:
        return _rule_count(text, {"pattern": r"^\s*[-*]\s+", "min": 1})
    if "number" in desc or "step" in desc:
        return _rule_count(text, {"pattern": r"^\s*\d+\.\s+", "min": 1})
    return ConstraintResult(0.5, {"status": "heuristic_default"})

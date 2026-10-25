"""Offline evaluation helpers for labeled grounded-synthesis examples."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .evidence import synthesize_retrieved_evidence
from .provider import LLMProvider

STANCES = ("bearish", "bullish", "mixed", "neutral")


def _classification_metrics(expected: list[str], predicted: list[str]) -> dict[str, Any]:
    per_stance = {}
    for stance in STANCES:
        true_positive = sum(actual == stance and guess == stance for actual, guess in zip(expected, predicted))
        predicted_count = sum(guess == stance for guess in predicted)
        support = sum(actual == stance for actual in expected)
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_stance[stance] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return {
        "accuracy": sum(actual == guess for actual, guess in zip(expected, predicted)) / len(expected),
        "macro_f1": sum(item["f1"] for item in per_stance.values()) / len(STANCES),
        "per_stance": per_stance,
    }


def evaluate_llm_synthesis_dataset(
    payload: Mapping[str, Any],
    provider: LLMProvider,
) -> dict[str, Any]:
    """Evaluate model stance and cited evidence against human-labeled cases."""
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("At least one labeled synthesis case is required")
    if provider is None:
        raise ValueError("An explicit LLM provider is required for synthesis evaluation")

    expected_stances = []
    predicted_stances = []
    citation_true_positive = 0
    citation_false_positive = 0
    citation_false_negative = 0
    per_case = []

    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise ValueError(f"Synthesis case {index} must be an object")
        ticker = str(case.get("ticker") or "").strip().upper()
        expected_stance = case.get("expected_stance")
        evidence = case.get("evidence")
        relevant_ids = case.get("relevant_evidence_ids")
        if not ticker:
            raise ValueError(f"Synthesis case {index} requires a ticker")
        if expected_stance not in STANCES:
            raise ValueError(f"Synthesis case {index} has an invalid expected_stance")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError(f"Synthesis case {index} requires timestamped evidence")
        if any(not isinstance(item, Mapping) for item in evidence):
            raise ValueError(f"Synthesis case {index} evidence items must be objects")
        if not isinstance(relevant_ids, list) or not relevant_ids:
            raise ValueError(f"Synthesis case {index} requires relevant_evidence_ids")

        supplied_ids = {str(item.get("chunk_id") or "") for item in evidence}
        gold_ids = {value for value in relevant_ids if isinstance(value, str) and value}
        if len(gold_ids) != len(relevant_ids) or not gold_ids.issubset(supplied_ids):
            raise ValueError(f"Synthesis case {index} has unknown or invalid relevant evidence IDs")

        result = synthesize_retrieved_evidence(
            ticker=ticker,
            evidence=evidence,
            as_of=case.get("as_of"),
            provider=provider,
        )
        cited_ids = set(result["summary_evidence_ids"])
        for claim in result["claims"]:
            cited_ids.update(claim["evidence_ids"])
        true_positive = len(cited_ids & gold_ids)
        false_positive = len(cited_ids - gold_ids)
        false_negative = len(gold_ids - cited_ids)
        citation_true_positive += true_positive
        citation_false_positive += false_positive
        citation_false_negative += false_negative
        expected_stances.append(expected_stance)
        predicted_stances.append(result["stance"])
        per_case.append({
            "ticker": ticker,
            "as_of": result["as_of"],
            "expected_stance": expected_stance,
            "predicted_stance": result["stance"],
            "stance_correct": result["stance"] == expected_stance,
            "citation_precision": (
                true_positive / (true_positive + false_positive)
                if true_positive + false_positive else 0.0
            ),
            "citation_recall": (
                true_positive / (true_positive + false_negative)
                if true_positive + false_negative else 0.0
            ),
        })

    stance_metrics = _classification_metrics(expected_stances, predicted_stances)
    citation_precision = (
        citation_true_positive / (citation_true_positive + citation_false_positive)
        if citation_true_positive + citation_false_positive else 0.0
    )
    citation_recall = (
        citation_true_positive / (citation_true_positive + citation_false_negative)
        if citation_true_positive + citation_false_negative else 0.0
    )
    citation_f1 = (
        2 * citation_precision * citation_recall / (citation_precision + citation_recall)
        if citation_precision + citation_recall else 0.0
    )
    return {
        "dataset_kind": payload.get("kind", "unspecified"),
        "provenance": payload.get("provenance"),
        "sample_count": len(cases),
        "stance_accuracy": stance_metrics["accuracy"],
        "stance_macro_f1": stance_metrics["macro_f1"],
        "stance_per_class": stance_metrics["per_stance"],
        "citation_precision": citation_precision,
        "citation_recall": citation_recall,
        "citation_f1": citation_f1,
        "per_case": per_case,
        "limitation": (
            "Citation overlap measures agreement with human annotations; it does not "
            "prove semantic entailment or establish forecast accuracy."
        ),
    }

"""Conservative staged-trial gate for experimental fusion evaluation results."""
from __future__ import annotations

import math
from typing import Any, Mapping

from config import HORIZONS


def _metric(metrics: Mapping[str, Any], key: str, partition: str) -> float:
    value = metrics.get(key)
    if isinstance(value, bool):
        raise ValueError(f"{partition} metric {key} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{partition} metric {key} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{partition} metric {key} must be finite")
    return number


def assess_fusion_staged_trial(model_result: Mapping[str, Any]) -> dict[str, Any]:
    """Assess fixed promotion conditions without activating or mutating any model."""
    if not isinstance(model_result, Mapping):
        raise ValueError("A fusion model evaluation result is required")
    partitions = {}
    for name in ("validation", "test"):
        partition = model_result.get(name)
        if not isinstance(partition, Mapping) or not isinstance(partition.get("macro_average"), Mapping):
            raise ValueError(f"Fusion model result is missing {name} metrics")
        groups = partition.get("per_ticker_horizon")
        if not isinstance(groups, list):
            raise ValueError(f"Fusion model result is missing per-horizon {name} metrics")
        partitions[name] = (partition, groups)

    required_horizons = set(HORIZONS)
    present_horizons = {}
    for name, (_, groups) in partitions.items():
        present_horizons[name] = {
            horizon
            for row in groups
            if isinstance(row, Mapping)
            for horizon in (row.get("horizon"),)
            if isinstance(horizon, int) and not isinstance(horizon, bool)
        }
    horizon_coverage = all(required_horizons.issubset(present_horizons[name]) for name in partitions)

    validation_macro = partitions["validation"][0]["macro_average"]
    test_macro = partitions["test"][0]["macro_average"]
    validation_noninferior = (
        _metric(validation_macro, "fused_mae", "validation")
        <= _metric(validation_macro, "lstm_mae", "validation")
        and _metric(validation_macro, "fused_directional_accuracy", "validation")
        >= _metric(validation_macro, "lstm_directional_accuracy", "validation")
    )
    test_beats_baselines = (
        _metric(test_macro, "fused_mae", "test") < _metric(test_macro, "lstm_mae", "test")
        and _metric(test_macro, "fused_mae", "test") < _metric(test_macro, "persistence_mae", "test")
        and _metric(test_macro, "fused_directional_accuracy", "test")
        >= _metric(test_macro, "lstm_directional_accuracy", "test")
    )

    per_horizon_noninferior = True
    horizon_results = {}
    test_groups = partitions["test"][1]
    for horizon in HORIZONS:
        groups = [
            row for row in test_groups
            if isinstance(row, Mapping) and row.get("horizon") == horizon
        ]
        if not groups:
            per_horizon_noninferior = False
            continue
        passes = all(
            _metric(row, "fused_mae", "test") <= _metric(row, "lstm_mae", "test")
            and _metric(row, "fused_directional_accuracy", "test")
            >= _metric(row, "lstm_directional_accuracy", "test")
            for row in groups
        )
        horizon_results[horizon] = {
            "passes": passes,
            "group_count": len(groups),
        }
        per_horizon_noninferior = per_horizon_noninferior and passes

    checks = {
        "all_horizons_present": horizon_coverage,
        "validation_noninferior_to_lstm": validation_noninferior,
        "test_macro_beats_lstm_and_persistence": test_beats_baselines,
        "no_test_horizon_regression": per_horizon_noninferior,
    }
    return {
        "eligible_for_staged_trial": all(checks.values()),
        "production_activation": False,
        "checks": checks,
        "test_horizons": horizon_results,
        "required_horizons": sorted(required_horizons),
        "decision": (
            "All predefined diagnostic checks passed; staged shadow testing may be considered."
            if all(checks.values())
            else "One or more predefined checks failed; keep the LSTM-only forecast active."
        ),
        "limitation": (
            "This gate is an initial diagnostic, not a significance test or production "
            "approval; repeated model selection can invalidate the held-out test."
        ),
    }

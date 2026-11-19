"""Leakage-safe chronological splits for fusion evaluation datasets."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Sequence

from src.fusion_dataset import normalize_as_of, normalize_target_date, validate_fusion_dataset


def chronological_purged_split(
    records: Sequence[Mapping[str, Any]],
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> dict[str, Any]:
    """Split on global forecast origins and purge labels crossing split boundaries."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 < validation_fraction < 1 or train_fraction + validation_fraction >= 1:
        raise ValueError("validation_fraction must be positive and leave a non-empty test fraction")

    rows = validate_fusion_dataset(records)
    origins = sorted({normalize_as_of(row["as_of"]) for row in rows})
    if len(origins) < 3:
        raise ValueError("At least three distinct forecast origins are required for train/validation/test")
    validation_index = int(len(origins) * train_fraction)
    test_index = int(len(origins) * (train_fraction + validation_fraction))
    validation_index = min(max(validation_index, 1), len(origins) - 2)
    test_index = min(max(test_index, validation_index + 1), len(origins) - 1)
    validation_start = origins[validation_index]
    test_start = origins[test_index]

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    purged = {"train": 0, "validation": 0}
    for row in rows:
        origin = normalize_as_of(row["as_of"])
        target = normalize_target_date(row["target_date"])
        if origin < validation_start:
            if target <= validation_start.date():
                splits["train"].append(row)
            else:
                purged["train"] += 1
        elif origin < test_start:
            if target <= test_start.date():
                splits["validation"].append(row)
            else:
                purged["validation"] += 1
        else:
            splits["test"].append(row)

    empty = [name for name, items in splits.items() if not items]
    if empty:
        raise ValueError(
            "Purging left an empty " + ", ".join(empty) +
            " split; provide more forecast origins or a longer history"
        )
    return {
        **splits,
        "metadata": {
            "split_method": "global_chronological_target_date_purged",
            "train_fraction": train_fraction,
            "validation_fraction": validation_fraction,
            "test_fraction": 1.0 - train_fraction - validation_fraction,
            "validation_start": validation_start.isoformat(),
            "test_start": test_start.isoformat(),
            "purged_rows": purged,
            "origin_count": len(origins),
        },
    }

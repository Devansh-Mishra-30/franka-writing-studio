"""Structured evidence writers for robot experiments."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def prepare_output_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    """Write deterministic and human-readable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        file.write("\n")


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Write records using one stable CSV schema."""

    if not rows:
        raise ValueError(
            "cannot write an empty CSV dataset"
        )

    fieldnames = list(rows[0].keys())
    expected_fields = set(fieldnames)

    for index, row in enumerate(rows):
        if set(row.keys()) != expected_fields:
            raise ValueError(
                f"row {index} does not match the CSV schema"
            )

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(rows)

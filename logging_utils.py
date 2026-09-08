"""Structured evidence writers for robot experiments."""

from __future__ import annotations
import hashlib
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
def _git_output(
    repo_root: Path,
    *arguments: str,
) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    return result.stdout.strip()


def build_run_provenance(
    *,
    repo_root: Path,
    model_path: Path,
) -> dict[str, Any]:
    """Capture source, runtime, dependency, and model identity."""

    root = repo_root.resolve()
    model = model_path.resolve()

    if not model.is_file():
        raise FileNotFoundError(
            f"model file does not exist: {model}"
        )

    commit = _git_output(root, "rev-parse", "HEAD")
    status = _git_output(
        root,
        "status",
        "--porcelain",
        "--untracked-files=no",
    )

    package_names = (
        "numpy",
        "scipy",
        "pybullet",
        "pin",
        "PySide6",
        "pyqtgraph",
        "pytest",
    )

    packages: dict[str, str | None] = {}

    for package_name in package_names:
        try:
            packages[package_name] = metadata.version(
                package_name
            )
        except metadata.PackageNotFoundError:
            packages[package_name] = None

    return {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "git": {
            "commit": commit,
            "dirty": (
                None
                if status is None
                else bool(status)
            ),
        },
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "packages": packages,
        },
        "model": {
            "path": str(
                model.relative_to(root)
                if model.is_relative_to(root)
                else model
            ),
            "sha256": hashlib.sha256(
                model.read_bytes()
            ).hexdigest(),
        },
    }

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

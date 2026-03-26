"""Generic CSV readers shared by table-backed IO readers."""

from __future__ import annotations

import csv
from pathlib import Path


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with open(Path(path), encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


__all__ = ["read_csv_rows"]

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Any

def _to_csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    # list/dict/other json-able values
    return json.dumps(value, ensure_ascii=False)


def build_rows(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for json_path in sorted(root.rglob("*.json")):
        if not json_path.is_file():
            continue
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            # skip invalid json files
            continue
        if isinstance(data, dict):
            rows.append(data)
    return rows


def write_csv(rows: List[Dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Union of all keys across JSON objects
    column_set = set()
    for row in rows:
        column_set.update(row.keys())
    columns = sorted(column_set)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            out_row = {k: _to_csv_cell(row.get(k)) for k in columns}
            writer.writerow(out_row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a CSV from JSON contents in a folder tree."
    )
    parser.add_argument(
        "input_folder",
        type=Path,
        help="Root folder to scan recursively for JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("structural_index.csv"),
        help="Output CSV path (default: structural_index.csv).",
    )
    args = parser.parse_args()

    root = args.input_folder.resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Input folder does not exist or is not a directory: {root}")

    rows = build_rows(root)
    write_csv(rows, args.output.resolve())

    print(f"Wrote {len(rows)} rows to {args.output.resolve()}")


if __name__ == "__main__":
    main()

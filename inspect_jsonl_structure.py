"""Print the nested structure of a JSON Lines file."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    return type(value).__name__


def collect_structure(value: Any, path: str, structure: dict[str, set[str]]) -> None:
    structure[path].add(type_name(value))

    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            collect_structure(child, child_path, structure)
    elif isinstance(value, list):
        item_path = f"{path}[]"
        if value:
            for item in value:
                collect_structure(item, item_path, structure)
        else:
            structure[item_path].add("<empty>")


def inspect_jsonl(path: Path, max_records: int | None) -> dict[str, set[str]]:
    structure: dict[str, set[str]] = defaultdict(set)

    with path.open("r", encoding="utf-8") as file:
        for record_number, line in enumerate(file, start=1):
            if max_records is not None and record_number > max_records:
                break

            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {record_number}: {error.msg}"
                ) from error

            collect_structure(record, "$", structure)

    return structure


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print all columns and nested sub-columns in a JSONL file."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path(r"goal_dataset\train\shard_00000.jsonl"),
        help="Path to the JSONL file.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Only inspect this many records (default: inspect the whole file).",
    )
    args = parser.parse_args()

    if not args.path.is_file():
        parser.error(f"File not found: {args.path}")
    if args.max_records is not None and args.max_records < 1:
        parser.error("--max-records must be at least 1")

    structure = inspect_jsonl(args.path, args.max_records)

    print(f"Structure of: {args.path}")
    for path, types in sorted(structure.items()):
        print(f"{path}: {', '.join(sorted(types))}")


if __name__ == "__main__":
    main()

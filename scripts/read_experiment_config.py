from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def get_nested(data: dict[str, Any], dotted_key: str) -> Any:
    value: Any = data
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise SystemExit(f"Missing config field: {dotted_key}")
        value = value[part]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Read values from the frozen ratio010 experiment config.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fields", nargs="+", required=True)
    args = parser.parse_args()

    data = json.loads(args.config.read_text(encoding="utf-8"))
    values = [get_nested(data, field) for field in args.fields]
    if any(isinstance(value, (dict, list)) for value in values):
        raise SystemExit("Requested fields must be scalar values")
    print("\t".join(str(value) for value in values))


if __name__ == "__main__":
    main()

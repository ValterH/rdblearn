from __future__ import annotations

import json
from pathlib import Path


def resolve_output_path(
    output_path: str,
    default_filename: str,
    results_prefix: str,
    random_state: int | None,
) -> Path:
    path = Path(output_path)
    if path.name == default_filename:
        seed_suffix = f"_{random_state}" if random_state is not None else ""
        path = Path("results") / f"{results_prefix}{seed_suffix}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_results_by_key(path: Path) -> dict[tuple[str | None, str | None], dict]:
    existing_results = []
    if path.exists():
        with open(path, "r") as f:
            existing_results = json.load(f)
    return {(r.get("dataset"), r.get("task")): r for r in existing_results}


def write_results(
    path: Path,
    results_by_key: dict[tuple[str | None, str | None], dict],
) -> None:
    sorted_items = sorted(
        results_by_key.items(),
        key=lambda kv: ((kv[0][0] or ""), (kv[0][1] or "")),
    )
    results = [row for _, row in sorted_items]
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

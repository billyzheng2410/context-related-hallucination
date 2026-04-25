#!/usr/bin/env python3
"""Add bootstrap confidence intervals to a Gumbel ablation run."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def metric_vectors(base: list[dict[str, Any]], ablated: list[dict[str, Any]]) -> dict[str, list[float]]:
    if len(base) != len(ablated):
        raise ValueError("Base and ablated files must contain the same number of rows.")

    attacked_delta = [
        new["attacked_margin"] - old["attacked_margin"]
        for old, new in zip(base, ablated)
    ]
    clean_delta = [
        new["clean_margin"] - old["clean_margin"]
        for old, new in zip(base, ablated)
    ]
    selective_effect = [
        attacked - abs(clean)
        for attacked, clean in zip(attacked_delta, clean_delta)
    ]
    return {
        "attacked_delta_margin": attacked_delta,
        "clean_delta_margin": clean_delta,
        "selective_effect": selective_effect,
    }


def summarize_vector(values: list[float], n_bootstrap: int, seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    n = len(values)
    observed = mean(values)
    samples = []
    for _ in range(n_bootstrap):
        draw = [values[rng.randrange(n)] for _ in range(n)]
        samples.append(mean(draw))
    samples.sort()
    return {
        "mean": observed,
        "ci_low": percentile(samples, 0.025),
        "ci_high": percentile(samples, 0.975),
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute percentile of empty values.")
    idx = q * (len(sorted_values) - 1)
    lower = int(idx)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = idx - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def summarize_pair(
    base: list[dict[str, Any]],
    ablated: list[dict[str, Any]],
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    vectors = metric_vectors(base, ablated)
    return {
        name: summarize_vector(values, n_bootstrap=n_bootstrap, seed=seed + i)
        for i, (name, values) in enumerate(vectors.items())
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-name", default="summary_with_ci.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = read_json(args.run_dir / "summary.json")
    base = read_jsonl(args.run_dir / "base_eval.jsonl")
    learned = read_jsonl(args.run_dir / "learned_ablation_eval.jsonl")
    random_rows = read_jsonl(args.run_dir / "random_ablation_eval.jsonl")

    summary["bootstrap"] = {
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "learned_ablation": summarize_pair(base, learned, args.n_bootstrap, args.seed),
        "random_ablation": summarize_pair(base, random_rows, args.n_bootstrap, args.seed + 10000),
    }

    output = args.run_dir / args.output_name
    write_json(output, summary)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

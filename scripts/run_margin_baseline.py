#!/usr/bin/env python3
"""Run a next-token logit-margin baseline on normalized samples.

This does not perform Gumbel search yet. It gives a cheap smoke test for
whether a model and dataset produce meaningful gold-vs-distractor margins.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_model_and_tokenizer(args: argparse.Namespace):
    if args.offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map=args.device_map,
        local_files_only=args.local_files_only,
    )
    model.eval()
    return model, tokenizer


def encode_single_token(tokenizer, text: str) -> int | None:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        return None
    return ids[0]


@torch.no_grad()
def margin_for_prompt(model, tokenizer, prompt: str, gold: str, distractor: str) -> dict[str, Any]:
    gold_id = encode_single_token(tokenizer, gold)
    distractor_id = encode_single_token(tokenizer, distractor)
    if gold_id is None or distractor_id is None:
        return {
            "status": "skipped_multi_token_answer",
            "gold_token_id": gold_id,
            "distractor_token_id": distractor_id,
        }

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]
    gold_logit = logits[gold_id].float().item()
    distractor_logit = logits[distractor_id].float().item()

    return {
        "status": "ok",
        "gold_token_id": gold_id,
        "distractor_token_id": distractor_id,
        "gold_logit": gold_logit,
        "distractor_logit": distractor_logit,
        "margin": gold_logit - distractor_logit,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in results if row["status"] == "ok"]
    summary: dict[str, Any] = {
        "total_rows": len(results),
        "ok_rows": len(ok),
        "status_counts": dict(Counter(row["status"] for row in results)),
    }
    if not ok:
        return summary

    for prefix in ["clean", "attacked"]:
        margins = [row[f"{prefix}_margin"] for row in ok]
        summary[f"{prefix}_margin_mean"] = sum(margins) / len(margins)
        summary[f"{prefix}_margin_positive_rate"] = sum(m > 0 for m in margins) / len(margins)

    deltas = [row["attacked_margin"] - row["clean_margin"] for row in ok]
    summary["attacked_minus_clean_margin_mean"] = sum(deltas) / len(deltas)

    by_relation = defaultdict(list)
    by_relation_domain = defaultdict(list)
    for row in ok:
        by_relation[row["relation"]].append(row)
        by_relation_domain[f"{row['relation']}::{row['domain']}"].append(row)

    summary["by_relation"] = summarize_groups(by_relation)
    summary["by_relation_domain"] = summarize_groups(by_relation_domain)
    return summary


def summarize_groups(groups: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    out = {}
    for name, rows in groups.items():
        clean = [row["clean_margin"] for row in rows]
        attacked = [row["attacked_margin"] for row in rows]
        delta = [row["attacked_margin"] - row["clean_margin"] for row in rows]
        out[name] = {
            "n": len(rows),
            "clean_margin_mean": sum(clean) / len(clean),
            "attacked_margin_mean": sum(attacked) / len(attacked),
            "attacked_minus_clean_margin_mean": sum(delta) / len(delta),
            "clean_margin_positive_rate": sum(m > 0 for m in clean) / len(clean),
            "attacked_margin_positive_rate": sum(m > 0 for m in attacked) / len(attacked),
        }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/better_counterfactual.jsonl"))
    parser.add_argument("--model-name", default="EleutherAI/pythia-410m")
    parser.add_argument("--output", type=Path, default=Path("results/baselines/pythia_410m_margin_results.jsonl"))
    parser.add_argument("--summary-output", type=Path, default=Path("results/baselines/pythia_410m_margin_summary.json"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--relation", default=None)
    parser.add_argument("--domain", default=None)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.dataset)

    if args.relation:
        rows = [row for row in rows if row["relation"] == args.relation]
    if args.domain:
        rows = [row for row in rows if row["domain"] == args.domain]
    if args.limit is not None:
        rows = rows[: args.limit]

    model, tokenizer = load_model_and_tokenizer(args)
    results = []

    for row in tqdm(rows, desc="Margin baseline"):
        clean = margin_for_prompt(
            model,
            tokenizer,
            row["clean_prompt"],
            row["gold_word"],
            row["distractor_word"],
        )
        attacked = margin_for_prompt(
            model,
            tokenizer,
            row["attacked_prompt"],
            row["gold_word"],
            row["distractor_word"],
        )

        if clean["status"] != "ok" or attacked["status"] != "ok":
            status = clean["status"] if clean["status"] != "ok" else attacked["status"]
            results.append({**row, "status": status})
            continue

        results.append(
            {
                **row,
                "status": "ok",
                "clean_gold_logit": clean["gold_logit"],
                "clean_distractor_logit": clean["distractor_logit"],
                "clean_margin": clean["margin"],
                "attacked_gold_logit": attacked["gold_logit"],
                "attacked_distractor_logit": attacked["distractor_logit"],
                "attacked_margin": attacked["margin"],
            }
        )

    summary = summarize(results)
    summary["model_name"] = args.model_name
    summary["dataset"] = str(args.dataset)
    summary["limit"] = args.limit
    summary["relation_filter"] = args.relation
    summary["domain_filter"] = args.domain

    write_jsonl(args.output, results)
    write_json(args.summary_output, summary)
    print(f"Wrote {len(results)} rows to {args.output}")
    print(f"Wrote summary to {args.summary_output}")


if __name__ == "__main__":
    main()

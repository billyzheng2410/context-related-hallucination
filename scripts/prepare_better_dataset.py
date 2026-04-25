#!/usr/bin/env python3
"""Normalize relation data into counterfactual evaluation samples.

The input format is the hand-built "better dataset" JSON:

{
  "taxonomy": {
    "pairs": {"natural": [["dog", " animal"], ...]},
    "templates": ["A {0} is a type of", ...]
  },
  ...
}

The output JSONL format is intentionally close to the original notebook:
each row has an attacked prompt, a clean prompt, the gold answer, and the
distractor answer. This gives later scripts a stable interface.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def load_tokenizer(model_name: str, local_files_only: bool):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )


def token_count(tokenizer: Any | None, text: str) -> int | None:
    if tokenizer is None:
        return None
    return len(tokenizer.encode(text, add_special_tokens=False))


def iter_counterfactual_rows(
    raw: dict[str, Any],
    tokenizer: Any | None = None,
    filter_single_token_answers: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "relations": {},
        "skipped": Counter(),
    }

    for relation, spec in raw.items():
        relation_counts = Counter()
        domains = spec.get("pairs", {})
        templates = spec.get("templates", [])

        for domain, pairs in domains.items():
            normalized_pairs = [(str(subject), str(answer)) for subject, answer in pairs]
            if len(normalized_pairs) < 2:
                summary["skipped"]["domain_too_small"] += 1
                continue

            for template_idx, template in enumerate(templates):
                for query_idx, (query_subject, gold_answer) in enumerate(normalized_pairs):
                    attack_pair = choose_distractor_pair(
                        normalized_pairs,
                        query_idx=query_idx,
                        gold_answer=gold_answer,
                    )
                    if attack_pair is None:
                        summary["skipped"]["no_distinct_distractor"] += 1
                        continue

                    attack_subject, distractor_answer = attack_pair
                    gold_token_count = token_count(tokenizer, gold_answer)
                    distractor_token_count = token_count(tokenizer, distractor_answer)

                    if filter_single_token_answers and (
                        gold_token_count != 1 or distractor_token_count != 1
                    ):
                        summary["skipped"]["multi_token_answer"] += 1
                        continue

                    attack_sentence = f"{render_template(template, attack_subject)}{distractor_answer}."
                    clean_prompt = render_template(template, query_subject)
                    attacked_prompt = f"{attack_sentence} {clean_prompt}"

                    row = {
                        "id": len(rows),
                        "relation": relation,
                        "domain": domain,
                        "template_idx": template_idx,
                        "template": template,
                        "attack_subject": attack_subject,
                        "query_subject": query_subject,
                        "gold_word": gold_answer,
                        "distractor_word": distractor_answer,
                        "gold_token_count": gold_token_count,
                        "distractor_token_count": distractor_token_count,
                        "clean_prompt": clean_prompt,
                        "attacked_prompt": attacked_prompt,
                    }
                    rows.append(row)
                    relation_counts[domain] += 1

        summary["relations"][relation] = dict(relation_counts)

    summary["skipped"] = dict(summary["skipped"])
    summary["total_rows"] = len(rows)
    summary["unique_relations"] = len(summary["relations"])
    summary["domains_per_relation"] = {
        relation: sorted(counts.keys())
        for relation, counts in summary["relations"].items()
    }
    return rows, summary


def choose_distractor_pair(
    pairs: list[tuple[str, str]],
    query_idx: int,
    gold_answer: str,
) -> tuple[str, str] | None:
    """Pick a nearby pair whose answer differs from the query answer.

    Many relation datasets contain local runs with the same target
    (dog/cat/bird -> animal). Adjacent cycling would often create
    gold == distractor, which is useless for logit-margin evaluation.
    """
    n = len(pairs)
    for offset in range(1, n):
        candidate = pairs[(query_idx + offset) % n]
        if candidate[1] != gold_answer:
            return candidate
    return None


def render_template(template: str, subject: str) -> str:
    """Render a template and fix simple English article mismatches.

    Several hand-written templates start with "A {0}". Without this pass,
    examples such as "A eagle" appear. We keep the fix intentionally narrow
    so the original template semantics stay unchanged.
    """
    rendered = template.format(subject)
    stripped_subject = subject.strip()
    if not stripped_subject:
        return rendered

    starts_with_vowel_sound = stripped_subject[0].lower() in {"a", "e", "i", "o", "u"}
    if starts_with_vowel_sound:
        if rendered.startswith("A "):
            return "An " + rendered[2:]
        if rendered.startswith("a "):
            return "an " + rendered[2:]
    return rendered


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def add_summary_breakdowns(rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    by_relation = Counter(row["relation"] for row in rows)
    by_relation_domain = defaultdict(Counter)
    by_template = defaultdict(Counter)

    for row in rows:
        by_relation_domain[row["relation"]][row["domain"]] += 1
        by_template[row["relation"]][row["template_idx"]] += 1

    summary["rows_by_relation"] = dict(by_relation)
    summary["rows_by_relation_domain"] = {
        relation: dict(counts) for relation, counts in by_relation_domain.items()
    }
    summary["rows_by_relation_template"] = {
        relation: dict(counts) for relation, counts in by_template.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("better dataset/data.json"),
        help="Path to the better dataset JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/better_counterfactual.jsonl"),
        help="Where to write normalized counterfactual samples.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("data/processed/better_counterfactual_summary.json"),
        help="Where to write dataset summary metadata.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional tokenizer name/path for single-token answer filtering.",
    )
    parser.add_argument(
        "--filter-single-token-answers",
        action="store_true",
        help="Keep only rows where both gold and distractor answers are single tokens.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load tokenizer from local Hugging Face cache only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = json.loads(args.input.read_text(encoding="utf-8"))

    tokenizer = None
    if args.model_name:
        tokenizer = load_tokenizer(args.model_name, args.local_files_only)

    rows, summary = iter_counterfactual_rows(
        raw,
        tokenizer=tokenizer,
        filter_single_token_answers=args.filter_single_token_answers,
    )
    add_summary_breakdowns(rows, summary)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, rows)
    write_json(args.summary_output, summary)

    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"Wrote summary to {args.summary_output}")
    if summary["skipped"]:
        print("Skipped:", summary["skipped"])


if __name__ == "__main__":
    main()

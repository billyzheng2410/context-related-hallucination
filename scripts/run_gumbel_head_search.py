#!/usr/bin/env python3
"""Run local Gumbel head search on normalized counterfactual samples.

This script is designed for the current experiment, not for the older local
prototype files. It consumes `data/processed/better_counterfactual.jsonl` and
produces a reusable head list plus ablation metrics.

Implementation note:
Pythia is loaded through Hugging Face `AutoModelForCausalLM`. We attach hooks to
each transformer layer's attention module and reshape the attention output into
`num_attention_heads` chunks before the residual add. This matches the local
Pythia prototype style and is sufficient for a first reproducible small-model
pipeline. If we later move to TransformerLens/nnsight for exact `hook_z` access,
the dataset and metric format can stay unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import Counter
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    for param in model.parameters():
        param.requires_grad_(False)
    return model, tokenizer


def encode_single_token(tokenizer, text: str) -> int | None:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        return None
    return ids[0]


def filter_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.relation:
        rows = [row for row in rows if row["relation"] == args.relation]
    if args.domain:
        rows = [row for row in rows if row["domain"] == args.domain]
    if args.template_idx is not None:
        rows = [row for row in rows if row["template_idx"] == args.template_idx]
    if args.limit is not None:
        rows = rows[: args.limit]
    return rows


def attach_token_ids(rows: list[dict[str, Any]], tokenizer) -> tuple[list[dict[str, Any]], Counter]:
    kept = []
    skipped = Counter()
    for row in rows:
        gold_id = encode_single_token(tokenizer, row["gold_word"])
        distractor_id = encode_single_token(tokenizer, row["distractor_word"])
        if gold_id is None or distractor_id is None:
            skipped["multi_token_answer"] += 1
            continue
        kept.append({**row, "gold_token_id": gold_id, "distractor_token_id": distractor_id})
    return kept, skipped


def get_transformer_layers(model) -> list[nn.Module]:
    if hasattr(model, "gpt_neox"):
        return list(model.gpt_neox.layers)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    raise AttributeError("Cannot find transformer layers for this model.")


def get_attention_module(layer: nn.Module) -> nn.Module:
    for name in ["attention", "self_attn", "attn"]:
        if hasattr(layer, name):
            return getattr(layer, name)
    raise AttributeError(f"Cannot find attention module in layer {type(layer).__name__}.")


class GumbelHeadMask(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        device: torch.device,
        init_std: float = 0.01,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.temperature = temperature
        self.logits = nn.Parameter(torch.empty(num_layers, num_heads, device=device, dtype=torch.float32))
        nn.init.normal_(self.logits, mean=0.0, std=init_std)

    def sample(self, hard: bool = True) -> torch.Tensor:
        uniform = torch.rand((2,) + tuple(self.logits.shape), device=self.logits.device)
        noise = -torch.log(-torch.log(uniform[1] + 1e-10) / (-torch.log(uniform[0] + 1e-10)) + 1e-10)
        probs = torch.sigmoid((self.logits + noise) / self.temperature)
        if not hard:
            return probs
        return ((probs > 0.5).to(probs.dtype) - probs).detach() + probs

    def expected_probs(self) -> torch.Tensor:
        return torch.sigmoid(self.logits)

    def sparsity_loss(self) -> torch.Tensor:
        return self.expected_probs().mean()


class AttentionHeadHooker:
    def __init__(self, model, num_heads: int) -> None:
        self.model = model
        self.layers = get_transformer_layers(model)
        self.num_heads = num_heads
        self.handles: list[Any] = []

    @contextmanager
    def apply_train_mask(self, mask_module: GumbelHeadMask):
        self._register_train_hooks(mask_module)
        try:
            yield
        finally:
            self.remove()

    @contextmanager
    def ablate_heads(self, heads_to_zero: list[tuple[int, int]]):
        self._register_ablation_hooks(heads_to_zero)
        try:
            yield
        finally:
            self.remove()

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _register_train_hooks(self, mask_module: GumbelHeadMask) -> None:
        sampled_mask = mask_module.sample(hard=True)
        for layer_idx, layer in enumerate(self.layers):
            attn = get_attention_module(layer)

            def make_hook(idx: int):
                def hook(_module, _inputs, output):
                    layer_mask = sampled_mask[idx]
                    return apply_head_mask_to_output(output, layer_mask, self.num_heads)

                return hook

            self.handles.append(attn.register_forward_hook(make_hook(layer_idx)))

    def _register_ablation_hooks(self, heads_to_zero: list[tuple[int, int]]) -> None:
        mask = torch.ones(len(self.layers), self.num_heads, device=self.model.device, dtype=torch.float32)
        for layer_idx, head_idx in heads_to_zero:
            if 0 <= layer_idx < len(self.layers) and 0 <= head_idx < self.num_heads:
                mask[layer_idx, head_idx] = 0.0

        for layer_idx, layer in enumerate(self.layers):
            attn = get_attention_module(layer)

            def make_hook(idx: int):
                def hook(_module, _inputs, output):
                    return apply_head_mask_to_output(output, mask[idx], self.num_heads)

                return hook

            self.handles.append(attn.register_forward_hook(make_hook(layer_idx)))


def apply_head_mask_to_output(output, layer_mask: torch.Tensor, num_heads: int):
    if isinstance(output, tuple):
        attn_output = output[0]
        masked = mask_attention_tensor(attn_output, layer_mask, num_heads)
        return (masked,) + output[1:]
    return mask_attention_tensor(output, layer_mask, num_heads)


def mask_attention_tensor(attn_output: torch.Tensor, layer_mask: torch.Tensor, num_heads: int) -> torch.Tensor:
    if attn_output.ndim != 3:
        return attn_output
    batch_size, seq_len, hidden_dim = attn_output.shape
    if hidden_dim % num_heads != 0:
        return attn_output
    head_dim = hidden_dim // num_heads
    reshaped = attn_output.view(batch_size, seq_len, num_heads, head_dim)
    mask = layer_mask.to(device=attn_output.device, dtype=attn_output.dtype).view(1, 1, num_heads, 1)
    return (reshaped * mask).view(batch_size, seq_len, hidden_dim)


def batch_iter(rows: list[dict[str, Any]], batch_size: int, shuffle: bool) -> Iterable[list[dict[str, Any]]]:
    indices = list(range(len(rows)))
    if shuffle:
        random.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield [rows[i] for i in indices[start : start + batch_size]]


def compute_batch_logits(model, tokenizer, prompts: list[str]) -> torch.Tensor:
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    outputs = model(**inputs)
    last_indices = inputs["attention_mask"].sum(dim=1) - 1
    batch_idx = torch.arange(len(prompts), device=model.device)
    return outputs.logits[batch_idx, last_indices, :]


def compute_effect_loss(logits: torch.Tensor, batch: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    gold_ids = torch.tensor([row["gold_token_id"] for row in batch], device=logits.device, dtype=torch.long)
    distractor_ids = torch.tensor([row["distractor_token_id"] for row in batch], device=logits.device, dtype=torch.long)
    idx = torch.arange(len(batch), device=logits.device)
    gold_logits = logits[idx, gold_ids]
    distractor_logits = logits[idx, distractor_ids]
    two_way_logits = torch.stack([gold_logits, distractor_logits], dim=-1)
    labels = torch.zeros(len(batch), dtype=torch.long, device=logits.device)
    loss = F.cross_entropy(two_way_logits.float(), labels)
    margin = (gold_logits - distractor_logits).float().mean()
    return loss, margin


def run_gumbel_search(
    model,
    tokenizer,
    train_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[GumbelHeadMask, dict[str, list[float]]]:
    num_layers = getattr(model.config, "num_hidden_layers")
    num_heads = getattr(model.config, "num_attention_heads")
    mask_module = GumbelHeadMask(
        num_layers=num_layers,
        num_heads=num_heads,
        device=model.device,
        temperature=args.temperature,
    )
    hooker = AttentionHeadHooker(model, num_heads=num_heads)
    optimizer = torch.optim.AdamW(mask_module.parameters(), lr=args.lr)
    history = {
        "loss": [],
        "effect_loss": [],
        "sparsity_loss": [],
        "margin": [],
        "keep_rate": [],
    }

    progress = tqdm(range(args.epochs), desc="Gumbel head search")
    for _epoch in progress:
        epoch_loss = []
        epoch_effect = []
        epoch_sparse = []
        epoch_margin = []
        for batch in batch_iter(train_rows, args.batch_size, shuffle=True):
            prompts = [row["attacked_prompt"] for row in batch]
            optimizer.zero_grad(set_to_none=True)
            with hooker.apply_train_mask(mask_module):
                logits = compute_batch_logits(model, tokenizer, prompts)
                effect_loss, margin = compute_effect_loss(logits, batch)
            sparse_loss = mask_module.sparsity_loss()
            if args.sparsity_mode == "reward_keep":
                loss = effect_loss - args.sparsity_lambda * sparse_loss
            else:
                loss = effect_loss + args.sparsity_lambda * sparse_loss
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
            epoch_effect.append(effect_loss.item())
            epoch_sparse.append(sparse_loss.item())
            epoch_margin.append(margin.item())

        history["loss"].append(float(np.mean(epoch_loss)))
        history["effect_loss"].append(float(np.mean(epoch_effect)))
        history["sparsity_loss"].append(float(np.mean(epoch_sparse)))
        history["margin"].append(float(np.mean(epoch_margin)))
        history["keep_rate"].append(mask_module.expected_probs().mean().item())
        progress.set_postfix(
            loss=history["loss"][-1],
            margin=history["margin"][-1],
            keep=history["keep_rate"][-1],
        )

    return mask_module, history


def bottom_k_heads(mask_module: GumbelHeadMask, k: int) -> list[tuple[int, int]]:
    probs = mask_module.expected_probs().detach().cpu().numpy()
    num_layers, num_heads = probs.shape
    flat = np.argsort(probs.reshape(-1))[:k]
    return [(int(idx // num_heads), int(idx % num_heads)) for idx in flat]


def random_heads(num_layers: int, num_heads: int, k: int, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    all_heads = [(layer, head) for layer in range(num_layers) for head in range(num_heads)]
    return rng.sample(all_heads, k=min(k, len(all_heads)))


@torch.no_grad()
def evaluate_rows(
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    heads_to_zero: list[tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    hooker = AttentionHeadHooker(model, num_heads=getattr(model.config, "num_attention_heads"))
    context = hooker.ablate_heads(heads_to_zero) if heads_to_zero else nullcontext()
    results = []
    with context:
        for batch in batch_iter(rows, args.eval_batch_size, shuffle=False):
            clean_prompts = [row["clean_prompt"] for row in batch]
            attacked_prompts = [row["attacked_prompt"] for row in batch]
            clean_logits = compute_batch_logits(model, tokenizer, clean_prompts)
            attacked_logits = compute_batch_logits(model, tokenizer, attacked_prompts)
            results.extend(score_batch(clean_logits, attacked_logits, batch))
    return results


def score_batch(
    clean_logits: torch.Tensor,
    attacked_logits: torch.Tensor,
    batch: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out = []
    for idx, row in enumerate(batch):
        gold_id = row["gold_token_id"]
        distractor_id = row["distractor_token_id"]
        clean_gold = clean_logits[idx, gold_id].float().item()
        clean_distractor = clean_logits[idx, distractor_id].float().item()
        attacked_gold = attacked_logits[idx, gold_id].float().item()
        attacked_distractor = attacked_logits[idx, distractor_id].float().item()
        out.append(
            {
                **row,
                "clean_gold_logit": clean_gold,
                "clean_distractor_logit": clean_distractor,
                "clean_margin": clean_gold - clean_distractor,
                "attacked_gold_logit": attacked_gold,
                "attacked_distractor_logit": attacked_distractor,
                "attacked_margin": attacked_gold - attacked_distractor,
            }
        )
    return out


def summarize_eval(base: list[dict[str, Any]], ablated: list[dict[str, Any]]) -> dict[str, Any]:
    assert len(base) == len(ablated)
    attacked_delta = [new["attacked_margin"] - old["attacked_margin"] for old, new in zip(base, ablated)]
    clean_delta = [new["clean_margin"] - old["clean_margin"] for old, new in zip(base, ablated)]
    attacked_base = [row["attacked_margin"] for row in base]
    clean_base = [row["clean_margin"] for row in base]
    attacked_new = [row["attacked_margin"] for row in ablated]
    clean_new = [row["clean_margin"] for row in ablated]
    return {
        "n": len(base),
        "attacked_base_margin_mean": mean(attacked_base),
        "attacked_ablated_margin_mean": mean(attacked_new),
        "attacked_delta_margin_mean": mean(attacked_delta),
        "clean_base_margin_mean": mean(clean_base),
        "clean_ablated_margin_mean": mean(clean_new),
        "clean_delta_margin_mean": mean(clean_delta),
        "selective_effect_mean": mean([a - abs(c) for a, c in zip(attacked_delta, clean_delta)]),
        "attacked_base_positive_rate": positive_rate(attacked_base),
        "attacked_ablated_positive_rate": positive_rate(attacked_new),
        "clean_base_positive_rate": positive_rate(clean_base),
        "clean_ablated_positive_rate": positive_rate(clean_new),
    }


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else math.nan


def positive_rate(values: list[float]) -> float:
    return float(sum(value > 0 for value in values) / len(values)) if values else math.nan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/better_counterfactual.jsonl"))
    parser.add_argument("--model-name", default="EleutherAI/pythia-410m")
    parser.add_argument("--relation", default="taxonomy")
    parser.add_argument("--domain", default="natural")
    parser.add_argument("--template-idx", type=int, default=0)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--sparsity-lambda", type=float, default=0.01)
    parser.add_argument(
        "--sparsity-mode",
        choices=["penalize_keep", "reward_keep"],
        default="penalize_keep",
        help=(
            "penalize_keep encourages a smaller active mask; reward_keep matches "
            "the original notebook loss sign."
        ),
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k-heads", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float32")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/gumbel"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rows = filter_rows(read_jsonl(args.dataset), args)
    if not rows:
        raise ValueError("No rows left after filtering.")

    model, tokenizer = load_model_and_tokenizer(args)
    rows, skipped = attach_token_ids(rows, tokenizer)
    if not rows:
        raise ValueError(f"No usable rows after token filtering. Skipped: {dict(skipped)}")

    eval_rows = rows[: args.eval_limit] if args.eval_limit else rows
    print(f"Using {len(rows)} train rows and {len(eval_rows)} eval rows.")
    if skipped:
        print("Skipped:", dict(skipped))

    mask_module, history = run_gumbel_search(model, tokenizer, rows, args)
    learned_heads = bottom_k_heads(mask_module, args.top_k_heads)
    random_baseline_heads = random_heads(
        getattr(model.config, "num_hidden_layers"),
        getattr(model.config, "num_attention_heads"),
        args.top_k_heads,
        seed=args.seed + 1000,
    )

    base_eval = evaluate_rows(model, tokenizer, eval_rows, args, heads_to_zero=None)
    learned_eval = evaluate_rows(model, tokenizer, eval_rows, args, heads_to_zero=learned_heads)
    random_eval = evaluate_rows(model, tokenizer, eval_rows, args, heads_to_zero=random_baseline_heads)

    run_name = f"{args.model_name.replace('/', '_')}_{args.relation}_{args.domain}_seed{args.seed}"
    out_dir = args.output_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "config": vars(args) | {"dataset": str(args.dataset), "output_dir": str(args.output_dir)},
        "num_train_rows": len(rows),
        "num_eval_rows": len(eval_rows),
        "learned_heads": learned_heads,
        "random_heads": random_baseline_heads,
        "learned_ablation": summarize_eval(base_eval, learned_eval),
        "random_ablation": summarize_eval(base_eval, random_eval),
        "final_history": {key: values[-1] for key, values in history.items()},
    }

    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "history.json", history)
    write_jsonl(out_dir / "base_eval.jsonl", base_eval)
    write_jsonl(out_dir / "learned_ablation_eval.jsonl", learned_eval)
    write_jsonl(out_dir / "random_ablation_eval.jsonl", random_eval)
    print(f"Wrote run outputs to {out_dir}")
    print(json.dumps(summary["learned_ablation"], indent=2))


if __name__ == "__main__":
    main()

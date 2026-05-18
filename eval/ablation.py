"""
Ablation study: isolating the contribution of each LASH component.

Compares training-level variants on A2A negotiation. Each variant requires
a separately trained checkpoint (see training flags below).

Variants and how to train them:
  full_lash    Stage 2 checkpoint (standard pipeline)
  stage1_only  Stage 1 final checkpoint — no RL; tests SFT-only performance
  no_oracle    Stage 1 trained with --oracle-decay-steps 0 (no curriculum) → Stage 2
  no_stage1    Stage 2 run from base: train_stage2.py with a base checkpoint
               (load_from_stage1 called on a base model saved as a dummy checkpoint)

Additionally tests within the full_lash checkpoint:
  no_hyp       full_lash weights but z_t zeroed → tests if z_t conditioning matters vs. base gen
               (architecture unchanged; just zero out the conditioning vector at inference)

Usage:
  python -m eval.ablation \\
      --full-lash   checkpoints/stage2/final \\
      --stage1-only checkpoints/stage1/final \\
      --no-oracle   checkpoints/stage2_no_oracle/final \\
      --no-stage1   checkpoints/stage2_no_stage1/final

  # Minimal (only full_lash required; others are optional):
  python -m eval.ablation --full-lash checkpoints/stage2/final
"""

import argparse
import statistics
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer

from lash.config import LASHConfig
from lash.grpo_env import GRPORolloutEnv
from lash.types import sample_types
from model import LASHModel


# ── Eval runner ────────────────────────────────────────────────────────────

def run_ablation_eval(
    model: LASHModel,
    tokenizer,
    sim_cfg: LASHConfig,
    n_configs: int,
    n_per_config: int,
    max_ctx_len: int,
    max_gen_len: int,
    device: str,
    zero_z: bool = False,
    seed: int = 42,
) -> dict:
    """
    Run evaluation episodes and return metrics dict.

    zero_z=True: zero out z_t before Pass 2 (tests hypothesis module contribution
    without changing architecture — equivalent to the 'no_hyp' ablation variant).
    """
    if zero_z:
        _patch_zero_z(model)

    env = GRPORolloutEnv(
        model=model,
        tokenizer=tokenizer,
        sim_config=sim_cfg,
        max_ctx_len=max_ctx_len,
        max_gen_len=max_gen_len,
        device=device,
    )

    deal_rates, buyer_rewards, seller_rewards, efficiencies = [], [], [], []

    with torch.no_grad():
        for i in range(n_configs):
            buyer_type, seller_type = sample_types(seed=seed + i)
            for _ in range(n_per_config):
                ep = env.run_episode(buyer_type, seller_type)
                deal_rates.append(float(ep.deal_reached))
                buyer_rewards.append(ep.buyer_reward)
                seller_rewards.append(ep.seller_reward)

                if ep.deal_reached and ep.deal_price is not None:
                    v_b = buyer_type.reservation_price
                    v_s = seller_type.reservation_price
                    zopa = v_b - v_s
                    bs = max(0.0, v_b - ep.deal_price)
                    ss = max(0.0, ep.deal_price - v_s)
                    efficiencies.append((bs + ss) / zopa if zopa > 0 else 0.0)
                else:
                    efficiencies.append(0.0)

    if zero_z:
        _unpatch_zero_z(model)

    return {
        "deal_rate": statistics.mean(deal_rates),
        "avg_buyer_reward": statistics.mean(buyer_rewards),
        "avg_seller_reward": statistics.mean(seller_rewards),
        "efficiency": statistics.mean(efficiencies),
        "n_episodes": len(deal_rates),
    }


# ── z_t zeroing patch (no_hyp ablation) ───────────────────────────────────

_orig_hypothesis_attn_forward = None


def _patch_zero_z(model: LASHModel) -> None:
    """
    Monkey-patch hypothesis_attn.forward to return a zero vector.
    Tests whether z_t conditioning carries any signal — if metrics collapse,
    the hypothesis module is doing meaningful work.
    """
    global _orig_hypothesis_attn_forward
    _orig_hypothesis_attn_forward = model.hypothesis_attn.forward

    def zero_forward(c_pooled, h_belief, h_intention):
        z = torch.zeros_like(c_pooled)         # (B, d) — zero vector
        weights = torch.zeros(c_pooled.size(0), 1, 2 * model.cfg.k,
                              device=c_pooled.device)
        return z, weights

    model.hypothesis_attn.forward = zero_forward


def _unpatch_zero_z(model: LASHModel) -> None:
    global _orig_hypothesis_attn_forward
    if _orig_hypothesis_attn_forward is not None:
        model.hypothesis_attn.forward = _orig_hypothesis_attn_forward
        _orig_hypothesis_attn_forward = None


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LASH ablation study")
    # Checkpoint paths (all optional except full_lash)
    p.add_argument("--full-lash",   required=True,  help="Full Stage 2 checkpoint")
    p.add_argument("--stage1-only", default=None,   help="Stage 1 SFT only (no RL)")
    p.add_argument("--no-oracle",   default=None,   help="No oracle curriculum → Stage 2")
    p.add_argument("--no-stage1",   default=None,   help="Stage 2 from base (no SFT init)")
    # Eval settings
    p.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--n-configs", type=int, default=10)
    p.add_argument("--n-per-config", type=int, default=5)
    p.add_argument("--item-desc", default="a used laptop computer")
    p.add_argument("--max-rounds", type=int, default=6)
    p.add_argument("--max-ctx-len", type=int, default=768)
    p.add_argument("--max-gen-len", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    from train_stage2 import load_from_stage1
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    sim_cfg = LASHConfig(
        model=args.model,
        item_description=args.item_desc,
        max_rounds=args.max_rounds,
    )

    eval_kwargs = dict(
        tokenizer=tokenizer,
        sim_cfg=sim_cfg,
        n_configs=args.n_configs,
        n_per_config=args.n_per_config,
        max_ctx_len=args.max_ctx_len,
        max_gen_len=args.max_gen_len,
        device=device,
        seed=args.seed,
    )

    # ── Load and evaluate each variant ────────────────────────────────────
    variants: list[tuple[str, Optional[str], bool]] = [
        ("Full LASH",         args.full_lash,   False),
        ("Full LASH (no-hyp)", args.full_lash,  True),   # z_t zeroed
        ("Stage 1 only",      args.stage1_only, False),
        ("No Oracle → Stage2", args.no_oracle,  False),
        ("No Stage1 → Stage2", args.no_stage1,  False),
    ]

    results: list[tuple[str, dict]] = []

    for label, ckpt_dir, zero_z in variants:
        if ckpt_dir is None:
            print(f"  [{label}] skipped (no checkpoint provided)")
            continue
        print(f"  Loading {label}: {ckpt_dir}")
        model, _ = load_from_stage1(ckpt_dir, args.model, device)
        model.eval()

        m = run_ablation_eval(model, zero_z=zero_z, **eval_kwargs)
        results.append((label, m))
        print(f"    deal_rate={m['deal_rate']:.3f}  efficiency={m['efficiency']:.3f}")

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # ── Results table ──────────────────────────────────────────────────────
    if not results:
        print("No results to display.")
        return

    full_deal = results[0][1]["deal_rate"] if results else 1.0

    print("\n" + "=" * 75)
    print("ABLATION STUDY RESULTS")
    print("=" * 75)
    print(f"  {'Variant':<26} {'Deal Rate':>10} {'Efficiency':>11} "
          f"{'R_buyer':>9} {'R_seller':>9} {'Δ Deal%':>8}")
    print(f"  {'-'*73}")
    for label, m in results:
        delta = (m["deal_rate"] - full_deal) / full_deal * 100 if full_deal > 0 else 0.0
        delta_str = f"{delta:+.1f}%" if label != "Full LASH" else "—"
        print(
            f"  {label:<26} {m['deal_rate']:>10.3f} {m['efficiency']:>11.3f} "
            f"{m['avg_buyer_reward']:>9.3f} {m['avg_seller_reward']:>9.3f} {delta_str:>8}"
        )
    print("=" * 75)
    print(f"  n_episodes per variant = {args.n_configs * args.n_per_config}")

    if len(results) >= 2:
        print("\n  Key takeaways:")
        for label, m in results[1:]:
            diff = m["deal_rate"] - full_deal
            direction = "↓" if diff < -0.02 else ("↑" if diff > 0.02 else "≈")
            print(f"    Removing '{label}': deal rate {direction} ({diff:+.3f})")


if __name__ == "__main__":
    main()

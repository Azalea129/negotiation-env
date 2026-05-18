"""
Empirical equilibrium analysis of LASH agents in A2A negotiation.

Runs a large LASH-vs-LASH tournament across diverse (v_b, v_s, δ_b, δ_s) type configs
and measures where the empirical distribution of deal prices and surplus splits lands
relative to classical game-theoretic benchmarks.

Benchmarks compared:
  Nash bargaining solution   p* = (v_b + v_s) / 2
    (equal split of ZOPA, symmetric Nash product)
  Rubinstein alternating-offers equilibrium
    p* = v_s + (v_b - v_s) * δ_s * (1-δ_b) / (1 - δ_b*δ_s)   [buyer opens]
    Captures asymmetric patience effects; unique SPNE.

Metrics:
  deal_rate       — fraction of episodes ending in deal
  efficiency      — total_welfare / max_possible_welfare (= v_b - v_s)
  buyer_share     — buyer_surplus / total_welfare
  nash_dev        — |deal_price - nash_price| / (v_b - v_s)   (ZOPA-normalised)
  rubinstein_dev  — |deal_price - rubinstein_price| / (v_b - v_s)
  avg_round       — average round at which deal closes

Usage:
  python -m eval.equilibrium --stage2-dir checkpoints/stage2/final
  python -m eval.equilibrium --stage2-dir checkpoints/stage2/final --n-episodes 200
"""

import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer

from lash.config import LASHConfig
from lash.grpo_env import GRPORolloutEnv, RolloutEpisode
from lash.types import BuyerType, SellerType, sample_types


# ── Game-theoretic benchmarks ──────────────────────────────────────────────

def nash_price(v_b: float, v_s: float) -> float:
    """Nash bargaining solution (equal ZOPA split)."""
    return (v_b + v_s) / 2.0


def rubinstein_price(v_b: float, v_s: float, delta_b: float, delta_s: float) -> float:
    """
    Rubinstein (1982) alternating-offers SPNE, buyer-opens protocol.

    Buyer's equilibrium share of ZOPA = (1 - δ_s) / (1 - δ_b * δ_s)
    Deal price = v_s + ZOPA * (1 - buyer_share)   [seller gets the complement]
    """
    denom = 1.0 - delta_b * delta_s
    if abs(denom) < 1e-9:
        return nash_price(v_b, v_s)
    buyer_share = (1.0 - delta_s) / denom
    return v_s + (v_b - v_s) * (1.0 - buyer_share)


# ── Outcome record ─────────────────────────────────────────────────────────

@dataclass
class EvalOutcome:
    buyer_type: BuyerType
    seller_type: SellerType
    deal_reached: bool
    deal_price: Optional[float]
    buyer_reward: float
    seller_reward: float
    n_turns: int


def run_tournament(
    env: GRPORolloutEnv,
    n_configs: int,
    n_per_config: int,
    seed: int = 0,
) -> list[EvalOutcome]:
    outcomes: list[EvalOutcome] = []
    for i in range(n_configs):
        buyer_type, seller_type = sample_types(seed=seed + i)
        for _ in range(n_per_config):
            ep = env.run_episode(buyer_type, seller_type)
            outcomes.append(EvalOutcome(
                buyer_type=buyer_type,
                seller_type=seller_type,
                deal_reached=ep.deal_reached,
                deal_price=ep.deal_price,
                buyer_reward=ep.buyer_reward,
                seller_reward=ep.seller_reward,
                n_turns=len(ep.turns),
            ))
    return outcomes


# ── Metric computation ────────────────────────────────────────────────────

def compute_metrics(outcomes: list[EvalOutcome]) -> dict:
    n = len(outcomes)
    deals = [o for o in outcomes if o.deal_reached and o.deal_price is not None]

    deal_rate = len(deals) / n

    if not deals:
        return {
            "n_episodes": n, "deal_rate": 0.0,
            "efficiency": 0.0, "buyer_share": float("nan"),
            "nash_dev": float("nan"), "rubinstein_dev": float("nan"),
            "avg_round": float("nan"),
        }

    efficiencies, buyer_shares, nash_devs, rubinstein_devs, rounds = [], [], [], [], []

    for o in deals:
        v_b = o.buyer_type.reservation_price
        v_s = o.seller_type.reservation_price
        zopa = v_b - v_s
        price = o.deal_price

        bs = max(0.0, v_b - price)
        ss = max(0.0, price - v_s)
        tw = bs + ss

        efficiencies.append(tw / zopa if zopa > 0 else 0.0)
        buyer_shares.append(bs / tw if tw > 0 else 0.5)
        nash_devs.append(abs(price - nash_price(v_b, v_s)) / zopa)
        rubinstein_devs.append(abs(
            price - rubinstein_price(v_b, v_s, o.buyer_type.delta, o.seller_type.delta)
        ) / zopa)
        rounds.append(o.n_turns // 2)   # turns → rounds

    return {
        "n_episodes": n,
        "deal_rate": deal_rate,
        "efficiency": statistics.mean(efficiencies),
        "buyer_share": statistics.mean(buyer_shares),
        "nash_dev": statistics.mean(nash_devs),
        "rubinstein_dev": statistics.mean(rubinstein_devs),
        "avg_round": statistics.mean(rounds),
    }


def print_distribution(outcomes: list[EvalOutcome], n_bins: int = 5) -> None:
    """Print a simple ASCII histogram of deal price relative to ZOPA midpoint."""
    deals = [o for o in outcomes if o.deal_reached and o.deal_price is not None]
    if not deals:
        return

    # Normalize: 0 = seller reservation, 1 = buyer reservation
    positions = []
    for o in deals:
        v_b = o.buyer_type.reservation_price
        v_s = o.seller_type.reservation_price
        zopa = v_b - v_s
        if zopa > 0:
            positions.append((o.deal_price - v_s) / zopa)

    if not positions:
        return

    bins = [0] * n_bins
    for p in positions:
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx] += 1

    print("\nDeal price distribution (normalized: 0=v_s, 1=v_b, 0.5=Nash):")
    labels = [f"{i/n_bins:.1f}–{(i+1)/n_bins:.1f}" for i in range(n_bins)]
    max_count = max(bins) if bins else 1
    for label, count in zip(labels, bins):
        bar = "█" * int(count / max_count * 30)
        pct = count / len(positions) * 100
        midpoint_marker = " ← Nash" if label == f"{(n_bins//2)/n_bins:.1f}–{(n_bins//2+1)/n_bins:.1f}" else ""
        print(f"  [{label}] {bar:<30} {pct:5.1f}%{midpoint_marker}")


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LASH empirical equilibrium analysis")
    p.add_argument("--stage2-dir", required=True)
    p.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--n-configs", type=int, default=20,
                   help="Distinct buyer/seller type configurations")
    p.add_argument("--n-per-config", type=int, default=10,
                   help="Episodes per config (total = n_configs × n_per_config)")
    p.add_argument("--item-desc", default="a used laptop computer")
    p.add_argument("--max-rounds", type=int, default=6)
    p.add_argument("--max-ctx-len", type=int, default=768)
    p.add_argument("--max-gen-len", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    from train_stage2 import load_from_stage1
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading {args.stage2_dir} ...")
    model, _ = load_from_stage1(args.stage2_dir, args.model, device)
    model.eval()

    sim_cfg = LASHConfig(
        model=args.model,
        item_description=args.item_desc,
        max_rounds=args.max_rounds,
    )
    env = GRPORolloutEnv(
        model=model,
        tokenizer=tokenizer,
        sim_config=sim_cfg,
        max_ctx_len=args.max_ctx_len,
        max_gen_len=args.max_gen_len,
        device=device,
    )

    total = args.n_configs * args.n_per_config
    print(f"Running tournament: {args.n_configs} configs × {args.n_per_config} episodes = {total} total")

    with torch.no_grad():
        outcomes = run_tournament(env, args.n_configs, args.n_per_config, args.seed)

    m = compute_metrics(outcomes)

    print("\n" + "=" * 60)
    print("EMPIRICAL EQUILIBRIUM ANALYSIS")
    print("=" * 60)
    print(f"  Episodes         : {m['n_episodes']}")
    print(f"  Deal rate        : {m['deal_rate']:.3f}")
    print(f"  Efficiency       : {m['efficiency']:.3f}  (welfare / ZOPA)")
    print(f"  Buyer share      : {m['buyer_share']:.3f}  (0.5 = equal split)")
    print(f"  Avg deal round   : {m['avg_round']:.2f}")
    print(f"\n  Deviation from Nash bargaining    : {m['nash_dev']:.3f}  (ZOPA-normalised |Δ|)")
    print(f"  Deviation from Rubinstein SPNE     : {m['rubinstein_dev']:.3f}  (ZOPA-normalised |Δ|)")
    print(f"\n  Interpretation:")
    closer = "Nash" if m["nash_dev"] <= m["rubinstein_dev"] else "Rubinstein"
    print(f"    Empirical equilibrium is closer to {closer} solution.")
    if m["buyer_share"] > 0.55:
        print(f"    Buyer-favourable split — opener's advantage visible.")
    elif m["buyer_share"] < 0.45:
        print(f"    Seller-favourable split — possibly seller patience effect.")
    else:
        print(f"    Roughly equal surplus split.")

    print_distribution(outcomes)
    print("=" * 60)


if __name__ == "__main__":
    main()

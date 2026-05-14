"""
Batch experiment runner and result serialization.
"""

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .config import Condition, NegotiationConfig
from .env import EpisodeResult, NegotiationEnv
from .types import sample_types


def run_batch(
    n_episodes: int,
    config: NegotiationConfig,
    base_seed: int = 42,
    verbose: bool = True,
) -> list[EpisodeResult]:
    env = NegotiationEnv(config)
    results = []

    for i in range(n_episodes):
        seed = base_seed + i
        buyer_type, seller_type = sample_types(seed=seed)

        if verbose:
            print(f"[{i+1}/{n_episodes}] seed={seed} | "
                  f"v_b=${buyer_type.reservation_price:.0f} u={buyer_type.urgency:.2f} | "
                  f"v_s=${seller_type.reservation_price:.0f} p={seller_type.inventory_pressure:.2f}")

        try:
            result = env.run(buyer_type=buyer_type, seller_type=seller_type, seed=seed)
            results.append(result)
            if verbose:
                status = f"DEAL @ ${result.deal_price:.0f} (round {result.deal_round})" \
                    if result.deal_reached else f"NO DEAL ({result.termination})"
                print(f"  → {status} | welfare={result.total_welfare:.1f}")
        except Exception as e:
            print(f"  → ERROR: {e}")

    return results


def run_condition_comparison(
    n_episodes: int,
    base_config: NegotiationConfig,
    conditions: Optional[list[Condition]] = None,
    base_seed: int = 42,
    verbose: bool = True,
) -> dict[str, list[EpisodeResult]]:
    """Run the same set of episodes across multiple conditions (same seeds = same types)."""
    if conditions is None:
        conditions = list(Condition)

    all_results = {}
    for condition in conditions:
        if verbose:
            print(f"\n{'='*50}")
            print(f"CONDITION: {condition.value}")
            print(f"{'='*50}")
        config = NegotiationConfig(
            **{**vars(base_config), "condition": condition}
        )
        all_results[condition.value] = run_batch(n_episodes, config, base_seed, verbose)

    return all_results


def summarize(results: list[EpisodeResult]) -> dict:
    n = len(results)
    deals = [r for r in results if r.deal_reached]
    no_deals = [r for r in results if not r.deal_reached]

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "n_episodes": n,
        "agreement_rate": len(deals) / n if n else 0,
        "avg_deal_price": mean([r.deal_price for r in deals]),
        "avg_deal_round": mean([r.deal_round for r in deals]),
        "avg_buyer_surplus": mean([r.buyer_surplus for r in deals]),
        "avg_seller_surplus": mean([r.seller_surplus for r in deals]),
        "avg_total_welfare": mean([r.total_welfare for r in deals]),
        "avg_total_welfare_all": mean([r.total_welfare for r in results]),
        "failure_rate": len(no_deals) / n if n else 0,
        "termination_counts": {
            t: sum(1 for r in results if r.termination == t)
            for t in ["accept", "reject", "max_rounds"]
        },
    }


def _result_to_dict(result: EpisodeResult) -> dict:
    d = asdict(result)
    # Convert nested dataclasses that asdict doesn't fully handle
    d["condition"] = result.condition
    return d


def save_results(results: list[EpisodeResult], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = [_result_to_dict(r) for r in results]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} episodes → {path}")


def save_condition_comparison(
    all_results: dict[str, list[EpisodeResult]], path: str
) -> None:
    output = {
        condition: {
            "summary": summarize(results),
            "episodes": [_result_to_dict(r) for r in results],
        }
        for condition, results in all_results.items()
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Saved condition comparison → {path}")

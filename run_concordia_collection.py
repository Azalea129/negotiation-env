"""
LASH Stage 1 multi-game data collection script.

Runs mixed-motive game episodes and saves structured BDI CoT training pairs
alongside the A2A negotiation data. All output goes to the same data/ directory —
training_pairs.jsonl contains turns from ALL game types, giving the model exposure
to diverse strategic contexts for Stage 1 representation alignment (SFT).

Usage:
  # All games, 20 episodes each, OpenAI
  python run_concordia_collection.py --n 20

  # Specific games only
  python run_concordia_collection.py --n 20 --games prisoners_dilemma public_goods

  # Local Llama via vLLM
  python run_concordia_collection.py --n 20 \\
      --model meta-llama/Meta-Llama-3-8B-Instruct \\
      --api-base http://localhost:8000/v1 --api-key token

Available games:
  prisoners_dilemma    2-player repeated cooperation/defection
  public_goods         N-player contribution game (N=4 by default)
  resource_allocation  N-player sealed-bid multi-item auction
  haggling             2-player bilateral price negotiation (Stage 1 variant)
  stag_hunt            2-player coordination game (held-out evaluation candidate)
  ultimatum            2-player Ultimatum Game (fairness norm reasoning)
  liars_dice           2-player Liar's Dice (deception detection)
  mafia                5-player Mafia (hidden role inference, multi-party ToM)
"""

import argparse
import traceback

from lash.concordia_adapter import GAME_REGISTRY, run_multitask_collection
from lash.config import LASHConfig
from lash.data_collector import collection_stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LASH Stage 1 multi-game data collection")
    p.add_argument("--n", type=int, default=10,
                   help="Episodes per game type")
    p.add_argument("--games", nargs="+", default=None,
                   choices=list(GAME_REGISTRY.keys()),
                   help="Which games to run (default: all)")
    p.add_argument("--model", default="gpt-4o")
    p.add_argument("--api-base", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--output-dir", default="data")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--parallel", type=int, default=8,
                   help="Concurrent episodes per game type (default: 8)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    config = LASHConfig(
        model=args.model,
        temperature=args.temperature,
        api_base=args.api_base,
        api_key=args.api_key,
    )

    games = args.games or list(GAME_REGISTRY.keys())

    print(f"Model  : {config.model}")
    if config.api_base:
        print(f"API    : {config.api_base}")
    print(f"Games  : {', '.join(games)}")
    print(f"N/game : {args.n}")
    print(f"Output : {args.output_dir}/")

    game_stats = run_multitask_collection(
        config=config,
        n_episodes_per_game=args.n,
        output_dir=args.output_dir,
        games=games,
        seed=args.seed,
        parallel=args.parallel,
        verbose=True,
    )

    print(f"\n── Per-game training pairs ──────────────────────────")
    for game, n_pairs in game_stats.items():
        print(f"  {game:<25} {n_pairs:>5} pairs")

    overall = collection_stats(args.output_dir)
    print(f"\n── Overall data/{args.output_dir} ─────────────────────────────")
    print(f"  Total episodes    : {overall['episodes']}")
    print(f"  Training pairs    : {overall['training_pairs']}")


if __name__ == "__main__":
    main()

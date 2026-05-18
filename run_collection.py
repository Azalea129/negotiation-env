"""
Stage 1 data collection script.

Runs N negotiation episodes and saves:
  data/episodes.jsonl        — full episode records
  data/training_pairs.jsonl  — (context, belief_gt, intention_gt, message) tuples

Usage:
  # OpenAI (default)
  python run_collection.py --n 50

  # Local Llama via vLLM
  python run_collection.py --n 50 --model meta-llama/Meta-Llama-3-8B-Instruct \\
      --api-base http://localhost:8000/v1 --api-key token

  # Local Llama via Ollama
  python run_collection.py --n 50 --model llama3 \\
      --api-base http://localhost:11434/v1 --api-key ollama
"""

import argparse
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from lash import A2ANegotiationEnv, LASHConfig, collection_stats, save_episode

_print_lock = threading.Lock()

def _safe_print(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LASH Stage 1 data collection")
    p.add_argument("--n", type=int, default=10, help="Number of episodes to run")
    p.add_argument("--model", default="gpt-4o", help="Model name")
    p.add_argument("--api-base", default=None, help="API base URL (for local vLLM/Ollama)")
    p.add_argument("--api-key", default=None, help="API key override")
    p.add_argument("--max-rounds", type=int, default=10, help="Max rounds per episode")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--lambda-selfishness", type=float, default=1.0,
                   help="0=cooperative reward, 1=selfish reward")
    p.add_argument("--output-dir", default="data", help="Directory for output JSONL files")
    p.add_argument("--seed", type=int, default=None, help="Base random seed (incremented per episode)")
    p.add_argument("--parallel", type=int, default=8,
                   help="Number of episodes to run concurrently (default: 8)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    config = LASHConfig(
        model=args.model,
        temperature=args.temperature,
        api_base=args.api_base,
        api_key=args.api_key,
        max_rounds=args.max_rounds,
        lambda_selfishness=args.lambda_selfishness,
    )
    env = A2ANegotiationEnv(config)

    print(f"Model    : {config.model}")
    if config.api_base:
        print(f"API      : {config.api_base}")
    print(f"Output   : {args.output_dir}/")
    print(f"Episodes : {args.n}  |  parallel={args.parallel}\n")

    def _run_one(i: int):
        seed = (args.seed + i) if args.seed is not None else None
        ep = env.run(seed=seed)
        n_pairs = save_episode(ep, args.output_dir)
        return i, ep, n_pairs

    completed = 0
    try:
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {executor.submit(_run_one, i): i for i in range(args.n)}
            for future in as_completed(futures):
                try:
                    i, ep, n_pairs = future.result()
                    completed += 1
                    status = "DEAL" if ep.deal_reached else ep.termination.upper()
                    price_str = f"${ep.deal_price:.0f}" if ep.deal_price else "-"
                    _safe_print(
                        f"[{completed:>4}/{args.n}] {ep.episode_id}  {status:<12} "
                        f"price={price_str:<8} welfare={ep.total_welfare:>7.1f}  pairs={n_pairs}"
                    )
                except Exception:
                    completed += 1
                    _safe_print(f"[{completed:>4}/{args.n}] ERROR — skipping episode")
                    traceback.print_exc()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    stats = collection_stats(args.output_dir)
    print(f"\n── Collection complete ─────────────────────────────")
    print(f"Episodes       : {stats['episodes']}")
    print(f"Deal rate      : {stats['deal_rate']:.1%}")
    print(f"Avg welfare    : {stats['avg_welfare']:.1f}")
    print(f"Training pairs : {stats['training_pairs']}")


if __name__ == "__main__":
    main()

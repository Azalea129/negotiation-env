"""
Zero-shot generalization evaluation on StagHunt (held-out game).

StagHunt was never seen during Stage 1 (trained on PD / PublicGoods / RA / Haggling)
nor Stage 2 (trained on A2A negotiation only). This tests whether the ToM latent
space generalizes to a structurally different coordination problem.

Compares three conditions:
  full_lash  — hypothesis modules active (z_t from Pass 1 conditions generation)
  no_lash    — same weights, z_t bypassed (model.base.generate directly)
               isolates the hypothesis attention module's contribution
  random     — random STAG/HARE baseline (upper bound = 4.0 avg payoff, lower = 2.0)

Also runs on PD (Stage 1 trained) for reference — should show LASH > no_lash there too.

Usage:
  python -m eval.zero_shot --stage2-dir checkpoints/stage2/final
  python -m eval.zero_shot --stage2-dir checkpoints/stage2/final --n-episodes 50
"""

import argparse
import random
import statistics
from pathlib import Path
from typing import Callable

import torch
from transformers import AutoTokenizer

from lash.concordia_adapter import (
    _AGENT_SYSTEM, _SH_RULES, _SH_ACTION_FORMAT,
    _PD_RULES, _PD_ACTION_FORMAT,
    _parse_sh_action, _parse_pd_action,
    _sh_payoff, _pd_payoff,
)


# ── Prompt builders ────────────────────────────────────────────────────────

def _sh_prompt(agent: str, opp: str, n_rounds: int, history: list[tuple], rnd: int) -> str:
    system = _AGENT_SYSTEM.format(
        agent_name=agent,
        game_name="Stag Hunt",
        game_rules=_SH_RULES.format(n_rounds=n_rounds),
        private_info="No private information — symmetric coordination game.",
        opponent_description=opp,
        action_format=_SH_ACTION_FORMAT,
    )
    obs = [f"=== STAG HUNT — Round {rnd + 1}/{n_rounds} ==="]
    if history:
        last = history[-1]
        obs.append(f"Last round: {agent}={last[0]}, {opp}={last[1]}")
    obs.append("\nYour choice:")
    hist_text = ""
    if history:
        rows = [f"  Round {i+1}: {agent}={h[0]}, {opp}={h[1]}" for i, h in enumerate(history)]
        hist_text = "=== History ===\n" + "\n".join(rows) + "\n\n"
    return f"{system}\n\n{hist_text}" + "\n".join(obs)


def _pd_prompt(agent: str, opp: str, n_rounds: int, last_actions: list, rnd: int) -> str:
    system = _AGENT_SYSTEM.format(
        agent_name=agent,
        game_name="Prisoner's Dilemma",
        game_rules=_PD_RULES.format(n_rounds=n_rounds),
        private_info="No private information — symmetric game.",
        opponent_description=opp,
        action_format=_PD_ACTION_FORMAT,
    )
    obs = [f"=== PRISONER'S DILEMMA — Round {rnd + 1}/{n_rounds} ==="]
    if last_actions[0]:
        obs.append(f"Last round: {agent}={last_actions[0]}, {opp}={last_actions[1]}")
    obs.append("\nYour move:")
    return f"{system}\n\n" + "\n".join(obs)


# ── Generation ─────────────────────────────────────────────────────────────

@torch.no_grad()
def _generate(model, tokenizer, prompt: str, use_lash: bool, max_new: int, device: str) -> str:
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    gen_kwargs = dict(
        max_new_tokens=max_new,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if use_lash:
        out = model.generate(ids, mask, **gen_kwargs)
    else:
        # Bypass hypothesis modules — direct base LM, no z_t conditioning
        out = model.base.generate(input_ids=ids, attention_mask=mask, **gen_kwargs)
    return tokenizer.decode(out[0], skip_special_tokens=True)


# ── Episode runners ────────────────────────────────────────────────────────

@torch.no_grad()
def run_sh_episode(
    model, tokenizer, n_rounds: int, use_lash: bool, max_new: int, device: str
) -> dict:
    names = ["Alpha", "Beta"]
    history: list[tuple] = []   # [(alpha_choice, beta_choice), ...]
    payoffs = [0.0, 0.0]

    for rnd in range(n_rounds):
        choices = []
        for i, name in enumerate(names):
            opp = names[1 - i]
            # Each agent only sees its own perspective (own choice first in history)
            agent_hist = [(h[i], h[1 - i]) for h in history]
            prompt = _sh_prompt(name, opp, n_rounds, agent_hist, rnd)
            raw = _generate(model, tokenizer, prompt, use_lash, max_new, device)
            choices.append(_parse_sh_action(raw))
        history.append((choices[0], choices[1]))
        p0, p1 = _sh_payoff(choices[0], choices[1])
        payoffs[0] += p0
        payoffs[1] += p1

    coord = sum(1 for h in history if h[0] == "STAG" and h[1] == "STAG")
    stag = sum(1 for h in history for c in h if c == "STAG")
    return {
        "avg_payoff": (payoffs[0] + payoffs[1]) / (2 * n_rounds),
        "coordination_rate": coord / n_rounds,
        "stag_rate": stag / (2 * n_rounds),
    }


@torch.no_grad()
def run_pd_episode(
    model, tokenizer, n_rounds: int, use_lash: bool, max_new: int, device: str
) -> dict:
    names = ["Alice", "Bob"]
    last = [None, None]
    payoffs = [0.0, 0.0]
    coop_rounds = 0

    for rnd in range(n_rounds):
        choices = []
        for i, name in enumerate(names):
            opp = names[1 - i]
            prompt = _pd_prompt(name, opp, n_rounds, [last[i], last[1 - i]], rnd)
            raw = _generate(model, tokenizer, prompt, use_lash, max_new, device)
            choices.append(_parse_pd_action(raw))
        last = choices
        p0, p1 = _pd_payoff(choices[0], choices[1])
        payoffs[0] += p0
        payoffs[1] += p1
        if choices[0] == "COOPERATE" and choices[1] == "COOPERATE":
            coop_rounds += 1

    return {
        "avg_payoff": (payoffs[0] + payoffs[1]) / (2 * n_rounds),
        "cooperation_rate": coop_rounds / n_rounds,
    }


def run_random_sh(n_episodes: int, n_rounds: int) -> dict:
    rng = random.Random(0)
    all_coord, all_payoff = [], []
    for _ in range(n_episodes):
        coord, total = 0, 0.0
        for _ in range(n_rounds):
            a = rng.choice(["STAG", "HARE"])
            b = rng.choice(["STAG", "HARE"])
            p0, p1 = _sh_payoff(a, b)
            total += (p0 + p1) / 2
            if a == "STAG" and b == "STAG":
                coord += 1
        all_coord.append(coord / n_rounds)
        all_payoff.append(total / n_rounds)
    return {
        "avg_payoff": statistics.mean(all_payoff),
        "coordination_rate": statistics.mean(all_coord),
        "stag_rate": 0.5,
    }


# ── Aggregate eval ─────────────────────────────────────────────────────────

def eval_game(
    run_fn: Callable, model, tokenizer, n_episodes: int,
    use_lash: bool, max_new: int, device: str
) -> dict:
    results = [run_fn(model, tokenizer, use_lash=use_lash, max_new_tokens=max_new, device=device)
               for _ in range(n_episodes)]
    return {k: statistics.mean(r[k] for r in results) for k in results[0]}


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LASH zero-shot generalization eval")
    p.add_argument("--stage2-dir", required=True, help="Stage 2 checkpoint directory")
    p.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--n-episodes", type=int, default=30)
    p.add_argument("--n-rounds", type=int, default=6)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    from train_stage2 import load_from_stage1   # reuse loader
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading {args.stage2_dir} ...")
    model, _ = load_from_stage1(args.stage2_dir, args.model, device)
    model.eval()

    # Partial wrappers with fixed signature
    def sh_lash(model, tok, use_lash, max_new_tokens, device):
        return run_sh_episode(model, tok, args.n_rounds, use_lash, max_new_tokens, device)
    def pd_lash(model, tok, use_lash, max_new_tokens, device):
        return run_pd_episode(model, tok, args.n_rounds, use_lash, max_new_tokens, device)

    print(f"\nRunning zero-shot StagHunt ({args.n_episodes} episodes each) ...")
    sh_full  = eval_game(sh_lash, model, tokenizer, args.n_episodes, True,  args.max_new_tokens, device)
    sh_base  = eval_game(sh_lash, model, tokenizer, args.n_episodes, False, args.max_new_tokens, device)
    sh_rand  = run_random_sh(args.n_episodes, args.n_rounds)

    print(f"Running PD reference ({args.n_episodes} episodes each) ...")
    pd_full = eval_game(pd_lash, model, tokenizer, args.n_episodes, True,  args.max_new_tokens, device)
    pd_base = eval_game(pd_lash, model, tokenizer, args.n_episodes, False, args.max_new_tokens, device)

    # ── Results table ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("ZERO-SHOT EVALUATION RESULTS")
    print("=" * 65)
    print(f"\n{'StagHunt (held-out — zero-shot)'}")
    print(f"  {'Condition':<20} {'Coord. Rate':>12} {'Avg Payoff':>12} {'STAG Rate':>10}")
    print(f"  {'-'*54}")
    for label, r in [("Full LASH", sh_full), ("No-LASH (base)", sh_base), ("Random", sh_rand)]:
        print(f"  {label:<20} {r['coordination_rate']:>11.3f}  {r['avg_payoff']:>11.3f}  {r.get('stag_rate', 0.5):>9.3f}")

    print(f"\n{'PD (Stage 1 trained — reference)'}")
    print(f"  {'Condition':<20} {'Coop. Rate':>12} {'Avg Payoff':>12}")
    print(f"  {'-'*44}")
    for label, r in [("Full LASH", pd_full), ("No-LASH (base)", pd_base)]:
        print(f"  {label:<20} {r['cooperation_rate']:>11.3f}  {r['avg_payoff']:>11.3f}")
    print("=" * 65)


if __name__ == "__main__":
    main()

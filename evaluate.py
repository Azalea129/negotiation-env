"""
GRPO 학습 모델 vs 미학습 베이스라인 평가 스크립트.

각 세팅(1:1, 1:n, n:1) × 4가지 매칭 조건(grpo↔grpo / grpo↔base / base↔grpo / base↔base)
으로 n_episodes회 시뮬레이션하고 결과를 비교한다.

사용법:
  python evaluate.py \\
    --base_model_id   meta-llama/Meta-Llama-3.1-8B-Instruct \\
    --trained_path    checkpoints/grpo_run/final \\
    --n_episodes      100 \\
    --n_parties       3 \\
    --output_dir      eval_results/

  # 학습 모델 없이 베이스라인만 실행 (미학습 Llama vs 미학습 Llama):
  python evaluate.py --base_model_id meta-llama/... --skip_trained

매칭 코드:
  GG = GRPO buyer  vs GRPO seller
  GB = GRPO buyer  vs Base seller  (구매자 역할 GRPO 우위 측정)
  BG = Base buyer  vs GRPO seller  (판매자 역할 GRPO 우위 측정)
  BB = Base buyer  vs Base seller  (베이스라인)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from negotiation_project import Condition, NegotiationConfig
from negotiation_project.agents import NegotiationAgent, make_buyer_agent, make_seller_agent
from negotiation_project.env import NegotiationEnv
from negotiation_project.multi_env import MultiEpisodeResult, NVsOneEnv, OneVsNEnv
from negotiation_project.types import BuyerType, SellerType, sample_types


# ---------------------------------------------------------------------------
# 모델 로딩
# ---------------------------------------------------------------------------

def load_base_model(model_id: str):
    print(f"  베이스라인 모델 로딩: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    return model, tokenizer


def load_trained_model(base_model_id: str, trained_path: str):
    from peft import PeftModel
    print(f"  GRPO 학습 모델 로딩: {trained_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, trained_path)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# 에이전트 팩토리 빌더
# ---------------------------------------------------------------------------

def make_local_factory(model, tokenizer, temperature: float = 0.7):
    """LocalNegotiationAgent 팩토리 (inference 전용, log-prob 기록 없음)."""
    from negotiation_project.grpo.local_agent import LocalNegotiationAgent

    def factory(role: str, system_prompt: str, config: NegotiationConfig):
        return LocalNegotiationAgent(
            role=role,
            system_prompt=system_prompt,
            model=model,
            tokenizer=tokenizer,
            config=config,
            temperature=temperature,
            max_new_tokens=256,
        )
    return factory


# ---------------------------------------------------------------------------
# 1:1 평가
# ---------------------------------------------------------------------------

def eval_1v1(
    buyer_factory: Callable,
    seller_factory: Callable,
    config: NegotiationConfig,
    n_episodes: int,
    base_seed: int = 0,
    verbose: bool = False,
) -> list[dict]:
    from negotiation_project.agents import (
        _BUYER_SYSTEM, _SELLER_SYSTEM, _CONDITION_INSTRUCTIONS,
    )
    results = []

    for i in range(n_episodes):
        seed = base_seed + i
        buyer_type, seller_type = sample_types(seed=seed)
        buyer_delta = config.delta_from_level(buyer_type.urgency)
        seller_delta = config.delta_from_level(seller_type.inventory_pressure)
        lam = config.lambda_selfishness

        # 에이전트 생성
        from negotiation_project.agents import (
            _BUYER_SYSTEM, _SELLER_SYSTEM, _CONDITION_INSTRUCTIONS,
        )
        buyer_system = _BUYER_SYSTEM.format(
            item_description=config.item_description,
            reservation_price=buyer_type.reservation_price,
            urgency_label=buyer_type.urgency_label(),
            urgency=buyer_type.urgency,
            delta=buyer_delta,
            outside_option=buyer_type.outside_option,
            max_rounds=config.max_rounds,
            condition_instructions=_CONDITION_INSTRUCTIONS[config.condition],
        )
        seller_system = _SELLER_SYSTEM.format(
            item_description=config.item_description,
            reservation_price=seller_type.reservation_price,
            pressure_label=seller_type.pressure_label(),
            pressure=seller_type.inventory_pressure,
            delta=seller_delta,
            max_rounds=config.max_rounds,
            condition_instructions=_CONDITION_INSTRUCTIONS[config.condition],
        )

        buyer = buyer_factory("buyer", buyer_system, config)
        seller = seller_factory("seller", seller_system, config)

        # 에피소드 실행 (env.py 루프 재현)
        current_price = None
        rnd = 0
        termination = "max_rounds"
        deal_price = None

        resp = buyer.respond(None)
        if resp["action"] == "reject":
            termination = "reject"
        elif resp["price"] is not None:
            current_price = resp["price"]
            incoming = resp["visible_message"]

            max_turns = config.max_rounds * 2
            turn = 1
            while turn < max_turns:
                resp = seller.respond(incoming)
                turn += 1
                if resp["action"] == "accept":
                    deal_price, termination = current_price, "accept"
                    break
                if resp["action"] == "reject":
                    termination = "reject"
                    break
                if resp["price"] is not None:
                    current_price = resp["price"]
                incoming = resp["visible_message"]
                rnd += 1
                if rnd >= config.max_rounds:
                    break

                resp = buyer.respond(incoming)
                turn += 1
                if resp["action"] == "accept":
                    deal_price, termination = current_price, "accept"
                    break
                if resp["action"] == "reject":
                    termination = "reject"
                    break
                if resp["price"] is not None:
                    current_price = resp["price"]
                incoming = resp["visible_message"]
        else:
            incoming = resp["visible_message"]

        # 보상 계산
        if deal_price is not None:
            bs = max(0.0, buyer_type.reservation_price - deal_price) * (buyer_delta ** rnd)
            ss = max(0.0, deal_price - seller_type.reservation_price) * (seller_delta ** rnd)
            tw = bs + ss
            br = lam * bs + (1 - lam) * tw
            sr = lam * ss + (1 - lam) * tw
        else:
            bs = ss = tw = br = sr = 0.0

        r = {
            "setting": "1:1", "n_parties": 1, "episode": i,
            "deal_reached": deal_price is not None,
            "deal_price": deal_price, "deal_round": rnd if deal_price else None,
            "termination": termination,
            "buyer_reward": br, "seller_reward": sr, "total_welfare": tw,
            "buyer_surplus": bs, "seller_surplus": ss,
            "v_b": buyer_type.reservation_price, "v_s": seller_type.reservation_price,
            "max_welfare": max(0.0, buyer_type.reservation_price - seller_type.reservation_price),
        }
        results.append(r)
        if verbose:
            status = f"DEAL@${deal_price:.0f}" if deal_price else f"NO DEAL({termination})"
            print(f"    ep{i:3d}: {status} | welfare={tw:.1f}")

    return results


# ---------------------------------------------------------------------------
# 1:n 평가
# ---------------------------------------------------------------------------

def eval_1vn(
    buyer_factory: Callable,
    seller_factory: Callable,
    config: NegotiationConfig,
    n_episodes: int,
    n_sellers: int = 3,
    base_seed: int = 0,
    verbose: bool = False,
) -> list[dict]:
    env = OneVsNEnv(config, n_sellers=n_sellers)
    results = []

    for i in range(n_episodes):
        seed = base_seed + i
        buyer_type, _ = sample_types(seed=seed)
        seller_types = []
        for j in range(n_sellers):
            _, st = sample_types(seed=seed * 100 + j + 1)
            seller_types.append(st)

        result = env.run(buyer_type, seller_types, buyer_factory, seller_factory)
        r = _multi_result_to_dict(result, i)
        r["v_b"] = buyer_type.reservation_price
        r["v_s_min"] = min(st.reservation_price for st in seller_types)
        r["max_welfare"] = max(0.0, buyer_type.reservation_price - r["v_s_min"])
        results.append(r)

        if verbose:
            status = f"DEAL@${result.deal_price:.0f}" if result.deal_reached else f"NO DEAL({result.termination})"
            print(f"    ep{i:3d}: {status} | welfare={result.total_welfare:.1f}")

    return results


# ---------------------------------------------------------------------------
# n:1 평가
# ---------------------------------------------------------------------------

def eval_nv1(
    buyer_factory: Callable,
    seller_factory: Callable,
    config: NegotiationConfig,
    n_episodes: int,
    n_buyers: int = 3,
    base_seed: int = 0,
    verbose: bool = False,
) -> list[dict]:
    env = NVsOneEnv(config, n_buyers=n_buyers)
    results = []

    for i in range(n_episodes):
        seed = base_seed + i
        _, seller_type = sample_types(seed=seed)
        buyer_types = []
        for j in range(n_buyers):
            bt, _ = sample_types(seed=seed * 100 + j + 1)
            buyer_types.append(bt)

        result = env.run(buyer_types, seller_type, buyer_factory, seller_factory)
        r = _multi_result_to_dict(result, i)
        r["v_s"] = seller_type.reservation_price
        r["v_b_max"] = max(bt.reservation_price for bt in buyer_types)
        r["max_welfare"] = max(0.0, r["v_b_max"] - seller_type.reservation_price)
        results.append(r)

        if verbose:
            status = f"DEAL@${result.deal_price:.0f}" if result.deal_reached else f"NO DEAL({result.termination})"
            print(f"    ep{i:3d}: {status} | welfare={result.total_welfare:.1f}")

    return results


def _multi_result_to_dict(result: MultiEpisodeResult, episode_idx: int) -> dict:
    return {
        "setting": result.setting, "n_parties": result.n_parties,
        "episode": episode_idx,
        "deal_reached": result.deal_reached,
        "deal_price": result.deal_price, "deal_round": result.deal_round,
        "winning_party_idx": result.winning_party_idx,
        "termination": result.termination,
        "buyer_reward": result.buyer_reward, "seller_reward": result.seller_reward,
        "total_welfare": result.total_welfare,
    }


# ---------------------------------------------------------------------------
# 통계 계산
# ---------------------------------------------------------------------------

def compute_stats(results: list[dict]) -> dict:
    n = len(results)
    deals = [r for r in results if r["deal_reached"]]

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    def std(xs):
        if len(xs) < 2:
            return 0.0
        m = mean(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

    deal_welfares = [r["total_welfare"] for r in deals]
    all_welfares = [r["total_welfare"] for r in results]
    deal_rounds = [r["deal_round"] for r in deals if r["deal_round"] is not None]
    deal_prices = [r["deal_price"] for r in deals if r["deal_price"] is not None]
    buyer_rewards = [r["buyer_reward"] for r in results]
    seller_rewards = [r["seller_reward"] for r in results]
    max_welfares = [r.get("max_welfare", 0) for r in results]
    efficiency = (
        mean(all_welfares) / mean(max_welfares) if mean(max_welfares) > 0 else 0.0
    )

    return {
        "n_episodes": n,
        "agreement_rate": len(deals) / n if n else 0.0,
        "avg_deal_price": mean(deal_prices),
        "std_deal_price": std(deal_prices),
        "avg_deal_round": mean(deal_rounds),
        "avg_total_welfare_deals": mean(deal_welfares),
        "avg_total_welfare_all": mean(all_welfares),
        "std_total_welfare_all": std(all_welfares),
        "avg_buyer_reward": mean(buyer_rewards),
        "avg_seller_reward": mean(seller_rewards),
        "efficiency": efficiency,
        "termination_counts": {
            t: sum(1 for r in results if r["termination"] == t)
            for t in ["accept", "reject", "max_rounds"]
        },
    }


def cohen_d(a: list[float], b: list[float]) -> float:
    """Cohen's d 효과 크기."""
    if not a or not b:
        return 0.0
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    var_a = sum((x - mean_a) ** 2 for x in a) / max(len(a) - 1, 1)
    var_b = sum((x - mean_b) ** 2 for x in b) / max(len(b) - 1, 1)
    pooled_std = math.sqrt((var_a + var_b) / 2)
    return (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0


# ---------------------------------------------------------------------------
# 보고서 출력
# ---------------------------------------------------------------------------

def print_comparison_table(
    setting: str,
    condition_results: dict[str, dict],
):
    """4개 매칭 조건 비교 테이블 출력."""
    COLS = ["GG", "GB", "BG", "BB"]
    METRICS = [
        ("agreement_rate", "Agreement Rate", ".1%"),
        ("avg_deal_round", "Avg Rounds", ".2f"),
        ("avg_total_welfare_all", "Avg Welfare (all)", ".1f"),
        ("avg_buyer_reward", "Avg Buyer Reward", ".1f"),
        ("avg_seller_reward", "Avg Seller Reward", ".1f"),
        ("efficiency", "Efficiency", ".1%"),
    ]

    print(f"\n{'='*65}")
    print(f"  Setting: {setting}")
    print(f"{'='*65}")
    print(f"{'Metric':<28}" + "".join(f"{c:>9}" for c in COLS))
    print("-" * 65)

    for key, label, fmt in METRICS:
        row = f"{label:<28}"
        for cond in COLS:
            val = condition_results.get(cond, {}).get(key, 0)
            row += f"{val:{fmt}:>9}"
        print(row)

    # GRPO 우위 분석
    print("-" * 65)
    # 구매자 역할: GB vs BB (GRPO buyer vs Base buyer, 판매자 Base 고정)
    gb_wr = condition_results.get("GB", {}).get("avg_buyer_reward", 0)
    bb_wr = condition_results.get("BB", {}).get("avg_buyer_reward", 0)
    # 판매자 역할: BG vs BB
    bg_wr = condition_results.get("BG", {}).get("avg_seller_reward", 0)
    bb_sr = condition_results.get("BB", {}).get("avg_seller_reward", 0)
    print(f"  GRPO Buyer  vs Base Buyer  (seller fixed): Δ buyer reward = {gb_wr - bb_wr:+.2f}")
    print(f"  GRPO Seller vs Base Seller (buyer fixed):  Δ seller reward = {bg_wr - bb_sr:+.2f}")


def save_results(output_dir: Path, setting: str, match_code: str, results: list[dict], stats: dict):
    d = output_dir / setting / match_code
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "episodes.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(d / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="GRPO vs Baseline Evaluation")
    p.add_argument("--base_model_id", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--trained_path", default=None,
                   help="GRPO LoRA 가중치 경로 (train_grpo.py 출력 디렉토리)")
    p.add_argument("--skip_trained", action="store_true",
                   help="학습 모델 없이 BB 조건만 실행")
    p.add_argument("--n_episodes", type=int, default=100)
    p.add_argument("--n_parties", type=int, default=3,
                   help="1:n 또는 n:1에서의 n 값")
    p.add_argument("--max_rounds", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--condition", default="free_form",
                   choices=["free_form", "numeric_only", "templated", "cot_only"])
    p.add_argument("--lambda_s", type=float, default=0.5)
    p.add_argument("--settings", nargs="+", default=["1v1", "1vn", "nv1"],
                   choices=["1v1", "1vn", "nv1"])
    p.add_argument("--output_dir", default="eval_results")
    p.add_argument("--base_seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = NegotiationConfig(
        max_rounds=args.max_rounds,
        condition=Condition(args.condition),
        lambda_selfishness=args.lambda_s,
    )

    # --- 매칭 조건 결정 ---
    if args.skip_trained or args.trained_path is None:
        match_codes = ["BB"]
        print("학습 모델 없음 → BB(baseline vs baseline) 조건만 실행")
    else:
        match_codes = ["GG", "GB", "BG", "BB"]

    # --- 모델 로딩 ---
    print("\n모델 로딩 중...")
    base_model, base_tok = load_base_model(args.base_model_id)
    base_factory = make_local_factory(base_model, base_tok, args.temperature)

    if "GG" in match_codes or "GB" in match_codes or "BG" in match_codes:
        trained_model, trained_tok = load_trained_model(args.base_model_id, args.trained_path)
        trained_factory = make_local_factory(trained_model, trained_tok, args.temperature)
    else:
        trained_factory = base_factory  # dummy

    # match_code → (buyer_factory, seller_factory)
    factories = {
        "GG": (trained_factory, trained_factory),
        "GB": (trained_factory, base_factory),
        "BG": (base_factory, trained_factory),
        "BB": (base_factory, base_factory),
    }

    # 메타 정보 저장
    meta = vars(args)
    meta["match_codes"] = match_codes
    with open(output_dir / "eval_config.json", "w") as f:
        json.dump(meta, f, indent=2)

    all_stats: dict[str, dict[str, dict]] = {}  # setting → match_code → stats

    # --- 세팅별 평가 루프 ---
    for setting in args.settings:
        print(f"\n{'='*60}")
        print(f"  Setting: {setting}  (n_parties={args.n_parties})")
        print(f"{'='*60}")
        all_stats[setting] = {}

        for mc in match_codes:
            buyer_fac, seller_fac = factories[mc]
            t0 = time.time()
            print(f"\n  [{mc}] 실행 중 ({args.n_episodes} 에피소드)...")

            seed = args.base_seed + hash(setting + mc) % 10000

            if setting == "1v1":
                results = eval_1v1(
                    buyer_fac, seller_fac, config,
                    args.n_episodes, seed, args.verbose
                )
            elif setting == "1vn":
                results = eval_1vn(
                    buyer_fac, seller_fac, config,
                    args.n_episodes, args.n_parties, seed, args.verbose
                )
            else:  # nv1
                results = eval_nv1(
                    buyer_fac, seller_fac, config,
                    args.n_episodes, args.n_parties, seed, args.verbose
                )

            stats = compute_stats(results)
            all_stats[setting][mc] = stats
            save_results(output_dir, setting, mc, results, stats)

            elapsed = time.time() - t0
            print(f"    완료: {elapsed:.1f}s | "
                  f"agreement={stats['agreement_rate']:.1%} | "
                  f"welfare={stats['avg_total_welfare_all']:.1f} | "
                  f"efficiency={stats['efficiency']:.1%}")

        # 세팅별 비교 테이블
        print_comparison_table(setting, all_stats[setting])

    # --- 전체 요약 저장 ---
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\n\n전체 요약 저장 → {summary_path}")

    # --- 최종 텍스트 리포트 ---
    report_path = output_dir / "report.txt"
    with open(report_path, "w") as f:
        f.write(f"GRPO Evaluation Report\n")
        f.write(f"Model: {args.base_model_id}\n")
        f.write(f"Trained path: {args.trained_path}\n")
        f.write(f"Episodes/cell: {args.n_episodes}\n")
        f.write(f"n_parties: {args.n_parties}\n")
        f.write(f"Condition: {args.condition}, λ={args.lambda_s}\n\n")

        for setting, cond_stats in all_stats.items():
            f.write(f"\n{'='*60}\n")
            f.write(f"Setting: {setting}\n")
            f.write(f"{'='*60}\n")
            for mc, stats in cond_stats.items():
                f.write(f"\n[{mc}]\n")
                for k, v in stats.items():
                    if k != "termination_counts":
                        f.write(f"  {k}: {v}\n")
                    else:
                        f.write(f"  terminations: {v}\n")

    print(f"상세 리포트 저장 → {report_path}")
    print("\n평가 완료.")


if __name__ == "__main__":
    main()

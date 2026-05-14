"""
에피소드 롤아웃 샘플러.

같은 (buyer_type, seller_type) 설정으로 G개 에피소드를 실행하고,
에피소드별 보상을 그룹 내에서 정규화하여 GRPO 이점(advantage)을 계산한다.

보상 함수: r = δ^t · [λ · own_surplus + (1-λ) · total_welfare]
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..agents import (
    _BUYER_SYSTEM,
    _SELLER_SYSTEM,
    _CONDITION_INSTRUCTIONS,
    _extract_action,
)
from ..config import NegotiationConfig
from ..types import BuyerType, SellerType
from .local_agent import LocalNegotiationAgent, TurnRecord


@dataclass
class EpisodeRollout:
    """단일 에피소드의 전체 트래젝토리 및 결과."""
    buyer_turns: list[TurnRecord]
    seller_turns: list[TurnRecord]
    buyer_reward: float
    seller_reward: float
    deal_reached: bool
    deal_price: Optional[float]
    termination: str            # 'accept' | 'reject' | 'max_rounds'


@dataclass
class GroupRollout:
    """같은 설정으로 실행된 G개 에피소드 + 정규화된 이점."""
    episodes: list[EpisodeRollout]
    buyer_advantages: list[float]   # 에피소드별 정규화 이점 (구매자 관점)
    seller_advantages: list[float]  # 에피소드별 정규화 이점 (판매자 관점)


# ---------------------------------------------------------------------------
# 에이전트 팩토리
# ---------------------------------------------------------------------------

def _make_local_buyer(
    buyer_type: BuyerType,
    config: NegotiationConfig,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    temperature: float,
) -> LocalNegotiationAgent:
    delta = config.delta_from_level(buyer_type.urgency)
    system = _BUYER_SYSTEM.format(
        item_description=config.item_description,
        reservation_price=buyer_type.reservation_price,
        urgency_label=buyer_type.urgency_label(),
        urgency=buyer_type.urgency,
        delta=delta,
        outside_option=buyer_type.outside_option,
        max_rounds=config.max_rounds,
        condition_instructions=_CONDITION_INSTRUCTIONS[config.condition],
    )
    return LocalNegotiationAgent("buyer", system, model, tokenizer, config, temperature)


def _make_local_seller(
    seller_type: SellerType,
    config: NegotiationConfig,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    temperature: float,
) -> LocalNegotiationAgent:
    delta = config.delta_from_level(seller_type.inventory_pressure)
    system = _SELLER_SYSTEM.format(
        item_description=config.item_description,
        reservation_price=seller_type.reservation_price,
        pressure_label=seller_type.pressure_label(),
        pressure=seller_type.inventory_pressure,
        delta=delta,
        max_rounds=config.max_rounds,
        condition_instructions=_CONDITION_INSTRUCTIONS[config.condition],
    )
    return LocalNegotiationAgent("seller", system, model, tokenizer, config, temperature)


# ---------------------------------------------------------------------------
# 단일 에피소드 롤아웃
# ---------------------------------------------------------------------------

def rollout_episode(
    buyer_type: BuyerType,
    seller_type: SellerType,
    config: NegotiationConfig,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    temperature: float = 0.8,
) -> EpisodeRollout:
    """단일 에피소드를 실행하고 트래젝토리와 보상을 반환한다."""
    buyer_delta = config.delta_from_level(buyer_type.urgency)
    seller_delta = config.delta_from_level(seller_type.inventory_pressure)
    lam = config.lambda_selfishness

    buyer = _make_local_buyer(buyer_type, config, model, tokenizer, temperature)
    seller = _make_local_seller(seller_type, config, model, tokenizer, temperature)

    current_price: Optional[float] = None
    rnd = 0

    def compute_rewards(price: Optional[float], rnd: int):
        if price is not None and price > 0:
            bs = max(0.0, buyer_type.reservation_price - price) * (buyer_delta ** rnd)
            ss = max(0.0, price - seller_type.reservation_price) * (seller_delta ** rnd)
            tw = bs + ss
            br = lam * bs + (1 - lam) * tw
            sr = lam * ss + (1 - lam) * tw
        else:
            br = sr = 0.0
        return br, sr

    def make_result(price, rnd, termination) -> EpisodeRollout:
        br, sr = compute_rewards(price, rnd)
        return EpisodeRollout(
            buyer_turns=buyer.trajectory,
            seller_turns=seller.trajectory,
            buyer_reward=br,
            seller_reward=sr,
            deal_reached=price is not None,
            deal_price=price,
            termination=termination,
        )

    # --- 구매자 오프닝 ---
    resp = buyer.respond(None, round_num=rnd)
    if resp["action"] == "reject":
        return make_result(None, rnd, "reject")
    if resp["price"] is not None:
        current_price = resp["price"]

    incoming_to_seller = resp["visible_message"]
    max_turns = config.max_rounds * 2

    turn_index = 1
    while turn_index < max_turns:
        # 판매자 턴
        resp = seller.respond(incoming_to_seller, round_num=rnd)
        turn_index += 1

        if resp["action"] == "accept":
            return make_result(current_price, rnd, "accept")
        if resp["action"] == "reject":
            return make_result(None, rnd, "reject")
        if resp["price"] is not None:
            current_price = resp["price"]

        incoming_to_buyer = resp["visible_message"]
        rnd += 1

        if rnd >= config.max_rounds:
            return make_result(None, rnd, "max_rounds")

        # 구매자 턴
        resp = buyer.respond(incoming_to_buyer, round_num=rnd)
        turn_index += 1

        if resp["action"] == "accept":
            return make_result(current_price, rnd, "accept")
        if resp["action"] == "reject":
            return make_result(None, rnd, "reject")
        if resp["price"] is not None:
            current_price = resp["price"]

        incoming_to_seller = resp["visible_message"]

    return make_result(None, rnd, "max_rounds")


# ---------------------------------------------------------------------------
# 그룹 롤아웃 (GRPO 단위)
# ---------------------------------------------------------------------------

def rollout_group(
    buyer_type: BuyerType,
    seller_type: SellerType,
    config: NegotiationConfig,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    group_size: int = 8,
    temperature: float = 0.8,
) -> GroupRollout:
    """
    같은 (buyer_type, seller_type)으로 group_size개 에피소드를 실행하고
    그룹 내 정규화된 이점을 계산한다.
    """
    episodes = [
        rollout_episode(buyer_type, seller_type, config, model, tokenizer, temperature)
        for _ in range(group_size)
    ]

    buyer_rewards = [ep.buyer_reward for ep in episodes]
    seller_rewards = [ep.seller_reward for ep in episodes]

    return GroupRollout(
        episodes=episodes,
        buyer_advantages=_normalize(buyer_rewards),
        seller_advantages=_normalize(seller_rewards),
    )


def _normalize(rewards: list[float]) -> list[float]:
    """그룹 내 보상을 정규화: Â = (r - μ) / σ. 분산이 0이면 전부 0."""
    if len(set(rewards)) == 1:
        return [0.0] * len(rewards)
    mu = statistics.mean(rewards)
    sigma = statistics.stdev(rewards) + 1e-8
    return [(r - mu) / sigma for r in rewards]

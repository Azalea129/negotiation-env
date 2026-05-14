"""
다자 협상 환경 (Multi-party Negotiation).

1:n (구매자 1명 vs 판매자 n명):
  - 판매자들이 경쟁. 구매자가 최선 조건을 선택.
  - 매 라운드: 구매자가 제안 → 모든 판매자 응답 → 구매자가 최선 응답 처리.
  - 어느 판매자가 ACCEPT하면 거래 성사 (구매자 마지막 제안 가격으로).
  - 구매자가 ACCEPT하면 현재 최저 판매자 역제안 가격으로 거래 성사.

n:1 (구매자 n명 vs 판매자 1명):
  - 구매자들이 경쟁. 판매자가 최선 조건을 선택.
  - 매 라운드: 모든 구매자 제안 → 판매자가 응답 → 최선 구매자 처리.
  - 어느 구매자가 ACCEPT하면 거래 성사 (판매자 마지막 제안 가격으로).
  - 판매자가 ACCEPT하면 현재 최고 구매자 제안 가격으로 거래 성사.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .agents import _BUYER_SYSTEM, _SELLER_SYSTEM, _CONDITION_INSTRUCTIONS, _extract_action
from .config import NegotiationConfig
from .types import BuyerType, SellerType

# 에이전트 팩토리 타입: (role, system_prompt, config) → respond 메서드를 가진 객체
AgentFactory = Callable[[str, str, NegotiationConfig], object]


# ---------------------------------------------------------------------------
# 다자 협상 시스템 프롬프트
# ---------------------------------------------------------------------------

_BUYER_SYSTEM_1VN = """\
You are a buyer negotiating to purchase: {item_description}

You are negotiating SIMULTANEOUSLY with {n_sellers} competing sellers.
Use this competition to your advantage.

YOUR PRIVATE INFORMATION (do NOT reveal exact numbers):
- Your reservation price (maximum you will pay): ${reservation_price:.0f}
- Your urgency level: {urgency_label} (urgency={urgency:.2f})
  → Each round of delay multiplies your payoff by {delta:.3f}.
- Your outside option: ${outside_option:.0f}

GOAL: Close a deal at the lowest possible price. If a seller offers a great price,
accept it. Use competition between sellers to drive prices down.

RULES:
- Each round you send ONE message (seen by ALL sellers).
- Maximum {max_rounds} rounds total.
- You will receive a summary of all sellers' responses each round.

{condition_instructions}

ACTION TAG FORMAT (required in every response):
  [ACCEPT]        — accept the best available seller counter-offer
  [REJECT]        — walk away from all negotiations (no deal)
  [OFFER: $X]     — propose price $X to all sellers
"""

_SELLER_SYSTEM_1VN = """\
You are a seller negotiating to sell: {item_description}

IMPORTANT: You are competing with {n_other_sellers} other sellers for this single buyer.
If you price too high, the buyer will choose a competitor.

YOUR PRIVATE INFORMATION (do NOT reveal exact numbers):
- Your reservation price (minimum you will accept): ${reservation_price:.0f}
- Your inventory pressure: {pressure_label} (pressure={pressure:.2f})
  → Each round of delay multiplies your payoff by {delta:.3f}.

GOAL: Close a deal above your reservation price. Be competitive enough to win.

RULES:
- Maximum {max_rounds} rounds total.
- You receive the buyer's offer each round and respond independently.

{condition_instructions}

ACTION TAG FORMAT (required in every response):
  [ACCEPT]        — accept the buyer's last proposed price
  [REJECT]        — withdraw from this negotiation
  [OFFER: $X]     — propose a new price of $X
"""

_SELLER_SYSTEM_NV1 = """\
You are a seller negotiating to sell: {item_description}

You are negotiating SIMULTANEOUSLY with {n_buyers} competing buyers.
You have ONE item to sell — only one deal can be made.

YOUR PRIVATE INFORMATION (do NOT reveal exact numbers):
- Your reservation price (minimum you will accept): ${reservation_price:.0f}
- Your inventory pressure: {pressure_label} (pressure={pressure:.2f})
  → Each round of delay multiplies your payoff by {delta:.3f}.

GOAL: Close a deal at the highest possible price. If a buyer offers well, accept.
Use competition among buyers to drive prices up.

RULES:
- Each round you send ONE message (seen by ALL buyers).
- Maximum {max_rounds} rounds total.
- You will receive a summary of all buyers' offers each round.

{condition_instructions}

ACTION TAG FORMAT (required in every response):
  [ACCEPT]        — accept the best available buyer offer
  [REJECT]        — end all negotiations (no deal)
  [OFFER: $X]     — counter-propose price $X to all buyers
"""

_BUYER_SYSTEM_NV1 = """\
You are a buyer negotiating to purchase: {item_description}

IMPORTANT: You are competing with {n_other_buyers} other buyers for this single seller.
If you offer too low, the seller will choose a competitor.

YOUR PRIVATE INFORMATION (do NOT reveal exact numbers):
- Your reservation price (maximum you will pay): ${reservation_price:.0f}
- Your urgency level: {urgency_label} (urgency={urgency:.2f})
  → Each round of delay multiplies your payoff by {delta:.3f}.
- Your outside option: ${outside_option:.0f}

GOAL: Secure the item before another buyer does. Be competitive but don't overpay.

RULES:
- Maximum {max_rounds} rounds total.
- You receive the seller's counter-offer each round and respond independently.

{condition_instructions}

ACTION TAG FORMAT (required in every response):
  [ACCEPT]        — accept the seller's last proposed price
  [REJECT]        — withdraw from this negotiation
  [OFFER: $X]     — propose a new price of $X
"""


# ---------------------------------------------------------------------------
# 결과 데이터 클래스
# ---------------------------------------------------------------------------

@dataclass
class MultiEpisodeResult:
    setting: str                      # '1:1', '1:n', 'n:1'
    n_parties: int                    # n in 1:n or n:1 (1 for 1:1)
    deal_reached: bool
    deal_price: Optional[float]
    deal_round: Optional[int]
    winning_party_idx: Optional[int]  # 거래 성사한 판매자(1:n) 또는 구매자(n:1) 인덱스
    termination: str                  # 'accept' | 'reject' | 'max_rounds'
    buyer_reward: float
    seller_reward: float
    total_welfare: float
    condition: str
    lambda_selfishness: float


# ---------------------------------------------------------------------------
# 1:n 환경 (구매자 1명 vs 판매자 n명)
# ---------------------------------------------------------------------------

class OneVsNEnv:
    """1 buyer agent vs n seller agents."""

    def __init__(self, config: NegotiationConfig, n_sellers: int = 3):
        self.config = config
        self.n_sellers = n_sellers

    def run(
        self,
        buyer_type: BuyerType,
        seller_types: list[SellerType],
        buyer_factory: AgentFactory,
        seller_factory: AgentFactory,
    ) -> MultiEpisodeResult:
        assert len(seller_types) == self.n_sellers

        cfg = self.config
        buyer_delta = cfg.delta_from_level(buyer_type.urgency)
        lam = cfg.lambda_selfishness

        # --- 에이전트 생성 ---
        buyer_system = _BUYER_SYSTEM_1VN.format(
            item_description=cfg.item_description,
            n_sellers=self.n_sellers,
            reservation_price=buyer_type.reservation_price,
            urgency_label=buyer_type.urgency_label(),
            urgency=buyer_type.urgency,
            delta=buyer_delta,
            outside_option=buyer_type.outside_option,
            max_rounds=cfg.max_rounds,
            condition_instructions=_CONDITION_INSTRUCTIONS[cfg.condition],
        )
        buyer = buyer_factory("buyer", buyer_system, cfg)

        sellers = []
        seller_deltas = []
        for i, st in enumerate(seller_types):
            sd = cfg.delta_from_level(st.inventory_pressure)
            seller_deltas.append(sd)
            sys = _SELLER_SYSTEM_1VN.format(
                item_description=cfg.item_description,
                n_other_sellers=self.n_sellers - 1,
                reservation_price=st.reservation_price,
                pressure_label=st.pressure_label(),
                pressure=st.inventory_pressure,
                delta=sd,
                max_rounds=cfg.max_rounds,
                condition_instructions=_CONDITION_INSTRUCTIONS[cfg.condition],
            )
            sellers.append(seller_factory("seller", sys, cfg))

        # 상태
        buyer_offer: Optional[float] = None          # 구매자 마지막 제안
        seller_offers: list[Optional[float]] = [None] * self.n_sellers  # 각 판매자 역제안
        active_sellers: list[bool] = [True] * self.n_sellers  # 협상 중인 판매자
        rnd = 0

        def compute_result(price, seller_idx, rnd, term):
            if price is not None:
                # 거래 성사: 해당 판매자의 잉여 계산
                st = seller_types[seller_idx]
                sd = seller_deltas[seller_idx]
                bs = max(0.0, buyer_type.reservation_price - price) * (buyer_delta ** rnd)
                ss = max(0.0, price - st.reservation_price) * (sd ** rnd)
                tw = bs + ss
                br = lam * bs + (1 - lam) * tw
                sr = lam * ss + (1 - lam) * tw
            else:
                seller_idx = None
                bs = ss = tw = br = sr = 0.0
            return MultiEpisodeResult(
                setting="1:n", n_parties=self.n_sellers,
                deal_reached=price is not None,
                deal_price=price, deal_round=rnd if price is not None else None,
                winning_party_idx=seller_idx, termination=term,
                buyer_reward=br, seller_reward=sr, total_welfare=tw,
                condition=cfg.condition.value, lambda_selfishness=lam,
            )

        # --- 구매자 오프닝 ---
        resp = buyer.respond(None)
        if resp["action"] == "reject":
            return compute_result(None, None, rnd, "reject")
        if resp["price"] is not None:
            buyer_offer = resp["price"]
        buyer_msg = resp["visible_message"]

        # --- 메인 루프 ---
        for _ in range(cfg.max_rounds):
            # 활성 판매자들이 응답
            seller_resps = []
            for i, seller in enumerate(sellers):
                if not active_sellers[i]:
                    seller_resps.append(None)
                    continue
                r = seller.respond(buyer_msg)
                seller_resps.append(r)
                if r["action"] == "accept" and buyer_offer is not None:
                    # 판매자가 구매자 제안 수락 → 거래
                    return compute_result(buyer_offer, i, rnd, "accept")
                if r["action"] == "reject":
                    active_sellers[i] = False
                if r["price"] is not None:
                    seller_offers[i] = r["price"]

            # 모든 판매자가 거절했으면 종료
            if not any(active_sellers):
                return compute_result(None, None, rnd, "reject")

            rnd += 1
            if rnd >= cfg.max_rounds:
                return compute_result(None, None, rnd, "max_rounds")

            # 구매자에게 판매자 응답 요약 전달
            summary = self._format_seller_responses(
                seller_resps, active_sellers, buyer_offer, rnd
            )
            resp = buyer.respond(summary)

            if resp["action"] == "reject":
                return compute_result(None, None, rnd, "reject")

            if resp["action"] == "accept":
                # 구매자가 최선 판매자 역제안 수락
                best_price, best_idx = self._best_seller_offer(seller_offers, active_sellers)
                if best_price is not None:
                    return compute_result(best_price, best_idx, rnd, "accept")
                # 역제안 없으면 수락 불가 → 계속
            else:
                if resp["price"] is not None:
                    buyer_offer = resp["price"]
                buyer_msg = resp["visible_message"]

        return compute_result(None, None, rnd, "max_rounds")

    def _format_seller_responses(
        self,
        resps: list,
        active: list[bool],
        buyer_offer: Optional[float],
        rnd: int,
    ) -> str:
        lines = [f"Round {rnd} — {self.n_sellers} seller responses:"]
        for i, (r, act) in enumerate(zip(resps, active)):
            if r is None or not act:
                lines.append(f"[Seller {i+1}] withdrew from negotiations.")
                continue
            tag = f"action={r['action']}" + (f", price=${r['price']:.0f}" if r["price"] else "")
            lines.append(f"[Seller {i+1}] ({tag}): {r['visible_message']}")

        best_price, best_idx = self._best_seller_offer(
            [r["price"] if r and r["price"] else None for r in resps], active
        )
        if best_price is not None:
            lines.append(f"\nBest counter-offer: Seller {best_idx+1} at ${best_price:.0f}.")
        if buyer_offer is not None:
            lines.append(f"Your last offer was ${buyer_offer:.0f}.")
        lines.append("Respond with [ACCEPT], [OFFER: $X], or [REJECT].")
        return "\n".join(lines)

    @staticmethod
    def _best_seller_offer(
        offers: list[Optional[float]], active: list[bool]
    ) -> tuple[Optional[float], Optional[int]]:
        best_price, best_idx = None, None
        for i, (p, act) in enumerate(zip(offers, active)):
            if act and p is not None:
                if best_price is None or p < best_price:
                    best_price, best_idx = p, i
        return best_price, best_idx


# ---------------------------------------------------------------------------
# n:1 환경 (구매자 n명 vs 판매자 1명)
# ---------------------------------------------------------------------------

class NVsOneEnv:
    """n buyer agents vs 1 seller agent."""

    def __init__(self, config: NegotiationConfig, n_buyers: int = 3):
        self.config = config
        self.n_buyers = n_buyers

    def run(
        self,
        buyer_types: list[BuyerType],
        seller_type: SellerType,
        buyer_factory: AgentFactory,
        seller_factory: AgentFactory,
    ) -> MultiEpisodeResult:
        assert len(buyer_types) == self.n_buyers

        cfg = self.config
        seller_delta = cfg.delta_from_level(seller_type.inventory_pressure)
        lam = cfg.lambda_selfishness

        # --- 에이전트 생성 ---
        seller_system = _SELLER_SYSTEM_NV1.format(
            item_description=cfg.item_description,
            n_buyers=self.n_buyers,
            reservation_price=seller_type.reservation_price,
            pressure_label=seller_type.pressure_label(),
            pressure=seller_type.inventory_pressure,
            delta=seller_delta,
            max_rounds=cfg.max_rounds,
            condition_instructions=_CONDITION_INSTRUCTIONS[cfg.condition],
        )
        seller = seller_factory("seller", seller_system, cfg)

        buyers = []
        buyer_deltas = []
        for i, bt in enumerate(buyer_types):
            bd = cfg.delta_from_level(bt.urgency)
            buyer_deltas.append(bd)
            sys = _BUYER_SYSTEM_NV1.format(
                item_description=cfg.item_description,
                n_other_buyers=self.n_buyers - 1,
                reservation_price=bt.reservation_price,
                urgency_label=bt.urgency_label(),
                urgency=bt.urgency,
                delta=bd,
                outside_option=bt.outside_option,
                max_rounds=cfg.max_rounds,
                condition_instructions=_CONDITION_INSTRUCTIONS[cfg.condition],
            )
            buyers.append(buyer_factory("buyer", sys, cfg))

        # 상태
        seller_offer: Optional[float] = None
        buyer_offers: list[Optional[float]] = [None] * self.n_buyers
        active_buyers: list[bool] = [True] * self.n_buyers
        rnd = 0

        def compute_result(price, buyer_idx, rnd, term):
            if price is not None:
                bt = buyer_types[buyer_idx]
                bd = buyer_deltas[buyer_idx]
                bs = max(0.0, bt.reservation_price - price) * (bd ** rnd)
                ss = max(0.0, price - seller_type.reservation_price) * (seller_delta ** rnd)
                tw = bs + ss
                br = lam * bs + (1 - lam) * tw
                sr = lam * ss + (1 - lam) * tw
            else:
                buyer_idx = None
                bs = ss = tw = br = sr = 0.0
            return MultiEpisodeResult(
                setting="n:1", n_parties=self.n_buyers,
                deal_reached=price is not None,
                deal_price=price, deal_round=rnd if price is not None else None,
                winning_party_idx=buyer_idx, termination=term,
                buyer_reward=br, seller_reward=sr, total_welfare=tw,
                condition=cfg.condition.value, lambda_selfishness=lam,
            )

        # --- 모든 구매자 오프닝 ---
        buyer_msgs = []
        for i, buyer in enumerate(buyers):
            r = buyer.respond(None)
            if r["action"] == "reject":
                active_buyers[i] = False
                buyer_msgs.append(None)
            else:
                if r["price"] is not None:
                    buyer_offers[i] = r["price"]
                buyer_msgs.append(r["visible_message"])

        if not any(active_buyers):
            return compute_result(None, None, rnd, "reject")

        # --- 메인 루프 ---
        for _ in range(cfg.max_rounds):
            # 판매자가 모든 구매자 오퍼 요약 수신
            summary = self._format_buyer_offers(buyer_msgs, buyer_offers, active_buyers, seller_offer, rnd)
            resp = seller.respond(summary)

            if resp["action"] == "accept":
                best_price, best_idx = self._best_buyer_offer(buyer_offers, active_buyers)
                if best_price is not None:
                    return compute_result(best_price, best_idx, rnd, "accept")
            if resp["action"] == "reject":
                return compute_result(None, None, rnd, "reject")
            if resp["price"] is not None:
                seller_offer = resp["price"]
            seller_msg = resp["visible_message"]

            rnd += 1
            if rnd >= cfg.max_rounds:
                return compute_result(None, None, rnd, "max_rounds")

            # 각 구매자가 판매자 역제안에 응답
            buyer_msgs = []
            for i, buyer in enumerate(buyers):
                if not active_buyers[i]:
                    buyer_msgs.append(None)
                    continue
                r = buyer.respond(seller_msg)
                if r["action"] == "accept" and seller_offer is not None:
                    return compute_result(seller_offer, i, rnd, "accept")
                if r["action"] == "reject":
                    active_buyers[i] = False
                    buyer_msgs.append(None)
                else:
                    if r["price"] is not None:
                        buyer_offers[i] = r["price"]
                    buyer_msgs.append(r["visible_message"])

            if not any(active_buyers):
                return compute_result(None, None, rnd, "reject")

        return compute_result(None, None, rnd, "max_rounds")

    def _format_buyer_offers(
        self,
        msgs: list,
        offers: list[Optional[float]],
        active: list[bool],
        seller_offer: Optional[float],
        rnd: int,
    ) -> str:
        lines = [f"Round {rnd} — {self.n_buyers} buyer offers:"]
        for i, (msg, act) in enumerate(zip(msgs, active)):
            if not act or msg is None:
                lines.append(f"[Buyer {i+1}] withdrew from negotiations.")
                continue
            price_str = f", price=${offers[i]:.0f}" if offers[i] else ""
            lines.append(f"[Buyer {i+1}]{price_str}: {msg}")

        best_price, best_idx = self._best_buyer_offer(offers, active)
        if best_price is not None:
            lines.append(f"\nBest offer: Buyer {best_idx+1} at ${best_price:.0f}.")
        if seller_offer is not None:
            lines.append(f"Your last counter was ${seller_offer:.0f}.")
        lines.append("Respond with [ACCEPT], [OFFER: $X], or [REJECT].")
        return "\n".join(lines)

    @staticmethod
    def _best_buyer_offer(
        offers: list[Optional[float]], active: list[bool]
    ) -> tuple[Optional[float], Optional[int]]:
        best_price, best_idx = None, None
        for i, (p, act) in enumerate(zip(offers, active)):
            if act and p is not None:
                if best_price is None or p > best_price:
                    best_price, best_idx = p, i
        return best_price, best_idx

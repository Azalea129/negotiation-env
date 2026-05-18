"""
A2ANegotiationEnv: single A2A negotiation episode.

Protocol:
  - Buyer opens (turn 0, round 0).
  - Turns alternate: buyer → seller → buyer → ...
  - Round increments after each full buyer+seller exchange.
  - Deal closes on [ACCEPT]; episode ends on [ACCEPT], [REJECT], or max_rounds.

At each turn, the environment:
  1. Builds c_t (conversation context up to this turn)
  2. Calls the agent → gets raw_cot, belief_text, intention_text, visible_message
  3. Records TurnData for training data collection
  4. Forwards only visible_message to the counterparty

Reward design (GRPO Stage 2 target):
  R = λ × surplus_captured + (1-λ) × total_welfare + deal_bonus
"""

from typing import Optional

from .agents import NegotiationAgent, make_buyer_agent, make_seller_agent
from .config import LASHConfig
from .types import BuyerType, EpisodeData, SellerType, TurnData, new_episode_id, sample_types


class A2ANegotiationEnv:
    def __init__(self, config: LASHConfig):
        self.config = config

    def run(
        self,
        buyer_type: Optional[BuyerType] = None,
        seller_type: Optional[SellerType] = None,
        seed: Optional[int] = None,
    ) -> EpisodeData:
        if buyer_type is None or seller_type is None:
            buyer_type, seller_type = sample_types(seed=seed)

        buyer = make_buyer_agent(buyer_type, self.config)
        seller = make_seller_agent(seller_type, self.config)

        turns: list[TurnData] = []
        context_lines: list[str] = []   # running conversation log for c_t
        current_price: Optional[float] = None
        turn_index = 0
        rnd = 0

        def build_context() -> str:
            return "\n".join(context_lines)

        def record_turn(role: str, resp: dict, rnd: int) -> TurnData:
            td = TurnData(
                turn=turn_index,
                round=rnd,
                role=role,
                context=build_context(),
                raw_cot=resp["raw_cot"],
                belief_text=resp["belief_text"],
                intention_text=resp["intention_text"],
                visible_message=resp["visible_message"],
                action=resp["action"],
                price=resp["price"],
            )
            turns.append(td)
            # Append only the visible message to running context
            context_lines.append(f"[{role.upper()} turn {turn_index}]: {resp['visible_message']}")
            return td

        def compute_result(price: Optional[float], rnd: int, termination: str) -> EpisodeData:
            if price is not None:
                bs = max(0.0, buyer_type.reservation_price - price) * (buyer_type.delta ** rnd)
                ss = max(0.0, price - seller_type.reservation_price) * (seller_type.delta ** rnd)
                tw = bs + ss
                lam = self.config.lambda_selfishness
                br = lam * bs + (1 - lam) * tw + self.config.deal_bonus
                sr = lam * ss + (1 - lam) * tw + self.config.deal_bonus
            else:
                bs = ss = tw = br = sr = 0.0

            return EpisodeData(
                episode_id=new_episode_id(),
                buyer_type=buyer_type,
                seller_type=seller_type,
                deal_reached=price is not None,
                deal_price=price,
                deal_round=rnd if price is not None else None,
                termination=termination,
                buyer_surplus=bs,
                seller_surplus=ss,
                total_welfare=tw,
                buyer_reward=br,
                seller_reward=sr,
                turns=turns,
            )

        # ── Buyer opens ──────────────────────────────────────────────
        resp = buyer.respond(None)
        record_turn("buyer", resp, rnd)
        turn_index += 1

        if resp["action"] == "reject":
            return compute_result(None, rnd, "reject")
        if resp["price"] is not None:
            current_price = resp["price"]

        incoming_to_seller = resp["visible_message"]

        # ── Main alternating loop ────────────────────────────────────
        max_turns = self.config.max_rounds * 2

        while turn_index < max_turns:
            # Seller turn
            resp = seller.respond(incoming_to_seller)
            record_turn("seller", resp, rnd)
            turn_index += 1

            if resp["action"] == "accept":
                return compute_result(current_price, rnd, "accept")
            if resp["action"] == "reject":
                return compute_result(None, rnd, "reject")
            if resp["price"] is not None:
                current_price = resp["price"]

            incoming_to_buyer = resp["visible_message"]
            rnd += 1

            if rnd >= self.config.max_rounds:
                return compute_result(None, rnd, "max_rounds")

            # Buyer turn
            resp = buyer.respond(incoming_to_buyer)
            record_turn("buyer", resp, rnd)
            turn_index += 1

            if resp["action"] == "accept":
                return compute_result(current_price, rnd, "accept")
            if resp["action"] == "reject":
                return compute_result(None, rnd, "reject")
            if resp["price"] is not None:
                current_price = resp["price"]

            incoming_to_seller = resp["visible_message"]

        return compute_result(None, rnd, "max_rounds")

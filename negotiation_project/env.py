"""
NegotiationEnv: manages a single negotiation episode.

Protocol:
  - Buyer makes the opening offer (turn 0).
  - Turns alternate: buyer → seller → buyer → ...
  - Round t increments after each full buyer+seller exchange.
  - δ^t is applied at the round the deal is closed.
  - Deal closed when either party sends [ACCEPT].
  - Episode ends on [ACCEPT], [REJECT], or max_rounds exhausted.
"""

from dataclasses import dataclass, field
from typing import Optional

from .agents import NegotiationAgent, make_buyer_agent, make_seller_agent
from .config import NegotiationConfig
from .types import BuyerType, SellerType


@dataclass
class TurnLog:
    turn: int
    round: int
    speaker: str           # 'buyer' or 'seller'
    raw_response: str
    visible_message: str   # what the counterparty sees
    action: str
    price: Optional[float]


@dataclass
class EpisodeResult:
    # Outcome
    deal_reached: bool
    deal_price: Optional[float]
    deal_round: Optional[int]    # 0-indexed round at which deal closed
    termination: str             # 'accept' | 'reject' | 'max_rounds'

    # Payoffs (0 if no deal)
    buyer_surplus: float
    seller_surplus: float
    total_welfare: float
    buyer_reward: float          # δ^t weighted, λ-shaped
    seller_reward: float

    # Private types (for analysis)
    buyer_type: BuyerType
    seller_type: SellerType
    buyer_delta: float
    seller_delta: float

    # Full transcript
    turns: list[TurnLog] = field(default_factory=list)

    # Config snapshot
    condition: str = ""
    lambda_selfishness: float = 0.5
    rho_lie_penalty: float = 0.0


class NegotiationEnv:
    def __init__(self, config: NegotiationConfig):
        self.config = config

    def run(
        self,
        buyer_type: Optional[BuyerType] = None,
        seller_type: Optional[SellerType] = None,
        seed: Optional[int] = None,
    ) -> EpisodeResult:
        from .types import sample_types

        if buyer_type is None or seller_type is None:
            buyer_type, seller_type = sample_types(seed=seed)

        buyer_delta = self.config.delta_from_level(buyer_type.urgency)
        seller_delta = self.config.delta_from_level(seller_type.inventory_pressure)

        buyer = make_buyer_agent(buyer_type, self.config)
        seller = make_seller_agent(seller_type, self.config)

        turns: list[TurnLog] = []
        current_price: Optional[float] = None
        turn_index = 0

        def log_turn(speaker, resp, rnd) -> TurnLog:
            t = TurnLog(
                turn=turn_index,
                round=rnd,
                speaker=speaker,
                raw_response=resp["raw_response"],
                visible_message=resp["visible_message"],
                action=resp["action"],
                price=resp["price"],
            )
            turns.append(t)
            return t

        def compute_result(price, rnd, termination) -> EpisodeResult:
            if price is not None and price > 0:
                bs = max(0.0, buyer_type.reservation_price - price) * (buyer_delta ** rnd)
                ss = max(0.0, price - seller_type.reservation_price) * (seller_delta ** rnd)
                tw = bs + ss
                lam = self.config.lambda_selfishness
                br = lam * bs + (1 - lam) * tw
                sr = lam * ss + (1 - lam) * tw
            else:
                bs = ss = tw = br = sr = 0.0

            return EpisodeResult(
                deal_reached=price is not None,
                deal_price=price,
                deal_round=rnd if price is not None else None,
                termination=termination,
                buyer_surplus=bs,
                seller_surplus=ss,
                total_welfare=tw,
                buyer_reward=br,
                seller_reward=sr,
                buyer_type=buyer_type,
                seller_type=seller_type,
                buyer_delta=buyer_delta,
                seller_delta=seller_delta,
                turns=turns,
                condition=self.config.condition.value,
                lambda_selfishness=self.config.lambda_selfishness,
                rho_lie_penalty=self.config.rho_lie_penalty,
            )

        # --- Buyer opens ---
        rnd = 0
        resp = buyer.respond(None)
        t = log_turn("buyer", resp, rnd)
        turn_index += 1

        if resp["action"] == "reject":
            return compute_result(None, rnd, "reject")
        if resp["price"] is not None:
            current_price = resp["price"]

        incoming_to_seller = resp["visible_message"]

        # --- Main loop ---
        max_turns = self.config.max_rounds * 2  # each round = 1 buyer turn + 1 seller turn

        while turn_index < max_turns:
            # Seller's turn
            resp = seller.respond(incoming_to_seller)
            t = log_turn("seller", resp, rnd)
            turn_index += 1

            if resp["action"] == "accept":
                return compute_result(current_price, rnd, "accept")
            if resp["action"] == "reject":
                return compute_result(None, rnd, "reject")
            if resp["price"] is not None:
                current_price = resp["price"]

            incoming_to_buyer = resp["visible_message"]

            # Advance round after seller speaks
            rnd += 1

            if rnd >= self.config.max_rounds:
                return compute_result(None, rnd, "max_rounds")

            # Buyer's turn
            resp = buyer.respond(incoming_to_buyer)
            t = log_turn("buyer", resp, rnd)
            turn_index += 1

            if resp["action"] == "accept":
                return compute_result(current_price, rnd, "accept")
            if resp["action"] == "reject":
                return compute_result(None, rnd, "reject")
            if resp["price"] is not None:
                current_price = resp["price"]

            incoming_to_seller = resp["visible_message"]

        return compute_result(None, rnd, "max_rounds")

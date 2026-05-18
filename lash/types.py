"""
Core data types for LASH negotiation framework.

Each turn captures the full BDI cognitive trace:
  c_t  → context (conversation history)
  B_t^GT → belief_text (opponent type/state estimate)
  I_t^GT → intention_text (own strategic intention)
  M_t  → visible_message (what counterparty sees)

These (context, belief_gt, intention_gt, message) tuples are the
training signal for Stage 1 supervised learning.
"""

import random
import uuid
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class BuyerType:
    reservation_price: float  # v_b: max willingness to pay
    delta: float              # δ_b ∈ (0,1]: patience (lower = more urgent)


@dataclass
class SellerType:
    reservation_price: float  # v_s: minimum acceptable price
    delta: float              # δ_s ∈ (0,1]: patience (lower = more inventory pressure)


@dataclass
class TurnData:
    """
    Single negotiation turn with structured CoT.
    Serves as one training data point for Stage 1 supervised learning.
    """
    turn: int
    round: int
    role: str                  # 'buyer' | 'seller'

    # Training signal: c_t, B_t^GT, I_t^GT, M_t
    context: str               # full conversation history up to (not including) this turn
    raw_cot: str               # full model output with <belief>/<intention> tags
    belief_text: str           # B_t^GT: extracted from <belief>...</belief>
    intention_text: str        # I_t^GT: extracted from <intention>...</intention>
    visible_message: str       # M_t: what the counterparty sees

    # Action outcome
    action: str                # 'accept' | 'reject' | 'offer' | 'unknown'
    price: Optional[float]


@dataclass
class EpisodeData:
    """Complete episode with all turn data. buyer/seller_type is None for non-negotiation games."""
    episode_id: str
    buyer_type: Optional[BuyerType]
    seller_type: Optional[SellerType]

    # Outcome
    deal_reached: bool
    deal_price: Optional[float]
    deal_round: Optional[int]
    termination: str           # 'accept' | 'reject' | 'max_rounds'

    # Rewards: R = surplus_captured + deal_indicator (GRPO target)
    buyer_surplus: float
    seller_surplus: float
    total_welfare: float
    buyer_reward: float        # δ^round weighted
    seller_reward: float

    # Full CoT trace for training
    turns: list[TurnData] = field(default_factory=list)


def sample_types(
    price_range: Tuple[float, float] = (500, 1500),
    delta_range: Tuple[float, float] = (0.70, 0.99),
    seed: Optional[int] = None,
) -> Tuple[BuyerType, SellerType]:
    """Sample buyer/seller types with guaranteed positive surplus zone (v_b > v_s)."""
    rng = random.Random(seed)
    v_s = rng.uniform(price_range[0], price_range[1] * 0.65)
    v_b = rng.uniform(v_s * 1.1, price_range[1])
    delta_b = rng.uniform(*delta_range)
    delta_s = rng.uniform(*delta_range)
    return BuyerType(v_b, delta_b), SellerType(v_s, delta_s)


def new_episode_id() -> str:
    return str(uuid.uuid4())[:8]

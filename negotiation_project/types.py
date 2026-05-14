import random
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class BuyerType:
    reservation_price: float   # v_b: max willingness to pay
    urgency: float              # u ∈ [0,1]: higher = more impatient
    outside_option: float       # v_o: value of best alternative deal

    def urgency_label(self) -> str:
        if self.urgency >= 0.67:
            return "high"
        elif self.urgency >= 0.33:
            return "medium"
        return "low"


@dataclass
class SellerType:
    reservation_price: float       # v_s: minimum acceptable price
    inventory_pressure: float      # p ∈ [0,1]: higher = more eager to sell

    def pressure_label(self) -> str:
        if self.inventory_pressure >= 0.67:
            return "high"
        elif self.inventory_pressure >= 0.33:
            return "medium"
        return "low"


def sample_types(
    price_range: Tuple[float, float] = (500, 1500),
    seed: Optional[int] = None,
) -> Tuple[BuyerType, SellerType]:
    rng = random.Random(seed)

    # Guarantee positive surplus zone (v_b > v_s)
    v_s = rng.uniform(price_range[0], price_range[1] * 0.65)
    v_b = rng.uniform(v_s * 1.1, price_range[1])

    u = rng.random()
    v_o = rng.uniform(0, v_b * 0.85)  # outside option is below reservation price
    p = rng.random()

    return BuyerType(v_b, u, v_o), SellerType(v_s, p)

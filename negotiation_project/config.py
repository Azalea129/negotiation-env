from dataclasses import dataclass
from enum import Enum


class Condition(str, Enum):
    FREE_FORM = "free_form"
    NUMERIC_ONLY = "numeric_only"
    TEMPLATED = "templated"
    COT_ONLY = "cot_only"


@dataclass
class NegotiationConfig:
    # Episode
    max_rounds: int = 10

    # OpenAI
    model: str = "gpt-4o"
    temperature: float = 0.7

    # δ mapping: urgency/pressure level → discount factor
    delta_low: float = 0.99     # level < 0.33
    delta_mid: float = 0.90     # 0.33 <= level < 0.67
    delta_high: float = 0.70    # level >= 0.67

    # Reward shaping
    lambda_selfishness: float = 0.5   # 0=cooperative, 1=selfish
    rho_lie_penalty: float = 0.0      # reputation penalty (Experiment B)

    # Experiment condition
    condition: Condition = Condition.FREE_FORM

    # Domain
    domain: str = "used laptop"
    item_description: str = "A used MacBook Pro (2022, M1, 16GB RAM, 512GB SSD)"

    def delta_from_level(self, level: float) -> float:
        if level < 0.33:
            return self.delta_low
        elif level < 0.67:
            return self.delta_mid
        else:
            return self.delta_high

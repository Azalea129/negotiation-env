"""
Buyer and Seller agents backed by GPT-4o.

Action protocol (all conditions):
  The agent's response must contain exactly one action tag:
    [ACCEPT]           — accept the counterparty's last price
    [REJECT]           — walk away (no deal)
    [OFFER: $X]        — propose price X

  Free-form / CoT-only: natural language may appear before/after the tag.
  Numeric-only / Templated: only the tag (or template string) is sent.

CoT-only: the model wraps reasoning in <thinking>...</thinking>; only the
  action tag is forwarded to the counterparty.
"""

import re
from typing import Optional

from openai import OpenAI

from .config import Condition, NegotiationConfig
from .types import BuyerType, SellerType

_CONDITION_INSTRUCTIONS = {
    Condition.FREE_FORM: (
        "You may communicate freely using natural language. "
        "Your full message (language + action tag) will be sent to your counterparty."
    ),
    Condition.NUMERIC_ONLY: (
        "You may ONLY send a price action tag — no other text whatsoever. "
        "Your response must be exactly one of: [ACCEPT], [REJECT], or [OFFER: $X]."
    ),
    Condition.TEMPLATED: (
        "You must use ONLY one of these fixed phrases — no other text:\n"
        '  "I propose $X"  (to counter-offer at price X)\n'
        '  "I accept"       (to accept the last price)\n'
        '  "I reject"       (to walk away)\n'
        "Convert the phrase to an action tag internally, but send only the phrase."
    ),
    Condition.COT_ONLY: (
        "First reason through your strategy inside <thinking>...</thinking> tags "
        "(this reasoning stays private). Then output your action tag on its own line. "
        "Only the action tag will be forwarded to your counterparty — they cannot see your thinking."
    ),
}

_BUYER_SYSTEM = """\
You are a buyer negotiating to purchase: {item_description}

YOUR PRIVATE INFORMATION (do NOT reveal these exact numbers):
- Your reservation price (maximum you will pay): ${reservation_price:.0f}
- Your urgency level: {urgency_label} (urgency={urgency:.2f})
  → Each round of delay costs you: your payoff is multiplied by {delta:.3f} per round.
  → If urgency is high, a long negotiation hurts you — but revealing urgency gives the seller power.
- Your outside option: you have an alternative deal worth ${outside_option:.0f} to you.
  → If no deal is reached, you fall back on this alternative.

GOAL: Maximize your surplus = (reservation_price − final_price) × δ^round.

NEGOTIATION RULES:
- The seller will make offers; you respond each turn.
- Maximum {max_rounds} rounds total (buyer + seller turns each count).
- At any point you may accept, reject, or counter-offer.

{condition_instructions}

ACTION TAG FORMAT (required in every response):
  [ACCEPT]        — accept the seller's last proposed price
  [REJECT]        — end negotiations (no deal)
  [OFFER: $X]     — propose a new price of $X
"""

_SELLER_SYSTEM = """\
You are a seller negotiating to sell: {item_description}

YOUR PRIVATE INFORMATION (do NOT reveal these exact numbers):
- Your reservation price (minimum you will accept): ${reservation_price:.0f}
- Your inventory pressure: {pressure_label} (pressure={pressure:.2f})
  → Each round of delay costs you: your payoff is multiplied by {delta:.3f} per round.
  → High pressure means you want to close quickly — but revealing it weakens your position.

GOAL: Maximize your profit = (final_price − reservation_price) × δ^round.

NEGOTIATION RULES:
- The buyer will make the first offer; you respond each turn.
- Maximum {max_rounds} rounds total.
- At any point you may accept, reject, or counter-offer.

{condition_instructions}

ACTION TAG FORMAT (required in every response):
  [ACCEPT]        — accept the buyer's last proposed price
  [REJECT]        — end negotiations (no deal)
  [OFFER: $X]     — propose a new price of $X
"""


def _extract_action(text: str) -> dict:
    """
    Parse action tag from agent output.
    Returns dict with keys: action ('accept'|'reject'|'offer'), price (float|None).
    """
    text_upper = text.upper()

    if "[ACCEPT]" in text_upper or "I ACCEPT" in text_upper:
        return {"action": "accept", "price": None}

    if "[REJECT]" in text_upper or "I REJECT" in text_upper:
        return {"action": "reject", "price": None}

    # [OFFER: $X] or "I propose $X"
    offer_match = re.search(
        r"\[OFFER:\s*\$?([\d,]+(?:\.\d+)?)\]"
        r"|I PROPOSE\s*\$?([\d,]+(?:\.\d+)?)",
        text_upper,
    )
    if offer_match:
        raw = offer_match.group(1) or offer_match.group(2)
        price = float(raw.replace(",", ""))
        return {"action": "offer", "price": price}

    # Fallback: any bare dollar amount
    dollar_match = re.search(r"\$\s*([\d,]+(?:\.\d+)?)", text)
    if dollar_match:
        price = float(dollar_match.group(1).replace(",", ""))
        return {"action": "offer", "price": price}

    return {"action": "unknown", "price": None}


def _filter_message(raw: str, condition: Condition) -> str:
    """Return the string that will actually be sent to the counterparty."""
    if condition == Condition.COT_ONLY:
        # Strip <thinking>...</thinking> block; forward only what remains
        stripped = re.sub(r"<thinking>.*?</thinking>", "", raw, flags=re.DOTALL).strip()
        return stripped

    if condition == Condition.NUMERIC_ONLY:
        # Only the action tag
        match = re.search(
            r"(\[ACCEPT\]|\[REJECT\]|\[OFFER:\s*\$?[\d,]+(?:\.\d+)?\])",
            raw,
            re.IGNORECASE,
        )
        return match.group(0) if match else raw.strip()

    # FREE_FORM and TEMPLATED: send as-is
    return raw.strip()


class NegotiationAgent:
    def __init__(self, role: str, system_prompt: str, config: NegotiationConfig):
        self.role = role
        self.config = config
        self.client = OpenAI()
        self.history: list[dict] = [{"role": "system", "content": system_prompt}]

    def respond(self, incoming_message: Optional[str]) -> dict:
        """
        Process incoming message and generate a response.

        Returns:
            raw_response:   full model output (for logging)
            visible_message: text forwarded to counterparty
            action:         'accept' | 'reject' | 'offer' | 'unknown'
            price:          float or None
        """
        if incoming_message is not None:
            self.history.append({"role": "user", "content": incoming_message})

        completion = self.client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            messages=self.history,
        )
        raw = completion.choices[0].message.content
        self.history.append({"role": "assistant", "content": raw})

        parsed = _extract_action(raw)
        visible = _filter_message(raw, self.config.condition)

        return {
            "raw_response": raw,
            "visible_message": visible,
            "action": parsed["action"],
            "price": parsed["price"],
        }

    def get_last_proposed_price(self) -> Optional[float]:
        """Scan history to find the most recent price proposal."""
        for msg in reversed(self.history):
            if msg["role"] == "user":
                parsed = _extract_action(msg["content"])
                if parsed["price"] is not None:
                    return parsed["price"]
        return None


def make_buyer_agent(buyer_type: BuyerType, config: NegotiationConfig) -> NegotiationAgent:
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
    return NegotiationAgent("buyer", system, config)


def make_seller_agent(seller_type: SellerType, config: NegotiationConfig) -> NegotiationAgent:
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
    return NegotiationAgent("seller", system, config)

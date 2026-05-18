"""
LASH negotiation agents with structured BDI CoT.

Each agent response follows the cognitive hierarchy from the proposal:
  1. <belief>  — infer opponent's Type and current state (B_t^GT)
  2. <intention> — form own strategic intention (I_t^GT)
  3. [ACTION TAG] — message derived only from intention (M_t)

The <belief> and <intention> blocks are PRIVATE (stripped before forwarding
to the counterparty). They become the ground-truth supervision signal for
Stage 1 latent hypothesis training.
"""

import re
from typing import Optional

from openai import OpenAI

from .config import LASHConfig
from .types import BuyerType, SellerType


# ── Prompts ────────────────────────────────────────────────────────────────

_BUYER_SYSTEM = """\
You are a buyer negotiating to purchase: {item_description}

YOUR PRIVATE INFORMATION (never reveal these exact numbers):
- Reservation price (maximum you will pay): ${reservation_price:.0f}
- Patience (δ={delta:.3f}): each round of delay multiplies your payoff by {delta:.3f}

GOAL: Maximize surplus = (reservation_price − final_price) × δ^round

NEGOTIATION RULES:
- You make the opening offer (round 0).
- Turns alternate: buyer → seller → buyer → ...
- Maximum {max_rounds} rounds total.
- You may ACCEPT, REJECT, or counter-OFFER at any turn.

REASONING FORMAT — you MUST follow this structure every turn:

<belief>
Based on the seller's offers and language, estimate:
- Their likely reservation price range (minimum they will accept)
- Their patience/urgency level (are they eager to close?)
- Whether their current position is a bluff or a genuine limit
</belief>
<intention>
Given your belief about the seller's state, decide your strategy:
- What price will you offer/accept and why?
- How does this exploit or accommodate the seller's estimated state?
</intention>
{message_section}
[ACTION TAG]

ACTION TAGS (one required{action_tag_note}):
  [ACCEPT]      — accept the seller's last proposed price
  [REJECT]      — walk away (no deal)
  [OFFER: $X]   — propose price $X

{visibility_note}
"""

_SELLER_SYSTEM = """\
You are a seller negotiating to sell: {item_description}

YOUR PRIVATE INFORMATION (never reveal these exact numbers):
- Reservation price (minimum you will accept): ${reservation_price:.0f}
- Patience (δ={delta:.3f}): each round of delay multiplies your payoff by {delta:.3f}

GOAL: Maximize profit = (final_price − reservation_price) × δ^round

NEGOTIATION RULES:
- The buyer makes the opening offer; you respond each turn.
- Maximum {max_rounds} rounds total.
- You may ACCEPT, REJECT, or counter-OFFER at any turn.

REASONING FORMAT — you MUST follow this structure every turn:

<belief>
Based on the buyer's offers and language, estimate:
- Their likely reservation price range (maximum they will pay)
- Their patience/urgency level (are they eager to close?)
- Whether their current position is a bluff or a genuine limit
</belief>
<intention>
Given your belief about the buyer's state, decide your strategy:
- What price will you offer/accept and why?
- How does this exploit or accommodate the buyer's estimated state?
</intention>
{message_section}
[ACTION TAG]

ACTION TAGS (one required{action_tag_note}):
  [ACCEPT]      — accept the buyer's last proposed price
  [REJECT]      — walk away (no deal)
  [OFFER: $X]   — propose price $X

{visibility_note}
"""

_MSG_SECTION_BUYER = """\
<message>
[Natural language message to send to the seller — 1 to 3 sentences.
 Do NOT reveal your reservation price. This is the only part the seller sees.]
</message>"""

_MSG_SECTION_SELLER = """\
<message>
[Natural language message to send to the buyer — 1 to 3 sentences.
 Do NOT reveal your reservation price. This is the only part the buyer sees.]
</message>"""

_VISIBILITY_WITH_MSG = (
    "The <belief>, <intention>, and [ACTION TAG] are PRIVATE.\n"
    "Only the <message> block is forwarded to the counterparty."
)
_VISIBILITY_NO_MSG = (
    "The <belief> and <intention> sections are PRIVATE.\n"
    "The [ACTION TAG] is forwarded directly to the counterparty."
)
_ACTION_TAG_NOTE_WITH_MSG = ", placed after </message>, PRIVATE — not shown to the counterparty"
_ACTION_TAG_NOTE_NO_MSG = ""


# ── Parsing helpers ────────────────────────────────────────────────────────

def extract_cot_sections(raw: str) -> tuple[str, str]:
    """
    Extract belief and intention text from structured CoT output.
    Returns (belief_text, intention_text); empty string if tag absent.
    """
    belief_match = re.search(r"<belief>(.*?)</belief>", raw, re.DOTALL | re.IGNORECASE)
    intention_match = re.search(r"<intention>(.*?)</intention>", raw, re.DOTALL | re.IGNORECASE)
    belief = belief_match.group(1).strip() if belief_match else ""
    intention = intention_match.group(1).strip() if intention_match else ""
    return belief, intention


def extract_message(raw: str) -> str:
    """
    Extract the visible natural-language message from the <message> tag.
    Falls back to strip_cot() if the tag is absent (e.g. older model outputs).
    """
    m = re.search(r"<message>(.*?)</message>", raw, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return strip_cot(raw)


def extract_action(text: str) -> dict:
    """
    Parse action tag from agent output.
    Returns {"action": str, "price": float | None}.
    """
    upper = text.upper()

    if "[ACCEPT]" in upper or "I ACCEPT" in upper:
        return {"action": "accept", "price": None}

    if "[REJECT]" in upper or "I REJECT" in upper:
        return {"action": "reject", "price": None}

    offer_match = re.search(
        r"\[OFFER:\s*\$?([\d,]+(?:\.\d+)?)\]|I PROPOSE\s*\$?([\d,]+(?:\.\d+)?)",
        upper,
    )
    if offer_match:
        raw_price = offer_match.group(1) or offer_match.group(2)
        return {"action": "offer", "price": float(raw_price.replace(",", ""))}

    dollar_match = re.search(r"\$\s*([\d,]+(?:\.\d+)?)", text)
    if dollar_match:
        return {"action": "offer", "price": float(dollar_match.group(1).replace(",", ""))}

    return {"action": "unknown", "price": None}


def strip_cot(raw: str) -> str:
    """
    Remove <belief> and <intention> blocks; return only the visible message.
    M_t = g(I_t^self) — counterparty sees only the action tag and surrounding text.
    """
    stripped = re.sub(r"<belief>.*?</belief>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    stripped = re.sub(r"<intention>.*?</intention>", "", stripped, flags=re.DOTALL | re.IGNORECASE)
    return stripped.strip()


# ── Agent class ────────────────────────────────────────────────────────────

class NegotiationAgent:
    def __init__(self, role: str, system_prompt: str, config: LASHConfig):
        self.role = role
        self.config = config

        client_kwargs: dict = {}
        if config.api_base:
            client_kwargs["base_url"] = config.api_base
        if config.api_key:
            client_kwargs["api_key"] = config.api_key

        self.client = OpenAI(**client_kwargs)
        self.history: list[dict] = [{"role": "system", "content": system_prompt}]

    def respond(self, incoming_message: Optional[str]) -> dict:
        """
        Process incoming message and generate a structured BDI response.

        Returns:
          raw_cot:         full model output including <belief>/<intention> tags
          belief_text:     B_t^GT — extracted belief reasoning
          intention_text:  I_t^GT — extracted intention reasoning
          visible_message: M_t — stripped of private CoT blocks
          action:          'accept' | 'reject' | 'offer' | 'unknown'
          price:           float or None
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

        belief, intention = extract_cot_sections(raw)
        visible = extract_message(raw)
        parsed = extract_action(raw)

        return {
            "raw_cot": raw,
            "belief_text": belief,
            "intention_text": intention,
            "visible_message": visible,
            "action": parsed["action"],
            "price": parsed["price"],
        }

    def get_last_proposed_price(self) -> Optional[float]:
        for msg in reversed(self.history):
            if msg["role"] == "user":
                parsed = extract_action(msg["content"])
                if parsed["price"] is not None:
                    return parsed["price"]
        return None


# ── Factory functions ──────────────────────────────────────────────────────

def make_buyer_agent(buyer_type: BuyerType, config: LASHConfig) -> NegotiationAgent:
    use_msg = config.natural_language_message
    system = _BUYER_SYSTEM.format(
        item_description=config.item_description,
        reservation_price=buyer_type.reservation_price,
        delta=buyer_type.delta,
        max_rounds=config.max_rounds,
        message_section=_MSG_SECTION_BUYER if use_msg else "",
        action_tag_note=_ACTION_TAG_NOTE_WITH_MSG if use_msg else _ACTION_TAG_NOTE_NO_MSG,
        visibility_note=_VISIBILITY_WITH_MSG if use_msg else _VISIBILITY_NO_MSG,
    )
    return NegotiationAgent("buyer", system, config)


def make_seller_agent(seller_type: SellerType, config: LASHConfig) -> NegotiationAgent:
    use_msg = config.natural_language_message
    system = _SELLER_SYSTEM.format(
        item_description=config.item_description,
        reservation_price=seller_type.reservation_price,
        delta=seller_type.delta,
        max_rounds=config.max_rounds,
        message_section=_MSG_SECTION_SELLER if use_msg else "",
        action_tag_note=_ACTION_TAG_NOTE_WITH_MSG if use_msg else _ACTION_TAG_NOTE_NO_MSG,
        visibility_note=_VISIBILITY_WITH_MSG if use_msg else _VISIBILITY_NO_MSG,
    )
    return NegotiationAgent("seller", system, config)

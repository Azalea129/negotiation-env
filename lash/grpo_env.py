"""
GRPO rollout environment for LASH Stage 2.

Runs A2A negotiation episodes with LASHModel directly (no OpenAI API),
capturing per-turn token sequences and log probabilities needed for
policy gradient computation.

Both buyer and seller use the same LASHModel; the role-specific system prompt
(with private type info) is prepended to each turn's context so the model
knows its reservation price and delta.

Data flow per turn:
  ctx_text (system + history) → tokenize → ctx_ids
  → model.generate() → msg_ids → decode → parse action
  → _sum_log_prob() → old_log_prob  (stored for importance ratio in GRPO update)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

from .agents import (
    _BUYER_SYSTEM, _SELLER_SYSTEM,
    _MSG_SECTION_BUYER, _MSG_SECTION_SELLER,
    _ACTION_TAG_NOTE_WITH_MSG, _ACTION_TAG_NOTE_NO_MSG,
    _VISIBILITY_WITH_MSG, _VISIBILITY_NO_MSG,
    extract_action, strip_cot,
)
from .config import LASHConfig
from .types import BuyerType, SellerType


# ── Rollout data structures ────────────────────────────────────────────────

@dataclass
class RolloutTurn:
    role: str               # 'buyer' | 'seller'
    ctx_ids: torch.Tensor  # (L_c,) — context token ids
    ctx_mask: torch.Tensor # (L_c,) — context attention mask
    msg_ids: torch.Tensor  # (L_m,) — generated message token ids
    old_log_prob: float    # Σ log π_old(token | prefix) — stored for importance ratio
    advantage: float = 0.0 # filled after group-relative normalization


@dataclass
class RolloutEpisode:
    turns: list[RolloutTurn]
    buyer_reward: float
    seller_reward: float
    deal_reached: bool
    deal_price: Optional[float]


# ── Rollout environment ────────────────────────────────────────────────────

class GRPORolloutEnv:
    """
    Runs A2A negotiation with LASHModel and captures rollout data for GRPO.

    The context at each turn includes:
      1. Role-specific system prompt (private info: reservation price, delta)
      2. Visible conversation history so far
    This format differs from Stage 1 training data (which stored only the visible
    history as c_t) — the system prompt is added so the model knows its own type.
    """

    def __init__(
        self,
        model,
        tokenizer,
        sim_config: LASHConfig,
        max_ctx_len: int = 768,
        max_gen_len: int = 512,
        temperature: float = 0.8,
        device: str = "cuda",
    ):
        self.model = model
        self.tok = tokenizer
        self.cfg = sim_config
        self.max_ctx_len = max_ctx_len
        self.max_gen_len = max_gen_len
        self.temperature = temperature
        self.device = device

    @torch.no_grad()
    def run_episode(
        self,
        buyer_type: BuyerType,
        seller_type: SellerType,
    ) -> RolloutEpisode:
        use_msg = self.cfg.natural_language_message
        buyer_sys = _BUYER_SYSTEM.format(
            item_description=self.cfg.item_description,
            reservation_price=buyer_type.reservation_price,
            delta=buyer_type.delta,
            max_rounds=self.cfg.max_rounds,
            message_section=_MSG_SECTION_BUYER if use_msg else "",
            action_tag_note=_ACTION_TAG_NOTE_WITH_MSG if use_msg else _ACTION_TAG_NOTE_NO_MSG,
            visibility_note=_VISIBILITY_WITH_MSG if use_msg else _VISIBILITY_NO_MSG,
        )
        seller_sys = _SELLER_SYSTEM.format(
            item_description=self.cfg.item_description,
            reservation_price=seller_type.reservation_price,
            delta=seller_type.delta,
            max_rounds=self.cfg.max_rounds,
            message_section=_MSG_SECTION_SELLER if use_msg else "",
            action_tag_note=_ACTION_TAG_NOTE_WITH_MSG if use_msg else _ACTION_TAG_NOTE_NO_MSG,
            visibility_note=_VISIBILITY_WITH_MSG if use_msg else _VISIBILITY_NO_MSG,
        )

        context_lines: list[str] = []
        turns: list[RolloutTurn] = []
        current_price: Optional[float] = None
        turn_index = 0
        rnd = 0

        def ctx_text(system: str) -> str:
            if context_lines:
                hist = "\n".join(context_lines)
                return f"{system}\n\n=== Negotiation so far ===\n{hist}\n\nYour response:"
            return f"{system}\n\nYour response (opening offer):"

        def do_turn(role: str, system: str) -> str:
            nonlocal turn_index
            msg_text, ctx_ids, ctx_mask, msg_ids, lp = self._generate_turn(ctx_text(system))
            visible = strip_cot(msg_text)
            context_lines.append(f"[{role.upper()} turn {turn_index}]: {visible}")
            turns.append(RolloutTurn(role=role, ctx_ids=ctx_ids, ctx_mask=ctx_mask,
                                     msg_ids=msg_ids, old_log_prob=lp))
            turn_index += 1
            return visible

        # Buyer opens
        visible = do_turn("buyer", buyer_sys)
        parsed = extract_action(visible)
        if parsed["action"] == "reject":
            return self._make_episode(turns, buyer_type, seller_type, None, rnd, "reject")
        if parsed["price"] is not None:
            current_price = parsed["price"]

        max_turns = self.cfg.max_rounds * 2
        while turn_index <= max_turns:
            # Seller
            visible = do_turn("seller", seller_sys)
            parsed = extract_action(visible)
            if parsed["action"] == "accept":
                return self._make_episode(turns, buyer_type, seller_type, current_price, rnd, "accept")
            if parsed["action"] == "reject":
                return self._make_episode(turns, buyer_type, seller_type, None, rnd, "reject")
            if parsed["price"] is not None:
                current_price = parsed["price"]
            rnd += 1

            if rnd >= self.cfg.max_rounds:
                return self._make_episode(turns, buyer_type, seller_type, None, rnd, "max_rounds")

            # Buyer
            visible = do_turn("buyer", buyer_sys)
            parsed = extract_action(visible)
            if parsed["action"] == "accept":
                return self._make_episode(turns, buyer_type, seller_type, current_price, rnd, "accept")
            if parsed["action"] == "reject":
                return self._make_episode(turns, buyer_type, seller_type, None, rnd, "reject")
            if parsed["price"] is not None:
                current_price = parsed["price"]

        return self._make_episode(turns, buyer_type, seller_type, None, rnd, "max_rounds")

    # ── Internal helpers ───────────────────────────────────────────────────

    def _make_episode(
        self,
        turns: list[RolloutTurn],
        buyer_type: BuyerType,
        seller_type: SellerType,
        deal_price: Optional[float],
        rnd: int,
        termination: str,
    ) -> RolloutEpisode:
        lam = self.cfg.lambda_selfishness
        bonus = self.cfg.deal_bonus
        if deal_price is not None:
            bs = max(0.0, buyer_type.reservation_price - deal_price) * (buyer_type.delta ** rnd)
            ss = max(0.0, deal_price - seller_type.reservation_price) * (seller_type.delta ** rnd)
            tw = bs + ss
            br = lam * bs + (1 - lam) * tw + bonus
            sr = lam * ss + (1 - lam) * tw + bonus
        else:
            br = sr = 0.0
        return RolloutEpisode(
            turns=turns,
            buyer_reward=br,
            seller_reward=sr,
            deal_reached=deal_price is not None,
            deal_price=deal_price,
        )

    @torch.no_grad()
    def _generate_turn(
        self, ctx_text: str
    ) -> tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        enc = self.tok(
            ctx_text,
            max_length=self.max_ctx_len,
            truncation=True,
            return_tensors="pt",
        )
        ctx_ids = enc["input_ids"].to(self.device)   # (1, L_c)
        ctx_mask = enc["attention_mask"].to(self.device)

        # model.generate() with inputs_embeds returns only the newly generated token ids
        gen_ids = self.model.generate(
            ctx_ids,
            ctx_mask,
            max_new_tokens=self.max_gen_len,
            do_sample=True,
            temperature=self.temperature,
            pad_token_id=self.tok.pad_token_id,
            eos_token_id=self.tok.eos_token_id,
        )
        msg_ids = gen_ids[0]   # (L_m,)

        log_prob = self._sum_log_prob(ctx_ids, ctx_mask, msg_ids).item()
        msg_text = self.tok.decode(msg_ids, skip_special_tokens=True)

        return (
            msg_text,
            ctx_ids.squeeze(0),    # (L_c,)
            ctx_mask.squeeze(0),   # (L_c,)
            msg_ids,               # (L_m,)
            log_prob,
        )

    @torch.no_grad()
    def _sum_log_prob(
        self,
        ctx_ids: torch.Tensor,  # (1, L_c)
        ctx_mask: torch.Tensor, # (1, L_c)
        msg_ids: torch.Tensor,  # (L_m,)
    ) -> torch.Tensor:
        """Sum log prob of msg_ids under current policy (stop-gradient)."""
        full_ids = torch.cat([ctx_ids, msg_ids.unsqueeze(0)], dim=1)      # (1, L_c+L_m)
        full_mask = torch.ones(1, full_ids.size(1), dtype=torch.long, device=self.device)

        h_b, h_i, c_pooled = self.model.extract_hypotheses(ctx_ids, ctx_mask)
        z_t, _ = self.model.hypothesis_attn(c_pooled, h_b, h_i)
        _, logits = self.model.forward_generation(z_t, full_ids, full_mask, labels=None)
        # logits: (1, 1+L_c+L_m, vocab)
        # logits[:, ctx_len : ctx_len+msg_len, :] predicts msg_ids[0..msg_len-1]

        ctx_len = ctx_ids.size(1)
        msg_len = msg_ids.size(0)
        msg_logits = logits[0, ctx_len: ctx_len + msg_len, :]             # (L_m, vocab)
        log_probs = F.log_softmax(msg_logits, dim=-1)
        return log_probs[torch.arange(msg_len, device=self.device), msg_ids].sum()

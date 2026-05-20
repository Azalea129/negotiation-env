"""
LASHModel: full LASH architecture wrapping Llama 3.x.

Forward pass has two phases per turn t:

  Pass 1 — Hypothesis Extraction (c_t → H^B_t, H^I_t)
  ┌─────────────────────────────────────────────────────────────────┐
  │  [HYP_1 ... HYP_k | c_t tokens]                                │
  │         ↓  Llama backbone (LoRA fine-tuned)                     │
  │  last hidden states[:k]  → H^B_t  ∈ R^{k × d}  (belief)       │
  │  mean(hidden states[k:]) → c_pooled ∈ R^{d}                    │
  │  IntentionMLP(H^B_t, c_pooled) → H^I_t  ∈ R^{k × d}           │
  │  HypothesisAttention(c_pooled, H^B_t, H^I_t) → z_t ∈ R^{d}    │
  └─────────────────────────────────────────────────────────────────┘

  Pass 2 — Message Generation (z_t, c_t → M_t)
  ┌─────────────────────────────────────────────────────────────────┐
  │  [z_embed | c_t embeds]    (z_t projected into embedding space) │
  │         ↓  Llama backbone (same LoRA weights)                   │
  │  autoregressive generation → M_t                               │
  └─────────────────────────────────────────────────────────────────┘

Stage 1 training adds auxiliary supervision:
  L_belief    = InfoNCE(H^B_t, encode(B_t^GT))
  L_intention = InfoNCE(H^I_t, encode(I_t^GT))
  L_lm        = NLL on M_t tokens
  L_total     = L_lm + α·L_belief + β·L_intention

Oracle-to-Self curriculum:
  During Stage 1, z_t can be blended with the oracle GT embedding:
    z_curriculum = oracle_ratio * z_oracle + (1 - oracle_ratio) * z_t
  oracle_ratio decays from 1.0 → 0.0 across training.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .config import LASHModelConfig
from .modules import HypothesisAttention, IntentionMLP, compute_alignment_loss, encode_gt_text


class LASHModel(nn.Module):
    def __init__(self, base_model: nn.Module, config: LASHModelConfig):
        """
        Args:
            base_model: a LlamaForCausalLM (or PEFT-wrapped variant).
                        Access to base_model.model.embed_tokens and
                        base_model.model (transformer) is required.
            config:     LASHModelConfig
        """
        super().__init__()
        self.base = base_model
        self.cfg = config
        d, k = config.d_model, config.k

        # ── k learnable hypothesis query tokens (Pass 1 prefix) ──────────
        # Initialized small so they don't dominate early token representations.
        self.hyp_queries = nn.Parameter(torch.randn(k, d) * 0.02)

        # ── LASH modules ──────────────────────────────────────────────────
        self.intention_mlp = IntentionMLP(d, k)
        self.hypothesis_attn = HypothesisAttention(d, config.n_heads)

        # ── z_t → embedding space projection (Pass 2 soft prefix) ────────
        self.z_to_embed = nn.Sequential(
            nn.Linear(d, d),
            nn.LayerNorm(d),
        )

    # ── Internal helpers ────────────────────────────────────────────────

    @property
    def _llama_model(self) -> nn.Module:
        """Return the inner LlamaModel (has embed_tokens), unwrapping PEFT layers."""
        m = self.base
        while not hasattr(m, 'embed_tokens'):
            m = m.model
        return m

    def _embed(self, input_ids: Tensor) -> Tensor:
        """Token id → embedding via base LLM's embedding table."""
        return self._llama_model.embed_tokens(input_ids)

    def _backbone_hidden(
        self,
        inputs_embeds: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Run Llama transformer (no LM head), return last hidden states."""
        # Cast to backbone dtype (e.g. bfloat16) — LASH modules init in float32
        dtype = next(self._llama_model.parameters()).dtype
        out = self._llama_model(
            inputs_embeds=inputs_embeds.to(dtype),
            attention_mask=attention_mask,
        )
        return out.last_hidden_state.float()   # (B, seq, d) — back to float32 for LASH modules

    def _mask_pool(self, hidden: Tensor, mask: Tensor) -> Tensor:
        """Mask-aware mean pooling: (B, seq, d), (B, seq) → (B, d)."""
        m = mask.unsqueeze(-1).float()
        return (hidden * m).sum(1) / m.sum(1).clamp(min=1)

    # ── Pass 1: hypothesis extraction ───────────────────────────────────

    def extract_hypotheses(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        c_t → H^B_t, H^I_t, c_pooled.

        Returns:
            h_belief:    (B, k, d)
            h_intention: (B, k, d)
            c_pooled:    (B, d)
        """
        B = input_ids.size(0)
        k = self.cfg.k

        # Embed context tokens
        c_embeds = self._embed(input_ids)                             # (B, seq, d)

        # Prepend k hypothesis query tokens
        hyp = self.hyp_queries.unsqueeze(0).expand(B, -1, -1)        # (B, k, d)
        inputs_embeds = torch.cat([hyp, c_embeds], dim=1)            # (B, k+seq, d)

        # Extend attention mask (hypothesis tokens are always attended to)
        hyp_mask = attention_mask.new_ones(B, k)
        full_mask = torch.cat([hyp_mask, attention_mask], dim=1)     # (B, k+seq)

        hidden = self._backbone_hidden(inputs_embeds, full_mask)     # (B, k+seq, d)

        h_belief = hidden[:, :k, :]                                  # (B, k, d)
        c_hidden = hidden[:, k:, :]                                  # (B, seq, d)
        c_pooled = self._mask_pool(c_hidden, attention_mask)         # (B, d)

        h_intention = self.intention_mlp(h_belief, c_pooled)         # (B, k, d)
        return h_belief, h_intention, c_pooled

    # ── Pass 2: generation conditioned on z_t ───────────────────────────

    def forward_generation(
        self,
        z_t: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Optional[Tensor] = None,
    ) -> tuple[Optional[Tensor], Tensor]:
        """
        Prepend z_t as a single soft token, then run the LM head.

        Returns:
            lm_loss:  scalar or None (None if labels not provided)
            logits:   (B, 1+seq, vocab)
        """
        B = input_ids.size(0)

        dtype = next(self._llama_model.parameters()).dtype

        z_embed = self.z_to_embed(z_t).unsqueeze(1)                  # (B, 1, d)
        c_embeds = self._embed(input_ids)                            # (B, seq, d)
        gen_embeds = torch.cat([z_embed, c_embeds], dim=1).to(dtype) # (B, 1+seq, d)

        z_mask = attention_mask.new_ones(B, 1)
        gen_mask = torch.cat([z_mask, attention_mask], dim=1)        # (B, 1+seq)

        if labels is not None:
            # Prepend -100 so the z token position is ignored in loss
            ignore = labels.new_full((B, 1), -100)
            labels_ext = torch.cat([ignore, labels], dim=1)          # (B, 1+seq)
        else:
            labels_ext = None

        out = self.base(
            inputs_embeds=gen_embeds,
            attention_mask=gen_mask,
            labels=labels_ext,
        )
        return out.loss, out.logits

    # ── Full forward (Stage 1 SFT) ───────────────────────────────────────

    def forward(
        self,
        # Context (c_t)
        input_ids: Tensor,
        attention_mask: Tensor,
        # Message M_t labels (for L_lm)
        labels: Optional[Tensor] = None,
        # Ground-truth belief text B_t^GT (for L_belief)
        belief_input_ids: Optional[Tensor] = None,
        belief_attention_mask: Optional[Tensor] = None,
        # Ground-truth intention text I_t^GT (for L_intention)
        intention_input_ids: Optional[Tensor] = None,
        intention_attention_mask: Optional[Tensor] = None,
        # Oracle-to-Self curriculum
        oracle_ratio: float = 0.0,          # 0 = full self; controlled by trainer
        oracle_z: Optional[Tensor] = None,  # precomputed oracle z_t (if ratio > 0)
    ) -> dict[str, Tensor]:
        """
        Full forward pass for Stage 1 SFT.

        Returns dict with keys: lm_loss, belief_loss, intention_loss, total_loss.
        Losses are only computed when the corresponding inputs are provided.
        """
        # Pass 1
        h_belief, h_intention, c_pooled = self.extract_hypotheses(input_ids, attention_mask)

        # Hypothesis attention → z_t
        z_t, attn_weights = self.hypothesis_attn(c_pooled, h_belief, h_intention)

        # Oracle-to-Self curriculum blending
        if oracle_ratio > 0.0 and oracle_z is not None:
            z_t = oracle_ratio * oracle_z + (1.0 - oracle_ratio) * z_t

        # Pass 2: LM loss on M_t
        lm_loss, _ = self.forward_generation(z_t, input_ids, attention_mask, labels)

        losses: dict[str, Tensor] = {}
        total = lm_loss if lm_loss is not None else z_t.new_tensor(0.0)

        if lm_loss is not None:
            losses["lm_loss"] = lm_loss

        # Auxiliary loss: belief alignment
        if belief_input_ids is not None:
            gt_belief = encode_gt_text(
                lambda ids, mask: self._backbone_hidden(self._embed(ids), mask),
                belief_input_ids,
                belief_attention_mask,
            )
            losses["belief_loss"] = compute_alignment_loss(
                h_belief, gt_belief, self.cfg.contrastive_temperature
            )
            total = total + self.cfg.alpha_belief * losses["belief_loss"]

        # Auxiliary loss: intention alignment
        if intention_input_ids is not None:
            gt_intention = encode_gt_text(
                lambda ids, mask: self._backbone_hidden(self._embed(ids), mask),
                intention_input_ids,
                intention_attention_mask,
            )
            losses["intention_loss"] = compute_alignment_loss(
                h_intention, gt_intention, self.cfg.contrastive_temperature
            )
            total = total + self.cfg.beta_intention * losses["intention_loss"]

        losses["total_loss"] = total
        losses["attn_weights"] = attn_weights   # (B, 1, 2k) — for interpretability logging
        return losses

    # ── Inference helpers ────────────────────────────────────────────────

    @torch.inference_mode()
    def get_hypothesis_weights(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """
        Return the cross-attention weights over 2k hypotheses.
        Shape: (B, 2k) — interpretable as which (belief, intention) hypothesis
        the agent is "relying on" given the current context.
        """
        _, _, c_pooled = self.extract_hypotheses(input_ids, attention_mask)
        h_b, h_i, c_pooled = self.extract_hypotheses(input_ids, attention_mask)
        _, weights = self.hypothesis_attn(c_pooled, h_b, h_i)
        return weights.squeeze(1)    # (B, 2k)

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        oracle_ratio: float = 0.0,
        oracle_z: Optional[Tensor] = None,
        **generate_kwargs,
    ) -> Tensor:
        """
        Autoregressive generation conditioned on z_t.
        Extra kwargs are forwarded to base_model.generate().
        """
        h_b, h_i, c_pooled = self.extract_hypotheses(input_ids, attention_mask)
        z_t, _ = self.hypothesis_attn(c_pooled, h_b, h_i)

        if oracle_ratio > 0.0 and oracle_z is not None:
            z_t = oracle_ratio * oracle_z + (1.0 - oracle_ratio) * z_t

        dtype = next(self._llama_model.parameters()).dtype
        B = input_ids.size(0)
        z_embed = self.z_to_embed(z_t).unsqueeze(1)                  # (B, 1, d)
        c_embeds = self._embed(input_ids)
        gen_embeds = torch.cat([z_embed, c_embeds], dim=1).to(dtype)
        z_mask = attention_mask.new_ones(B, 1)
        gen_mask = torch.cat([z_mask, attention_mask], dim=1)

        return self.base.generate(
            inputs_embeds=gen_embeds,
            attention_mask=gen_mask,
            **generate_kwargs,
        )

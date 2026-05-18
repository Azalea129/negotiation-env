"""
LASH neural modules.

Three components, matching the proposal's architecture:

  H^B_t = f_Belief(c_t)              ← hypothesis tokens in backbone forward pass
  H^I_t = MLP_Intention(H^B_t, c_t)  ← IntentionMLP (BDI causal ordering)
  z_t   = Attention(q_t, H^B⊕H^I)   ← HypothesisAttention (soft selection)

Plus auxiliary supervision helpers for Stage 1:
  alignment_loss(H_pred, gt_input_ids)  ← InfoNCE in LLM embedding space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ── IntentionMLP ────────────────────────────────────────────────────────────

class IntentionMLP(nn.Module):
    """
    Derives k intention hypothesis vectors from belief vectors + context.

    H^I_t = MLP(concat(H^B_t, c_pooled_expanded))  ∈ R^{B × k × d}

    Enforces BDI causal ordering: Belief must be computed before Intention.
    The concatenation of H^B_t (opponent belief) with c_pooled (own context)
    mirrors the agent using its belief about the opponent to shape its own strategy.
    """

    def __init__(self, d_model: int, k: int):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h_belief: Tensor, c_pooled: Tensor) -> Tensor:
        """
        Args:
            h_belief:  (B, k, d) — belief hypothesis vectors
            c_pooled:  (B, d)    — mask-aware mean-pool of context hidden states
        Returns:
            h_intention: (B, k, d)
        """
        # Broadcast c_pooled across all k hypotheses
        c_exp = c_pooled.unsqueeze(1).expand(-1, self.k, -1)       # (B, k, d)
        h_int = self.mlp(torch.cat([h_belief, c_exp], dim=-1))      # (B, k, d)
        return self.norm(h_int)


# ── HypothesisAttention ─────────────────────────────────────────────────────

class HypothesisAttention(nn.Module):
    """
    Soft selection over 2k hypotheses via cross-attention.

    z_t = Attention(Q=q_t, K=[H^B ⊕ H^I], V=[H^B ⊕ H^I])

    The softmax weights over the 2k keys serve as an interpretable uncertainty
    distribution: which (belief, intention) hypothesis does the agent rely on
    most given the current context?
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        c_pooled: Tensor,
        h_belief: Tensor,
        h_intention: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            c_pooled:    (B, d)    — context representation (query source)
            h_belief:    (B, k, d) — belief hypotheses
            h_intention: (B, k, d) — intention hypotheses
        Returns:
            z_t:         (B, d)    — aggregated latent cognitive state
            attn_weights:(B, 1, 2k)— hypothesis selection distribution
        """
        q = self.q_proj(c_pooled).unsqueeze(1)              # (B, 1, d)
        kv = torch.cat([h_belief, h_intention], dim=1)       # (B, 2k, d)
        z, weights = self.attn(q, kv, kv)                    # z: (B, 1, d)
        return self.norm(z.squeeze(1)), weights              # (B, d), (B, 1, 2k)


# ── Auxiliary alignment loss for Stage 1 ────────────────────────────────────

def compute_alignment_loss(
    h_pred: Tensor,
    gt_pooled: Tensor,
    temperature: float = 0.07,
) -> Tensor:
    """
    InfoNCE alignment loss between predicted hypothesis embeddings and
    ground-truth cognitive state embeddings (from opponent CoT).

    Intuition: the BEST hypothesis among the k candidates should be
    close to the GT embedding. Negatives are GT embeddings from other
    samples in the batch (in-batch negatives, cheap and effective).

    Args:
        h_pred:    (B, k, d) — predicted hypothesis vectors (belief or intention)
        gt_pooled: (B, d)    — LLM-encoded GT text (B_t^GT or I_t^GT), stop-gradient
        temperature: InfoNCE temperature τ
    Returns:
        scalar loss
    """
    B = h_pred.size(0)

    # Normalize
    gt_norm = F.normalize(gt_pooled, dim=-1)           # (B, d)
    h_norm = F.normalize(h_pred, dim=-1)               # (B, k, d)

    # For each sample, pick the hypothesis most similar to GT (max-over-k)
    # sim[b, j] = cosine(gt_b, hyp_{b,j})
    sims_to_own_gt = torch.einsum("bd,bkd->bk", gt_norm, h_norm)   # (B, k)
    best_hyp_norm = h_norm[
        torch.arange(B, device=h_pred.device),
        sims_to_own_gt.argmax(dim=1)
    ]                                                                # (B, d)

    # Cross-batch similarity matrix: best hypothesis i vs all GTs j
    logits = torch.matmul(best_hyp_norm, gt_norm.T) / temperature   # (B, B)
    labels = torch.arange(B, device=h_pred.device)
    return F.cross_entropy(logits, labels)


def encode_gt_text(
    base_model_fn,        # callable: (input_ids, attention_mask) → last_hidden_state
    gt_input_ids: Tensor,
    gt_attention_mask: Tensor,
) -> Tensor:
    """
    Mean-pool the base LLM's hidden states over the GT text tokens.
    Called with torch.no_grad() to obtain stop-gradient GT embeddings.

    Returns: (B, d)
    """
    with torch.no_grad():
        hidden = base_model_fn(gt_input_ids, gt_attention_mask)     # (B, seq, d)
        mask_exp = gt_attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1)
    return pooled                                                    # (B, d)

"""
LASHModelConfig: hyperparameters for the LASH PyTorch architecture.

Separate from lash.LASHConfig (which configures the API-based simulation env).
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class LASHModelConfig:
    # ── Base LLM ─────────────────────────────────────────────────────────
    base_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"

    # ── LASH hypothesis module ────────────────────────────────────────────
    k: int = 4            # number of latent hypothesis vectors (H^B_t, H^I_t each: k × d)
    d_model: int = 4096   # hidden dim — must match base LLM (Llama 3 8B = 4096)
    n_heads: int = 8      # heads in HypothesisAttention cross-attention

    # ── Stage 1 auxiliary loss weights ───────────────────────────────────
    # L_total = L_lm + alpha * L_belief + beta * L_intention
    alpha_belief: float = 0.5
    beta_intention: float = 0.5

    # Oracle-to-Self curriculum mixing ratio (1.0 = full oracle, 0.0 = full self)
    # Controlled externally during training; stored here for checkpointing.
    oracle_ratio: float = 1.0

    # ── LoRA (PEFT) ───────────────────────────────────────────────────────
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    lora_bias: str = "none"

    # ── Misc ──────────────────────────────────────────────────────────────
    contrastive_temperature: float = 0.07   # for InfoNCE belief/intention alignment loss
    pad_token_id: int = 0

"""
Configuration for the LASH negotiation framework.

Supports any OpenAI-compatible API endpoint:
  - OpenAI (gpt-4o, gpt-4o-mini)
  - Local vLLM serving Llama 3.x  →  api_base="http://localhost:8000/v1", api_key="token"
  - Ollama                         →  api_base="http://localhost:11434/v1", api_key="ollama"
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LASHConfig:
    # ── LLM endpoint ──────────────────────────────────────────────
    model: str = "gpt-4o"
    temperature: float = 0.7
    api_base: Optional[str] = None   # None → use OpenAI default; set for local vLLM
    api_key: Optional[str] = None    # None → read from OPENAI_API_KEY env var

    # ── Episode ───────────────────────────────────────────────────
    max_rounds: int = 10

    # ── Domain ────────────────────────────────────────────────────
    item_description: str = "A used MacBook Pro (2022, M1, 16GB RAM, 512GB SSD)"

    # ── Reward design (GRPO target) ───────────────────────────────
    # R = lambda * surplus_captured + (1-lambda) * total_welfare + deal_bonus
    lambda_selfishness: float = 1.0   # 1.0 = fully selfish; 0.0 = cooperative
    deal_bonus: float = 0.0           # flat bonus for closing any deal

    # ── Data collection ───────────────────────────────────────────
    collect_cot: bool = True          # whether to store structured CoT for training

    # ── Ablation ──────────────────────────────────────────────────
    # True  (default): agents produce a natural-language <message> block that is
    #                  forwarded to the counterparty; action tag stays private.
    # False (ablation): no <message> block; the bare action tag is forwarded
    #                  (e.g. "[OFFER: $800]").  Parsing fallback in extract_message()
    #                  handles this automatically — only the prompt changes.
    natural_language_message: bool = True

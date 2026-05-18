"""
Data collection utilities for LASH Stage 1 training.

Two output formats:

1. episodes.jsonl  — full episode records (for analysis and replay)
   One JSON object per line, keyed by episode_id.

2. training_pairs.jsonl  — (c_t, B_t^GT, I_t^GT, M_t) tuples
   One JSON object per negotiation turn, structured for supervised learning:
     {
       "context":        c_t  (conversation history up to this turn),
       "belief_gt":      B_t^GT (ground-truth belief CoT),
       "intention_gt":   I_t^GT (ground-truth intention CoT),
       "message":        M_t  (visible action output),
       "action":         "accept" | "reject" | "offer",
       "price":          float | null,
       "role":           "buyer" | "seller",
       "round":          int,
       "episode_id":     str,
       "deal_reached":   bool,
       "buyer_reward":   float,   # for RL weighting
       "seller_reward":  float
     }
"""

import json
import threading
from pathlib import Path
from typing import Optional

from .types import EpisodeData, TurnData

# Protects concurrent file writes when running parallel episode collection
_write_lock = threading.Lock()


# ── Episode serialization ──────────────────────────────────────────────────

def _episode_to_dict(ep: EpisodeData) -> dict:
    return {
        "episode_id": ep.episode_id,
        "buyer_reservation": ep.buyer_type.reservation_price if ep.buyer_type else None,
        "buyer_delta": ep.buyer_type.delta if ep.buyer_type else None,
        "seller_reservation": ep.seller_type.reservation_price if ep.seller_type else None,
        "seller_delta": ep.seller_type.delta if ep.seller_type else None,
        "deal_reached": ep.deal_reached,
        "deal_price": ep.deal_price,
        "deal_round": ep.deal_round,
        "termination": ep.termination,
        "buyer_surplus": ep.buyer_surplus,
        "seller_surplus": ep.seller_surplus,
        "total_welfare": ep.total_welfare,
        "buyer_reward": ep.buyer_reward,
        "seller_reward": ep.seller_reward,
        "turns": [_turn_to_dict(t) for t in ep.turns],
    }


def _turn_to_dict(t: TurnData) -> dict:
    return {
        "turn": t.turn,
        "round": t.round,
        "role": t.role,
        "context": t.context,
        "raw_cot": t.raw_cot,
        "belief_gt": t.belief_text,
        "intention_gt": t.intention_text,
        "message": t.visible_message,
        "action": t.action,
        "price": t.price,
    }


# ── Training pair extraction ───────────────────────────────────────────────

def _turn_to_training_pair(t: TurnData, ep: EpisodeData) -> dict:
    """
    Convert a single turn into a training pair for Stage 1 supervision.
    Skips turns with empty belief_gt or intention_gt (model failed to follow format).
    """
    return {
        "context": t.context,
        "belief_gt": t.belief_text,
        "intention_gt": t.intention_text,
        "message": t.visible_message,
        "action": t.action,
        "price": t.price,
        "role": t.role,
        "round": t.round,
        "episode_id": ep.episode_id,
        "deal_reached": ep.deal_reached,
        "buyer_reward": ep.buyer_reward,
        "seller_reward": ep.seller_reward,
    }


def is_valid_training_pair(pair: dict) -> bool:
    """Filter out turns where the model didn't produce structured CoT."""
    return bool(pair["belief_gt"]) and bool(pair["intention_gt"])


# ── I/O ────────────────────────────────────────────────────────────────────

def append_episode(ep: EpisodeData, output_dir: str) -> None:
    """Append one episode to episodes.jsonl (thread-safe)."""
    path = Path(output_dir) / "episodes.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with _write_lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_episode_to_dict(ep), ensure_ascii=False) + "\n")


def append_training_pairs(ep: EpisodeData, output_dir: str) -> int:
    """
    Append valid (context, belief_gt, intention_gt, message) tuples to training_pairs.jsonl.
    Returns number of pairs written (thread-safe).
    """
    path = Path(output_dir) / "training_pairs.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for t in ep.turns:
        pair = _turn_to_training_pair(t, ep)
        if is_valid_training_pair(pair):
            lines.append(json.dumps(pair, ensure_ascii=False) + "\n")
    with _write_lock:
        with path.open("a", encoding="utf-8") as f:
            f.writelines(lines)
    return len(lines)


def save_episode(ep: EpisodeData, output_dir: str) -> int:
    """
    Persist a completed episode: full record + training pairs.
    Returns number of training pairs written.
    """
    append_episode(ep, output_dir)
    return append_training_pairs(ep, output_dir)


def load_training_pairs(data_dir: str) -> list[dict]:
    """Load all training pairs from training_pairs.jsonl."""
    path = Path(data_dir) / "training_pairs.jsonl"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def collection_stats(data_dir: str) -> dict:
    """Return summary statistics for collected data."""
    episodes_path = Path(data_dir) / "episodes.jsonl"
    pairs_path = Path(data_dir) / "training_pairs.jsonl"

    n_episodes = 0
    n_deals = 0
    total_welfare = 0.0
    if episodes_path.exists():
        with episodes_path.open("r", encoding="utf-8") as f:
            for line in f:
                ep = json.loads(line)
                n_episodes += 1
                if ep["deal_reached"]:
                    n_deals += 1
                    total_welfare += ep["total_welfare"]

    n_pairs = 0
    if pairs_path.exists():
        with pairs_path.open("r", encoding="utf-8") as f:
            n_pairs = sum(1 for line in f if line.strip())

    return {
        "episodes": n_episodes,
        "deals": n_deals,
        "deal_rate": n_deals / n_episodes if n_episodes else 0.0,
        "avg_welfare": total_welfare / n_deals if n_deals else 0.0,
        "training_pairs": n_pairs,
    }

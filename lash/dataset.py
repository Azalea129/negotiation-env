"""
PyTorch Dataset for LASH Stage 1 SFT training.

Each sample comes from training_pairs.jsonl:
  context (c_t), message (M_t), belief_gt (B_t^GT), intention_gt (I_t^GT)

Produces tokenized tensors for LASHModel.forward():
  input_ids / attention_mask / labels         → Pass 2 LM loss (L_lm)
  belief_input_ids  / belief_attention_mask   → InfoNCE belief alignment (L_belief)
  intention_input_ids / intention_attention_mask → InfoNCE intention alignment (L_intention)

input_ids = tokenize([c_t; M_t])
labels    = [-100 × len(c_t)] + tokenize(M_t)   (context masked, message supervised)
"""

from __future__ import annotations

from functools import partial
from typing import Any

import torch
from torch.utils.data import Dataset

from .data_collector import load_training_pairs


class LASHDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        tokenizer,
        max_context_len: int = 512,
        max_gen_len: int = 256,
        max_gt_len: int = 256,
    ):
        self.samples = load_training_pairs(data_dir)
        self.tok = tokenizer
        self.max_context_len = max_context_len
        self.max_gen_len = max_gen_len
        self.max_gt_len = max_gt_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self.samples[idx]

        # Context: tokenize with BOS; truncate from the left to keep recent turns
        ctx_ids = self._tok_ids(s["context"], self.max_context_len, add_special=True)
        # Message: no BOS (continuation of context)
        msg_ids = self._tok_ids(s["message"], self.max_gen_len, add_special=False)

        input_ids = ctx_ids + msg_ids
        labels = [-100] * len(ctx_ids) + list(msg_ids)
        attention_mask = [1] * len(input_ids)

        belief_ids, belief_mask = self._tok_pair(s["belief_gt"])
        intention_ids, intention_mask = self._tok_pair(s["intention_gt"])

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "belief_input_ids": belief_ids,
            "belief_attention_mask": belief_mask,
            "intention_input_ids": intention_ids,
            "intention_attention_mask": intention_mask,
            "buyer_reward": torch.tensor(s.get("buyer_reward", 0.0), dtype=torch.float),
            "seller_reward": torch.tensor(s.get("seller_reward", 0.0), dtype=torch.float),
        }

    def _tok_ids(self, text: str, max_len: int, add_special: bool) -> list[int]:
        enc = self.tok(
            text,
            max_length=max_len,
            truncation=True,
            add_special_tokens=add_special,
        )
        return enc["input_ids"]

    def _tok_pair(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self.tok(
            text,
            max_length=self.max_gt_len,
            truncation=True,
            padding="max_length",
            add_special_tokens=True,
            return_tensors="pt",
        )
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)


def collate_fn(batch: list[dict], pad_token_id: int = 0) -> dict[str, torch.Tensor]:
    """Right-pad variable-length input_ids/labels/attention_mask to batch max length."""

    def pad_seq(seqs: list[torch.Tensor], pad_val: int) -> torch.Tensor:
        max_len = max(s.size(0) for s in seqs)
        out = torch.full((len(seqs), max_len), pad_val, dtype=seqs[0].dtype)
        for i, s in enumerate(seqs):
            out[i, : s.size(0)] = s
        return out

    return {
        "input_ids": pad_seq([b["input_ids"] for b in batch], pad_token_id),
        "attention_mask": pad_seq([b["attention_mask"] for b in batch], 0),
        "labels": pad_seq([b["labels"] for b in batch], -100),
        "belief_input_ids": torch.stack([b["belief_input_ids"] for b in batch]),
        "belief_attention_mask": torch.stack([b["belief_attention_mask"] for b in batch]),
        "intention_input_ids": torch.stack([b["intention_input_ids"] for b in batch]),
        "intention_attention_mask": torch.stack([b["intention_attention_mask"] for b in batch]),
        "buyer_reward": torch.stack([b["buyer_reward"] for b in batch]),
        "seller_reward": torch.stack([b["seller_reward"] for b in batch]),
    }

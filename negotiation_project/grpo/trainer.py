"""
GRPO 정책 업데이트 트레이너.

목적함수:
  L^GRPO(θ) = E[(1/G) Σ min(ratio_i · Â_i, clip(ratio_i, 1-ε, 1+ε) · Â_i)]
               - β · KL(π_θ || π_ref)

- 구매자 턴: buyer_advantage로 업데이트
- 판매자 턴: seller_advantage로 업데이트
- 단일 정책이 두 역할을 모두 담당 (self-play)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .sampler import EpisodeRollout, GroupRollout, TurnRecord


class GRPONegotiationTrainer:
    def __init__(
        self,
        model: PreTrainedModel,
        ref_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        lr: float = 1e-5,
        clip_eps: float = 0.2,
        kl_beta: float = 0.01,
        max_grad_norm: float = 1.0,
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.clip_eps = clip_eps
        self.kl_beta = kl_beta
        self.max_grad_norm = max_grad_norm
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr
        )

    # ------------------------------------------------------------------
    # 공개 인터페이스
    # ------------------------------------------------------------------

    def step(self, group_rollout: GroupRollout) -> dict:
        """그룹 롤아웃 하나에 대해 GRPO 업데이트 1스텝을 수행한다."""
        self.optimizer.zero_grad()
        loss, metrics = self._compute_group_loss(group_rollout)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm
        )
        self.optimizer.step()
        metrics["grad_norm"] = grad_norm.item()
        return metrics

    # ------------------------------------------------------------------
    # 내부 로직
    # ------------------------------------------------------------------

    def _compute_group_loss(self, group_rollout: GroupRollout) -> tuple[torch.Tensor, dict]:
        """
        G개 에피소드 전체에 대한 GRPO loss를 계산한다.
        buyer/seller 턴을 모두 포함한다.
        """
        total_loss = torch.zeros(1, device=self._device(), requires_grad=False)
        policy_loss_sum = 0.0
        kl_sum = 0.0
        n_turns = 0

        for ep, b_adv, s_adv in zip(
            group_rollout.episodes,
            group_rollout.buyer_advantages,
            group_rollout.seller_advantages,
        ):
            for turn in ep.buyer_turns:
                pl, kl = self._turn_losses(turn, b_adv)
                total_loss = total_loss + pl + self.kl_beta * kl
                policy_loss_sum += pl.item()
                kl_sum += kl.item()
                n_turns += 1

            for turn in ep.seller_turns:
                pl, kl = self._turn_losses(turn, s_adv)
                total_loss = total_loss + pl + self.kl_beta * kl
                policy_loss_sum += pl.item()
                kl_sum += kl.item()
                n_turns += 1

        if n_turns == 0:
            return total_loss.squeeze(), {"loss": 0.0, "policy_loss": 0.0, "kl": 0.0, "n_turns": 0}

        avg_loss = total_loss / n_turns
        return avg_loss.squeeze(), {
            "loss": avg_loss.item(),
            "policy_loss": policy_loss_sum / n_turns,
            "kl": kl_sum / n_turns,
            "n_turns": n_turns,
        }

    def _turn_losses(self, turn: TurnRecord, advantage: float) -> tuple[torch.Tensor, torch.Tensor]:
        """
        단일 턴에 대해 clipped surrogate loss와 KL penalty를 계산한다.

        Returns:
            policy_loss: -E[min(ratio·Â, clip(ratio,1-ε,1+ε)·Â)]
            kl:          E[log π_θ - log π_ref]  (per-token 평균)
        """
        if not turn.completion_ids:
            zero = torch.zeros(1, device=self._device())
            return zero, zero

        prompt_ids = torch.tensor(turn.prompt_ids, device=self._device())
        comp_ids = torch.tensor(turn.completion_ids, device=self._device())
        old_log_probs = torch.tensor(turn.log_probs, device=self._device())

        full_ids = torch.cat([prompt_ids, comp_ids]).unsqueeze(0)  # (1, seq_len)
        prompt_len = len(turn.prompt_ids)

        # 현재 정책 log-probs
        curr_log_probs = self._completion_log_probs(self.model, full_ids, prompt_len)

        # 레퍼런스 모델 log-probs (그래디언트 불필요)
        with torch.no_grad():
            ref_log_probs = self._completion_log_probs(self.ref_model, full_ids, prompt_len)

        # 길이 불일치 방어 (생성 시와 forward 시 토큰 수가 다를 수 있음)
        min_len = min(len(curr_log_probs), len(old_log_probs), len(ref_log_probs))
        curr_log_probs = curr_log_probs[:min_len]
        old_log_probs = old_log_probs[:min_len]
        ref_log_probs = ref_log_probs[:min_len]

        # 중요도 비율
        log_ratio = curr_log_probs - old_log_probs
        ratio = torch.exp(log_ratio)

        adv = torch.tensor(advantage, dtype=torch.float32, device=self._device())

        # Clipped surrogate loss (PPO 스타일)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
        policy_loss = -torch.min(surr1, surr2).mean()

        # KL: E[log π_θ - log π_ref]
        kl = (curr_log_probs - ref_log_probs).mean()

        return policy_loss, kl

    @staticmethod
    def _completion_log_probs(
        model: PreTrainedModel,
        full_ids: torch.Tensor,
        prompt_len: int,
    ) -> torch.Tensor:
        """
        모델 forward pass를 통해 completion 토큰들의 per-token log-prob을 반환한다.
        full_ids: (1, prompt_len + comp_len)
        """
        comp_len = full_ids.shape[1] - prompt_len
        if comp_len <= 0:
            return torch.zeros(0, device=full_ids.device)

        logits = model(full_ids).logits  # (1, seq_len, vocab)

        # logit[prompt_len-1] → completion[0] 예측
        shift_logits = logits[0, prompt_len - 1: prompt_len - 1 + comp_len]  # (comp_len, vocab)
        shift_ids = full_ids[0, prompt_len: prompt_len + comp_len]           # (comp_len,)

        log_probs = F.log_softmax(shift_logits, dim=-1)
        return log_probs.gather(1, shift_ids.unsqueeze(-1)).squeeze(-1)

    def _device(self) -> torch.device:
        return next(self.model.parameters()).device

"""
LocalNegotiationAgent: HuggingFace 모델 기반 협상 에이전트.

OpenAI 기반 NegotiationAgent와 동일한 인터페이스를 제공하되,
GRPO 학습에 필요한 (prompt_ids, completion_ids, log_probs) 트래젝토리를 기록한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..agents import _CONDITION_INSTRUCTIONS, _extract_action, _filter_message
from ..config import NegotiationConfig


@dataclass
class TurnRecord:
    """GRPO 학습을 위한 단일 턴 데이터."""
    prompt_ids: list[int]       # 프롬프트 토큰 ids (컨텍스트)
    completion_ids: list[int]   # 생성된 응답 토큰 ids
    log_probs: list[float]      # completion 각 토큰의 log-prob (π_old 하에서)
    role: str                   # 'buyer' or 'seller'
    round_num: int


class LocalNegotiationAgent:
    """
    단일 HuggingFace 모델로 구동되는 협상 에이전트.

    - respond()는 NegotiationAgent와 동일한 딕셔너리를 반환한다.
    - trajectory 속성에 GRPO 업데이트에 필요한 TurnRecord 목록이 누적된다.
    """

    def __init__(
        self,
        role: str,
        system_prompt: str,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        config: NegotiationConfig,
        temperature: float = 0.8,
        max_new_tokens: int = 512,
    ):
        self.role = role
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]
        self.trajectory: list[TurnRecord] = []

    def respond(self, incoming_message: Optional[str], round_num: int = 0) -> dict:
        """
        incoming_message를 받아 응답을 생성하고, trajectory에 기록한다.

        Returns:
            raw_response, visible_message, action, price
        """
        if incoming_message is not None:
            self.messages.append({"role": "user", "content": incoming_message})

        # 채팅 템플릿 적용 → 프롬프트 텍스트
        prompt_text = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_tensor = torch.tensor([prompt_ids], device=self.model.device)

        # 응답 생성
        with torch.no_grad():
            output = self.model.generate(
                prompt_tensor,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        completion_ids = output[0][len(prompt_ids):].tolist()
        raw = self.tokenizer.decode(completion_ids, skip_special_tokens=True)

        # log-prob 계산: 생성된 전체 시퀀스에 대해 forward pass
        log_probs = self._compute_log_probs(
            full_ids=output[0],
            prompt_len=len(prompt_ids),
        )

        self.trajectory.append(TurnRecord(
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            log_probs=log_probs,
            role=self.role,
            round_num=round_num,
        ))

        self.messages.append({"role": "assistant", "content": raw})

        parsed = _extract_action(raw)
        visible = _filter_message(raw, self.config.condition)

        return {
            "raw_response": raw,
            "visible_message": visible,
            "action": parsed["action"],
            "price": parsed["price"],
        }

    def _compute_log_probs(self, full_ids: torch.Tensor, prompt_len: int) -> list[float]:
        """
        완전한 시퀀스에 대해 forward pass를 수행하여
        completion 토큰들의 per-token log-prob을 반환한다.
        """
        full_tensor = full_ids.unsqueeze(0)  # (1, seq_len)
        comp_len = full_tensor.shape[1] - prompt_len
        if comp_len <= 0:
            return []

        with torch.no_grad():
            logits = self.model(full_tensor).logits  # (1, seq_len, vocab)

        # position i의 logit → position i+1 토큰 예측
        # completion[0] = input_ids[prompt_len] → logit[prompt_len - 1]
        shift_logits = logits[0, prompt_len - 1: prompt_len - 1 + comp_len]  # (comp_len, vocab)
        shift_ids = full_tensor[0, prompt_len: prompt_len + comp_len]         # (comp_len,)

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(1, shift_ids.unsqueeze(-1)).squeeze(-1)

        return token_log_probs.tolist()

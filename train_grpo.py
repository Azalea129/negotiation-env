"""
Self-play GRPO 학습 스크립트 (RQ1).

단일 Llama-3.1-8B-Instruct 정책이 구매자와 판매자 역할을 모두 수행하며
협상 보상을 극대화하도록 LoRA fine-tuning된다.

사용법:
  python train_grpo.py \\
    --model_id meta-llama/Meta-Llama-3.1-8B-Instruct \\
    --n_steps 500 \\
    --group_size 8 \\
    --output_dir checkpoints/grpo_run1

주요 하이퍼파라미터:
  --group_size    G: 동일 설정으로 실행할 에피소드 수 (GRPO 그룹 크기)
  --lambda_s      λ: 이기심 파라미터 (0=협력, 1=이기적)
  --clip_eps      ε: PPO 클리핑 계수
  --kl_beta       β: KL 패널티 계수
  --lora_r        LoRA rank
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from negotiation_project import Condition, NegotiationConfig
from negotiation_project.types import sample_types
from negotiation_project.grpo.sampler import rollout_group
from negotiation_project.grpo.trainer import GRPONegotiationTrainer


def parse_args():
    p = argparse.ArgumentParser(description="Self-play GRPO for negotiation")
    p.add_argument("--model_id", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--n_steps", type=int, default=500)
    p.add_argument("--group_size", type=int, default=8,
                   help="G: GRPO 그룹 크기 (스텝당 에피소드 수)")
    p.add_argument("--max_rounds", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max_new_tokens", type=int, default=512)

    # 보상 파라미터
    p.add_argument("--lambda_s", type=float, default=0.5,
                   help="λ: selfishness (0=협력, 1=이기적)")
    p.add_argument("--condition", default="free_form",
                   choices=["free_form", "numeric_only", "templated", "cot_only"])

    # GRPO 하이퍼파라미터
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--kl_beta", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # 양자화 (T4/16GB 환경에서는 --use_4bit 권장, A100 40GB는 기본 bfloat16)
    p.add_argument("--use_4bit", action="store_true",
                   help="QLoRA: 4-bit 양자화로 메모리 절약 (T4 16GB 환경)")

    # 저장
    p.add_argument("--output_dir", default="checkpoints/grpo_run")
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_model_and_tokenizer(
    model_id: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    use_4bit: bool = False,
):
    print(f"모델 로딩: {model_id}  ({'QLoRA 4-bit' if use_4bit else 'bfloat16'})")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_4bit:
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs = {"quantization_config": bnb_config, "device_map": "auto"}
    else:
        load_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}

    # 학습 대상 모델 (LoRA / QLoRA)
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    if use_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 레퍼런스 모델 (고정, KL 패널티용)
    print("레퍼런스 모델 로딩...")
    ref_model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()

    return model, ref_model, tokenizer


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 설정 저장
    config_dict = vars(args)
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    model, ref_model, tokenizer = load_model_and_tokenizer(
        args.model_id, args.lora_r, args.lora_alpha, args.lora_dropout,
        use_4bit=args.use_4bit,
    )

    condition = Condition(args.condition)
    neg_config = NegotiationConfig(
        max_rounds=args.max_rounds,
        condition=condition,
        lambda_selfishness=args.lambda_s,
    )

    trainer = GRPONegotiationTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        lr=args.lr,
        clip_eps=args.clip_eps,
        kl_beta=args.kl_beta,
        max_grad_norm=args.max_grad_norm,
    )

    log_history = []
    print(f"\n{'='*60}")
    print(f"GRPO 학습 시작: {args.n_steps} 스텝, 그룹 크기={args.group_size}")
    print(f"조건: {condition.value}, λ={args.lambda_s}")
    print(f"{'='*60}\n")

    for step in range(1, args.n_steps + 1):
        t0 = time.time()

        # 타입 샘플링 (스텝마다 다른 시드)
        seed = args.seed * 100000 + step
        buyer_type, seller_type = sample_types(seed=seed)

        # G개 에피소드 롤아웃
        group = rollout_group(
            buyer_type=buyer_type,
            seller_type=seller_type,
            config=neg_config,
            model=model,
            tokenizer=tokenizer,
            group_size=args.group_size,
            temperature=args.temperature,
        )

        # GRPO 업데이트
        metrics = trainer.step(group)

        # 로그 집계
        n_deals = sum(1 for ep in group.episodes if ep.deal_reached)
        avg_buyer_r = sum(ep.buyer_reward for ep in group.episodes) / len(group.episodes)
        avg_seller_r = sum(ep.seller_reward for ep in group.episodes) / len(group.episodes)
        elapsed = time.time() - t0

        log_entry = {
            "step": step,
            "loss": metrics["loss"],
            "policy_loss": metrics["policy_loss"],
            "kl": metrics["kl"],
            "grad_norm": metrics["grad_norm"],
            "n_turns": metrics["n_turns"],
            "n_deals": n_deals,
            "deal_rate": n_deals / args.group_size,
            "avg_buyer_reward": avg_buyer_r,
            "avg_seller_reward": avg_seller_r,
            "elapsed_sec": elapsed,
            "v_b": buyer_type.reservation_price,
            "v_s": seller_type.reservation_price,
        }
        log_history.append(log_entry)

        if step % args.log_every == 0:
            print(
                f"[{step:4d}/{args.n_steps}] "
                f"loss={metrics['loss']:.4f} | "
                f"policy={metrics['policy_loss']:.4f} | "
                f"kl={metrics['kl']:.4f} | "
                f"deals={n_deals}/{args.group_size} | "
                f"buyer_r={avg_buyer_r:.1f} | "
                f"seller_r={avg_seller_r:.1f} | "
                f"{elapsed:.1f}s"
            )

        # 체크포인트 저장
        if step % args.save_every == 0:
            ckpt_path = output_dir / f"step_{step:05d}"
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            # 로그 저장
            with open(output_dir / "training_log.jsonl", "a") as f:
                for entry in log_history[-args.save_every:]:
                    f.write(json.dumps(entry) + "\n")
            print(f"  → 체크포인트 저장: {ckpt_path}")

    # 최종 저장
    final_path = output_dir / "final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    with open(output_dir / "training_log.jsonl", "a") as f:
        remaining = log_history[-(len(log_history) % args.save_every or args.save_every):]
        for entry in remaining:
            f.write(json.dumps(entry) + "\n")
    print(f"\n학습 완료. 최종 모델 → {final_path}")


if __name__ == "__main__":
    main()

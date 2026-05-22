"""
LASH Stage 2 GRPO training.

Initializes from Stage 1 SFT checkpoint and trains the full model end-to-end
with policy gradient on A2A negotiation reward.

GRPO objective per turn:
  L = -min(r·A, clip(r, 1-ε, 1+ε)·A) + λ_kl · KL(π_θ || π_ref)

  r   = exp(log π_θ(M_t|c_t) − log π_old(M_t|c_t))   importance ratio
  A   = group-relative advantage (per-role normalization within each config group)
  KL  = log π_θ(M_t|c_t) − log π_ref(M_t|c_t)        vs Stage 1 reference
  π_ref = Stage 1 SFT model (frozen), protects ToM representations and base capabilities

Gradient flows end-to-end through Pass 1 (hypothesis extraction) + Pass 2 (generation).

Usage:
  python train_stage2.py --stage1-dir checkpoints/stage1/final

  python train_stage2.py \\
      --stage1-dir checkpoints/stage1/final \\
      --n-iterations 500 \\
      --n-configs 4 --G 8 \\
      --lambda-kl 0.01 --clip-eps 0.2 \\
      --output-dir checkpoints/stage2
"""

import argparse
import os
import statistics
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from lash.config import LASHConfig
from lash.grpo_env import GRPORolloutEnv, RolloutEpisode, RolloutTurn
from lash.types import sample_types
from model import LASHModel, LASHModelConfig


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LASH Stage 2 GRPO training")
    # Paths
    p.add_argument("--stage1-dir", required=True,
                   help="Path to Stage 1 final checkpoint directory (contains lash_modules.pt, lora_adapter/, train_state.pt)")
    p.add_argument("--output-dir", default="checkpoints/stage2")
    p.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct",
                   help="Base model name (must match Stage 1)")
    # GRPO
    p.add_argument("--n-iterations", type=int, default=500)
    p.add_argument("--n-configs", type=int, default=4,
                   help="Distinct buyer/seller type configs sampled per iteration")
    p.add_argument("--G", type=int, default=8,
                   help="Episode rollouts per config (group size for advantage normalization)")
    p.add_argument("--clip-eps", type=float, default=0.2,
                   help="PPO-style clipping range for importance ratio")
    p.add_argument("--lambda-kl", type=float, default=0.01,
                   help="KL penalty coefficient vs Stage 1 reference")
    p.add_argument("--n-grpo-steps", type=int, default=1,
                   help="Gradient updates per collected rollout batch")
    # Negotiation environment
    p.add_argument("--item-desc", default="a used laptop computer")
    p.add_argument("--max-rounds", type=int, default=6)
    p.add_argument("--lambda-selfishness", type=float, default=0.5)
    p.add_argument("--deal-bonus", type=float, default=1.0)
    p.add_argument("--max-ctx-len", type=int, default=768)
    p.add_argument("--max-gen-len", type=int, default=512)
    p.add_argument("--rollout-temp", type=float, default=0.8)
    # Optimization
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup-iters", type=int, default=20,
                   help="Linear LR warmup over first N iterations")
    # I/O
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--load-in-4bit", action="store_true",
                   help="QLoRA: 4-bit NF4 quantization (recommended for T4/16GB VRAM)")
    p.add_argument("--hf-repo", default=None,
                   help="If set, upload each saved checkpoint to this HF Hub repo (path_in_repo=stage2/iter_XXXXX). Protects against runtime termination.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── Model loading ──────────────────────────────────────────────────────────

def load_from_stage1(
    stage1_dir: str,
    base_model_name: str,
    device: str,
    load_in_4bit: bool = False,
) -> tuple[LASHModel, LASHModelConfig]:
    """
    Reconstruct LASHModel from a Stage 1 checkpoint directory.

    Checkpoint layout (written by train_stage1.py save_checkpoint):
      lash_modules.pt   — hypothesis modules (hyp_queries, mlp, attn, z_to_embed)
      lora_adapter/     — PEFT LoRA adapter weights
      train_state.pt    — model_cfg + optimizer state
    """
    import torch as _torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    ckpt = Path(stage1_dir)
    train_state = torch.load(ckpt / "train_state.pt", map_location="cpu", weights_only=False)
    model_cfg: LASHModelConfig = train_state["model_cfg"]
    model_cfg.base_model_name = base_model_name

    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=_torch.float16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name, quantization_config=bnb_config, device_map=device,
            token=os.environ.get("HF_TOKEN"),
        )
        base = prepare_model_for_kbit_training(base)
    else:
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype="auto", device_map=device,
            token=os.environ.get("HF_TOKEN"),
        )
    base = PeftModel.from_pretrained(base, str(ckpt / "lora_adapter"))

    model = LASHModel(base, model_cfg)
    lash_state = torch.load(ckpt / "lash_modules.pt", map_location=device, weights_only=False)
    # strict=False: lash_modules.pt has no "base.*" keys, rest of model is already loaded
    missing, unexpected = model.load_state_dict(lash_state, strict=False)
    assert not unexpected, f"Unexpected keys in lash_modules.pt: {unexpected}"

    return model.to(device), model_cfg


# ── Log prob computation ───────────────────────────────────────────────────

def compute_sum_log_prob(
    model: LASHModel,
    ctx_ids: torch.Tensor,  # (L_c,) — stored without batch dim
    ctx_mask: torch.Tensor, # (L_c,)
    msg_ids: torch.Tensor,  # (L_m,)
    device: str,
) -> torch.Tensor:
    """
    Compute Σ log π(msg_ids[t] | prefix_t) under `model`.

    Full 2-pass forward: Pass 1 extracts z_t from ctx, Pass 2 computes logits
    over [z_embed | ctx | msg] and indexes message token positions.

    Gradient flows through this function when called outside torch.no_grad().
    """
    ctx_b = ctx_ids.unsqueeze(0).to(device)                              # (1, L_c)
    mask_b = ctx_mask.unsqueeze(0).to(device)                            # (1, L_c)
    msg = msg_ids.to(device)                                              # (L_m,)

    full_ids = torch.cat([ctx_b, msg.unsqueeze(0)], dim=1)               # (1, L_c+L_m)
    full_mask = torch.ones(1, full_ids.size(1), dtype=torch.long, device=device)

    # Pass 1
    h_b, h_i, c_pooled = model.extract_hypotheses(ctx_b, mask_b)
    z_t, _ = model.hypothesis_attn(c_pooled, h_b, h_i)

    # Pass 2 — logits: (1, 1+L_c+L_m, vocab)
    _, logits = model.forward_generation(z_t, full_ids, full_mask, labels=None)

    ctx_len = ctx_ids.size(0)
    msg_len = msg.size(0)
    # logits[:, ctx_len : ctx_len+msg_len, :] predicts msg_ids[0..msg_len-1]
    msg_logits = logits[0, ctx_len: ctx_len + msg_len, :]                # (L_m, vocab)
    log_probs = F.log_softmax(msg_logits, dim=-1)                        # (L_m, vocab)
    return log_probs[torch.arange(msg_len, device=device), msg].sum()    # scalar


# ── Advantage computation ──────────────────────────────────────────────────

def assign_advantages(
    episodes: list[RolloutEpisode],
    eps: float = 1e-8,
) -> list[RolloutTurn]:
    """
    Group-relative advantage normalization, computed per role.

    Buyer turns use buyer_reward for normalization;
    seller turns use seller_reward — since the two agents have different reward
    scales (depending on deal split), mixing them would distort the gradient signal.
    """
    buyer_rewards = [ep.buyer_reward for ep in episodes]
    seller_rewards = [ep.seller_reward for ep in episodes]

    b_mean = statistics.mean(buyer_rewards)
    s_mean = statistics.mean(seller_rewards)
    b_std = (statistics.stdev(buyer_rewards) if len(buyer_rewards) > 1 else 1.0) + eps
    s_std = (statistics.stdev(seller_rewards) if len(seller_rewards) > 1 else 1.0) + eps

    all_turns: list[RolloutTurn] = []
    for ep in episodes:
        b_adv = (ep.buyer_reward - b_mean) / b_std
        s_adv = (ep.seller_reward - s_mean) / s_std
        for turn in ep.turns:
            turn.advantage = b_adv if turn.role == "buyer" else s_adv
            all_turns.append(turn)
    return all_turns


# ── GRPO loss ──────────────────────────────────────────────────────────────

def grpo_loss_for_turns(
    turns: list[RolloutTurn],
    model: LASHModel,
    ref_model: Optional[LASHModel],
    lambda_kl: float,
    clip_eps: float,
    device: str,
) -> torch.Tensor:
    """
    Accumulate GRPO loss over a list of turns (sequential, memory-efficient).

    L_turn = -min(r·A, clip(r, 1-ε, 1+ε)·A) + λ_kl · (log π_θ − log π_ref)

    The gradient flows end-to-end through Pass 1 + Pass 2 of model.
    ref_model is called under no_grad. If ref_model is None (memory-constrained
    setups), the KL term is dropped and only the clipped policy gradient is used.
    """
    total = torch.tensor(0.0, device=device)

    for turn in turns:
        # Current policy log prob — WITH gradient
        curr_lp = compute_sum_log_prob(model, turn.ctx_ids, turn.ctx_mask, turn.msg_ids, device)

        # Importance ratio vs rollout policy (π_old)
        ratio = torch.exp(curr_lp - turn.old_log_prob)
        ratio_clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        adv = torch.tensor(turn.advantage, dtype=torch.float32, device=device)

        policy_loss = -torch.min(ratio * adv, ratio_clipped * adv)

        if ref_model is not None and lambda_kl > 0:
            # Reference policy log prob — stop-gradient
            with torch.no_grad():
                ref_lp = compute_sum_log_prob(ref_model, turn.ctx_ids, turn.ctx_mask, turn.msg_ids, device)
            kl_penalty = curr_lp - ref_lp.detach()
            total = total + policy_loss + lambda_kl * kl_penalty
        else:
            total = total + policy_loss

    return total / max(len(turns), 1)


# ── Checkpoint ────────────────────────────────────────────────────────────

def save_checkpoint(
    model: LASHModel,
    model_cfg: LASHModelConfig,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    path: Path,
    hf_repo: Optional[str] = None,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    lash_state = {k: v for k, v in model.state_dict().items() if not k.startswith("base.")}
    torch.save(lash_state, path / "lash_modules.pt")
    model.base.save_pretrained(str(path / "lora_adapter"))
    torch.save(
        {"model_cfg": model_cfg, "optimizer": optimizer.state_dict(), "iteration": iteration},
        path / "train_state.pt",
    )
    print(f"  → checkpoint: {path}")

    if hf_repo:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=os.environ.get("HF_TOKEN"))
            api.upload_folder(
                folder_path=str(path),
                repo_id=hf_repo,
                path_in_repo=f"stage2/{path.name}",
                repo_type="model",
            )
            print(f"  → HF upload: {hf_repo}/stage2/{path.name}")
        except Exception as e:
            print(f"  ⚠ HF upload failed ({e}) — local checkpoint preserved at {path}")


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    # Tokenizer
    hf_token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Trainable policy + (optional) frozen reference
    print(f"Loading Stage 1 checkpoint: {args.stage1_dir}")
    model, model_cfg = load_from_stage1(args.stage1_dir, args.model, device, args.load_in_4bit)

    if args.lambda_kl > 0:
        ref_model, _ = load_from_stage1(args.stage1_dir, args.model, device, args.load_in_4bit)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)
    else:
        ref_model = None
        print("lambda_kl=0 → skipping ref_model load (saves ~6GB on small GPUs). "
              "KL anchor disabled; relying on clipped ratio for stability.")

    # Simulation config (used only for prompts + reward; model field unused in rollout)
    sim_cfg = LASHConfig(
        model=args.model,
        item_description=args.item_desc,
        max_rounds=args.max_rounds,
        lambda_selfishness=args.lambda_selfishness,
        deal_bonus=args.deal_bonus,
    )

    env = GRPORolloutEnv(
        model=model,
        tokenizer=tokenizer,
        sim_config=sim_cfg,
        max_ctx_len=args.max_ctx_len,
        max_gen_len=args.max_gen_len,
        temperature=args.rollout_temp,
        device=device,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    # Linear warmup then constant LR
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=max(1, args.warmup_iters),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Stage 2 GRPO — {args.n_iterations} iterations")
    print(f"  n_configs={args.n_configs}  G={args.G}  λ_kl={args.lambda_kl}  ε={args.clip_eps}")
    print(f"  Total rollouts/iter = {args.n_configs * args.G} episodes")

    # ── Training loop ──────────────────────────────────────────────────────
    for iteration in range(1, args.n_iterations + 1):

        # ── Rollout phase (no grad) ──────────────────────────────────────
        model.eval()
        all_episodes: list[RolloutEpisode] = []

        for cfg_idx in range(args.n_configs):
            seed = args.seed + iteration * 10_000 + cfg_idx
            buyer_type, seller_type = sample_types(seed=seed)
            for g in range(args.G):
                ep = env.run_episode(buyer_type, seller_type)
                all_episodes.append(ep)

        # Group-relative advantage (per role, over all n_configs*G episodes)
        all_turns = assign_advantages(all_episodes)

        n_eps = len(all_episodes)
        deal_rate = sum(ep.deal_reached for ep in all_episodes) / n_eps
        avg_buyer = sum(ep.buyer_reward for ep in all_episodes) / n_eps
        avg_seller = sum(ep.seller_reward for ep in all_episodes) / n_eps

        # ── GRPO update phase ────────────────────────────────────────────
        model.train()
        total_loss = 0.0

        for step in range(args.n_grpo_steps):
            optimizer.zero_grad()
            loss = grpo_loss_for_turns(
                all_turns, model, ref_model, args.lambda_kl, args.clip_eps, device
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        if iteration <= args.warmup_iters:
            scheduler.step()

        avg_loss = total_loss / args.n_grpo_steps

        if iteration % args.log_every == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"iter={iteration:>4}  loss={avg_loss:.4f}  "
                f"deal_rate={deal_rate:.2f}  "
                f"R_buyer={avg_buyer:.3f}  R_seller={avg_seller:.3f}  "
                f"turns={len(all_turns)}  lr={lr_now:.2e}"
            )

        if iteration % args.save_every == 0:
            save_checkpoint(model, model_cfg, optimizer, iteration,
                            output_dir / f"iter_{iteration:05d}", hf_repo=args.hf_repo)

    save_checkpoint(model, model_cfg, optimizer, args.n_iterations,
                    output_dir / "final", hf_repo=args.hf_repo)
    print(f"\nStage 2 training complete → {output_dir}/final")


if __name__ == "__main__":
    main()

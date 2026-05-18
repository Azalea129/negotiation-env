"""
LASH Stage 1 SFT training.

Loss:
  L_total = L_lm + α·L_belief + β·L_intention

  L_lm        — causal LM loss on M_t tokens (message supervised, context masked)
  L_belief    — InfoNCE alignment between H^B_t and encode(B_t^GT)
  L_intention — InfoNCE alignment between H^I_t and encode(I_t^GT)

Oracle-to-Self curriculum:
  oracle_ratio decays linearly 1.0 → 0.0 over --oracle-decay-steps optimizer steps.
  While oracle_ratio > 0, z_t is blended toward oracle_z = proj(mean(B_t^GT, I_t^GT))
  to bootstrap generation with GT cognitive state before the latent space matures.

Checkpointing:
  Saves LASH-specific parameters (lash_modules.pt) and LoRA adapters (lora_adapter/)
  separately so Stage 2 can load them independently.

Usage:
  python train_stage1.py --data-dir data --output-dir checkpoints/stage1

  python train_stage1.py \\
      --data-dir data \\
      --model meta-llama/Meta-Llama-3-8B-Instruct \\
      --batch-size 4 --grad-accum 8 \\
      --lr 2e-4 --epochs 3 \\
      --oracle-decay-steps 2000 \\
      --output-dir checkpoints/stage1
"""

import argparse
import math
import os
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from lash.dataset import LASHDataset, collate_fn
from model import LASHModel, LASHModelConfig, load_lash_model
from model.modules import encode_gt_text


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LASH Stage 1 SFT training")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output-dir", default="checkpoints/stage1")
    p.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    # Data
    p.add_argument("--max-context-len", type=int, default=512)
    p.add_argument("--max-gen-len", type=int, default=256)
    p.add_argument("--max-gt-len", type=int, default=256)
    # Training
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8,
                   help="Effective batch = batch_size × grad_accum")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--warmup-steps", type=int, default=100)
    # Loss weights
    p.add_argument("--alpha-belief", type=float, default=0.5)
    p.add_argument("--beta-intention", type=float, default=0.5)
    p.add_argument("--contrastive-temp", type=float, default=0.07)
    # Oracle-to-Self
    p.add_argument("--oracle-decay-steps", type=int, default=2000,
                   help="Optimizer steps to decay oracle_ratio 1.0 → 0.0")
    # LoRA
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--load-in-4bit", action="store_true",
                   help="QLoRA: 4-bit NF4 quantization (recommended for T4/16GB VRAM)")
    # I/O
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── Oracle z computation ───────────────────────────────────────────────────

def compute_oracle_z(
    model: LASHModel,
    belief_input_ids: torch.Tensor,
    belief_attention_mask: torch.Tensor,
    intention_input_ids: torch.Tensor,
    intention_attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    oracle_z = z_to_embed(mean(encode(B_t^GT), encode(I_t^GT)))

    encode() is stop-gradient (no_grad in encode_gt_text), so oracle_z does not
    backprop through the backbone encoding — only through z_to_embed.
    """
    def backbone_fn(ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return model._backbone_hidden(model._embed(ids), mask)

    gt_belief = encode_gt_text(backbone_fn, belief_input_ids, belief_attention_mask)
    gt_intention = encode_gt_text(backbone_fn, intention_input_ids, intention_attention_mask)
    return model.z_to_embed((gt_belief + gt_intention) / 2)   # (B, d)


# ── Scheduler with linear warmup ──────────────────────────────────────────

def _lr_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


# ── Checkpoint helpers ────────────────────────────────────────────────────

def save_checkpoint(
    model: LASHModel,
    model_cfg: LASHModelConfig,
    optimizer: torch.optim.Optimizer,
    scheduler,
    oracle_ratio: float,
    optimizer_step: int,
    path: Path,
) -> None:
    path.mkdir(parents=True, exist_ok=True)

    # LASH-specific parameters (hypothesis modules, not part of LoRA)
    lash_state = {k: v for k, v in model.state_dict().items() if not k.startswith("base.")}
    torch.save(lash_state, path / "lash_modules.pt")

    # LoRA adapters via PEFT
    model.base.save_pretrained(path / "lora_adapter")

    # Training state
    model_cfg.oracle_ratio = oracle_ratio
    torch.save(
        {
            "model_cfg": model_cfg,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "optimizer_step": optimizer_step,
            "oracle_ratio": oracle_ratio,
        },
        path / "train_state.pt",
    )
    print(f"  → checkpoint: {path}")


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

    # Dataset
    dataset = LASHDataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        max_context_len=args.max_context_len,
        max_gen_len=args.max_gen_len,
        max_gt_len=args.max_gt_len,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id),
        num_workers=0,
        pin_memory=(device == "cuda"),
    )
    print(f"Dataset : {len(dataset)} samples")
    print(f"Batches : {len(loader)}/epoch  |  effective batch = {args.batch_size * args.grad_accum}")

    # Model
    model_cfg = LASHModelConfig(
        base_model_name=args.model,
        alpha_belief=args.alpha_belief,
        beta_intention=args.beta_intention,
        contrastive_temperature=args.contrastive_temp,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = load_lash_model(model_cfg, device=device, load_in_4bit=args.load_in_4bit)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_opt_steps = math.ceil(len(loader) / args.grad_accum) * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=partial(_lr_lambda, warmup=args.warmup_steps, total=total_opt_steps),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model   : {args.model}")
    print(f"Output  : {output_dir}/")
    print(f"Opt steps: {total_opt_steps}  |  oracle decay over {args.oracle_decay_steps}")

    # Training loop
    global_step = 0   # gradient accumulation steps
    optimizer_step = 0
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        model.train()

        for batch in loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Oracle-to-Self: linear decay
            oracle_ratio = max(0.0, 1.0 - optimizer_step / max(1, args.oracle_decay_steps))

            oracle_z = None
            if oracle_ratio > 0.0:
                oracle_z = compute_oracle_z(
                    model,
                    batch["belief_input_ids"],
                    batch["belief_attention_mask"],
                    batch["intention_input_ids"],
                    batch["intention_attention_mask"],
                )

            losses = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                belief_input_ids=batch["belief_input_ids"],
                belief_attention_mask=batch["belief_attention_mask"],
                intention_input_ids=batch["intention_input_ids"],
                intention_attention_mask=batch["intention_attention_mask"],
                oracle_ratio=oracle_ratio,
                oracle_z=oracle_z,
            )

            (losses["total_loss"] / args.grad_accum).backward()
            global_step += 1

            if global_step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optimizer_step += 1

                if optimizer_step % args.log_every == 0:
                    lm = losses.get("lm_loss", torch.tensor(0.0)).item()
                    bl = losses.get("belief_loss", torch.tensor(0.0)).item()
                    il = losses.get("intention_loss", torch.tensor(0.0)).item()
                    tot = losses["total_loss"].item()
                    lr_now = scheduler.get_last_lr()[0]
                    print(
                        f"epoch={epoch+1} step={optimizer_step:>5}  "
                        f"total={tot:.4f}  lm={lm:.4f}  "
                        f"belief={bl:.4f}  intention={il:.4f}  "
                        f"oracle={oracle_ratio:.3f}  lr={lr_now:.2e}"
                    )

                if optimizer_step % args.save_every == 0:
                    save_checkpoint(
                        model, model_cfg, optimizer, scheduler,
                        oracle_ratio, optimizer_step,
                        output_dir / f"step_{optimizer_step:06d}",
                    )

        print(f"── Epoch {epoch + 1}/{args.epochs} complete ──")

    # Final checkpoint
    save_checkpoint(
        model, model_cfg, optimizer, scheduler,
        oracle_ratio=0.0, optimizer_step=optimizer_step,
        path=output_dir / "final",
    )
    print(f"\nStage 1 training complete → {output_dir}/final")


if __name__ == "__main__":
    main()

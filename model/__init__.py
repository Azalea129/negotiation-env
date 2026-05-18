from .config import LASHModelConfig
from .modules import HypothesisAttention, IntentionMLP, compute_alignment_loss
from .lash_model import LASHModel


def load_lash_model(
    config: LASHModelConfig,
    device: str = "cuda",
    load_in_4bit: bool = False,
) -> LASHModel:
    """
    Load base Llama model with LoRA adapters and wrap with LASH modules.

    load_in_4bit=True uses QLoRA (NF4 quantization + bitsandbytes).
    Recommended for Colab T4 (16 GB VRAM): reduces base model footprint
    from ~16 GB fp16 to ~5 GB, leaving room for LoRA + LASH modules.

    Requires: transformers, peft, bitsandbytes (for 4-bit)
    """
    import torch
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            config.base_model_name,
            quantization_config=bnb_config,
            device_map=device,
        )
        base = prepare_model_for_kbit_training(base)
    else:
        base = AutoModelForCausalLM.from_pretrained(
            config.base_model_name,
            torch_dtype="auto",
            device_map=device,
        )

    lora_cfg = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias=config.lora_bias,
        task_type="CAUSAL_LM",
    )
    base = get_peft_model(base, lora_cfg)
    base.print_trainable_parameters()

    return LASHModel(base, config).to(device)

#!/usr/bin/env python3
"""Run inference with an EcomRLVE-GYM GRPO LoRA checkpoint.

Examples:
    python scripts/infer_openenv.py \
        --model unsloth/qwen3-1.7b-unsloth-bnb-4bit \
        --adapter outputs/ecomrlve_grpo_c4/final \
        --collection C4 \
        --env_id PD
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch

# Ensure ecom_rlve and sibling scripts are importable.
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from ecom_rlve.training.collections import COLLECTIONS
from train_openenv import EcomRLVEOpenEnv, SYSTEM_PROMPT, _extract_json_from_completion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ecomrlve.infer_openenv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inference test for EcomRLVE-GYM GRPO LoRA checkpoints",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="unsloth/qwen3-1.7b-unsloth-bnb-4bit",
        help="Base model name or path",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Optional LoRA adapter directory, e.g. outputs/ecomrlve_grpo_c4/final",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="C1",
        choices=sorted(COLLECTIONS.keys()),
        help="Environment collection to sample from",
    )
    parser.add_argument(
        "--env_id",
        type=str,
        default=None,
        help="Optional env id to force, e.g. PD, SUB, CART, RETURN",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Episode seed",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=4096,
        help="Model context length",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Maximum generated tokens",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Nucleus sampling top_p",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        default=True,
        help="Load base model in 4-bit quantization",
    )
    parser.add_argument(
        "--load_in_16bit",
        action="store_true",
        default=False,
        help="Load base model in 16-bit instead of 4-bit",
    )
    parser.add_argument(
        "--disable_thinking",
        action="store_true",
        help="Pass enable_thinking=False to chat template if supported",
    )
    parser.add_argument(
        "--no_score",
        action="store_true",
        help="Skip environment scoring and only print generation",
    )
    return parser.parse_args()


def render_prompt(
    tokenizer: Any,
    messages: list[dict[str, str]],
    disable_thinking: bool,
) -> str:
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if disable_thinking:
        kwargs["enable_thinking"] = False

    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def main() -> None:
    args = parse_args()

    from unsloth import FastLanguageModel

    load_in_4bit = args.load_in_4bit and not args.load_in_16bit
    use_bf16 = (
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
        and not load_in_4bit
    )
    model_dtype = torch.bfloat16 if use_bf16 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading model: %s", args.model)
    logger.info("Adapter: %s", args.adapter or "<none>")
    logger.info("Precision: dtype=%s, load_in_4bit=%s", model_dtype, load_in_4bit)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=model_dtype,
        load_in_4bit=load_in_4bit,
    )

    if args.adapter:
        from peft import PeftModel

        logger.info("Loading LoRA adapter: %s", args.adapter)
        model = PeftModel.from_pretrained(model, args.adapter)

    FastLanguageModel.for_inference(model)

    openenv = EcomRLVEOpenEnv(collection=args.collection, seed=args.seed)
    if args.env_id:
        if args.env_id not in openenv.env_ids:
            raise ValueError(
                f"env_id '{args.env_id}' is not in collection {args.collection}: "
                f"{openenv.env_ids}"
            )
        env_id = args.env_id
        episode_seed = args.seed
        obs = openenv.env.reset(env_id=env_id, seed=episode_seed)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend({"role": m["role"], "content": m["content"]} for m in obs.conversation)
    else:
        messages, env_id, episode_seed = openenv.sample_prompt(tokenizer)

    prompt = render_prompt(tokenizer, messages, args.disable_thinking)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    logger.info("Collection=%s env=%s seed=%s", args.collection, env_id, episode_seed)
    logger.info("Prompt tokens=%d", int(inputs["input_ids"].shape[-1]))

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0, inputs["input_ids"].shape[-1]:]
    completion = tokenizer.decode(new_tokens, skip_special_tokens=False)

    print("\n" + "=" * 80)
    print("PROMPT")
    print("=" * 80)
    for message in messages:
        print(f"[{message['role']}]\n{message['content']}\n")

    print("=" * 80)
    print("COMPLETION")
    print("=" * 80)
    print(completion)

    extracted = _extract_json_from_completion(completion)
    print("\n" + "=" * 80)
    print("EXTRACTED JSON")
    print("=" * 80)
    if extracted is None:
        print("<none>")
    else:
        try:
            print(json.dumps(json.loads(extracted), indent=2))
        except json.JSONDecodeError:
            print(extracted)

    if args.no_score:
        return

    print("\n" + "=" * 80)
    print("ENV SCORE")
    print("=" * 80)
    if extracted is None:
        print(json.dumps({"reward": -1.0, "reason": "no_valid_json"}, indent=2))
        return

    result = openenv.evaluate_completion(
        completion=extracted,
        env_id=env_id,
        episode_seed=episode_seed,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

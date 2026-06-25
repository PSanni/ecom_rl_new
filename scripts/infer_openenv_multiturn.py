#!/usr/bin/env python3
"""Chat-style streamed inference for multi-turn EcomRLVE checkpoints.

The script asks for customer messages in the terminal, streams model output,
executes emitted tool calls, and prints the final reward when finish is called.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any, Callable

import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(path, override=False)
        return
    except ImportError:
        pass

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file(_PROJECT_ROOT / ".env")

from ecom_rlve.training.collections import COLLECTIONS, get_collection
from train_openenv_multiturn import (
    EcomRLVEMultiTurnEnv,
    _FACTORY_CONFIG,
    _bool_arg,
    _system_prompt_for_env,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ecomrlve.infer_openenv_multiturn")


TOOL_METHODS_BY_ENV: dict[str, list[str]] = {
    "PD": [
        "catalog_search",
        "catalog_rerank",
        "catalog_get_product",
        "catalog_get_variants",
        "finish",
    ],
    "SUB": [
        "catalog_search",
        "catalog_rerank",
        "catalog_get_product",
        "catalog_get_variants",
        "finish",
    ],
    "CART": [
        "user_get_visit_history",
        "catalog_search",
        "catalog_get_product",
        "catalog_get_variants",
        "cart_add",
        "cart_view",
        "cart_remove",
        "cart_set_quantity",
        "finish",
    ],
    "RETURN": [
        "order_list",
        "order_get_status",
        "return_check_eligibility",
        "return_initiate",
        "return_exchange",
        "catalog_search",
        "catalog_get_product",
        "catalog_get_variants",
        "finish",
    ],
    "ORDER": ["order_list", "order_get_status", "finish"],
    "POLICY": ["policy_search", "datetime_now", "finish"],
    "BUNDLE": [
        "catalog_search",
        "catalog_get_product",
        "catalog_get_variants",
        "cart_add",
        "cart_view",
        "finish",
    ],
    "JOURNEY": [
        "user_get_visit_history",
        "catalog_search",
        "catalog_get_product",
        "catalog_get_variants",
        "cart_add",
        "cart_view",
        "order_checkout",
        "finish",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chat with a trained multi-turn EcomRLVE model and print reward at finish.",
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Optional LoRA adapter directory, e.g. outputs/.../final.",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="C1",
        choices=sorted(COLLECTIONS.keys()),
    )
    parser.add_argument("--env_id", type=str, default="CART")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--difficulty", type=int, default=None)
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--max_tool_rounds", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--load_in_16bit", action="store_true", default=False)
    parser.add_argument(
        "--precision",
        type=str,
        default="none",
        choices=["none", "fp16", "bf16"],
    )
    parser.add_argument(
        "--disable_thinking",
        action="store_true",
        help="Pass enable_thinking=False to the chat template when supported.",
    )
    parser.add_argument(
        "--chat_template_tools",
        type=_bool_arg,
        default=True,
        metavar="true|false",
        help="Pass Python tool definitions to apply_chat_template when supported.",
    )
    parser.add_argument("--embedding_debug", type=_bool_arg, default=None, metavar="true|false")
    parser.add_argument("--embedding_model", type=str, default=None)
    parser.add_argument("--embedding_device", type=str, default=None)
    parser.add_argument("--n_synthetic_products", type=int, default=None)
    parser.add_argument("--faiss_index_factory", type=str, default=None)
    parser.add_argument("--faiss_use_gpu", type=_bool_arg, default=None, metavar="true|false")
    parser.add_argument("--faiss_index_path", type=str, default=None)
    parser.add_argument("--debug_result_chars", type=int, default=2000)
    parser.add_argument(
        "--force_finish_on_max_rounds",
        type=_bool_arg,
        default=True,
        metavar="true|false",
        help="If the model never calls finish, use fallback scoring at the max-round guard.",
    )
    return parser.parse_args()


def _tool_functions(env: EcomRLVEMultiTurnEnv, env_id: str) -> list[Callable[..., Any]]:
    names = TOOL_METHODS_BY_ENV.get(env_id, TOOL_METHODS_BY_ENV["CART"])
    return [getattr(env, name) for name in names if hasattr(env, name)]


def _render_prompt(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[Callable[..., Any]],
    *,
    disable_thinking: bool,
    include_tools: bool,
) -> str:
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if disable_thinking:
        kwargs["enable_thinking"] = False
    if include_tools and tools:
        kwargs["tools"] = tools

    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        try:
            return tokenizer.apply_chat_template(messages, **kwargs)
        except Exception:
            kwargs.pop("tools", None)
            return tokenizer.apply_chat_template(_tool_messages_as_user(messages), **kwargs)
    except Exception:
        kwargs.pop("tools", None)
        return tokenizer.apply_chat_template(_tool_messages_as_user(messages), **kwargs)


def _tool_messages_as_user(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    converted: list[dict[str, str]] = []
    for message in messages:
        if message.get("role") == "tool":
            name = message.get("name", "tool")
            converted.append({
                "role": "user",
                "content": f"Tool result from {name}:\n{message.get('content', '')}",
            })
        else:
            converted.append({
                "role": str(message.get("role", "user")),
                "content": str(message.get("content", "")),
            })
    return converted


def _stream_generate(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[Callable[..., Any]],
    args: argparse.Namespace,
) -> str:
    from transformers import TextIteratorStreamer

    prompt = _render_prompt(
        tokenizer,
        messages,
        tools,
        disable_thinking=args.disable_thinking,
        include_tools=args.chat_template_tools,
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_seq_length,
    ).to(model.device)
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=False,
    )
    generate_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.min_p > 0:
        generate_kwargs["min_p"] = args.min_p

    thread = threading.Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()

    chunks: list[str] = []
    for text in streamer:
        print(text, end="", flush=True)
        chunks.append(text)
    thread.join()
    print("", flush=True)
    return "".join(chunks)


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _json_loads_maybe(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _parse_json_tool_calls(text: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    patterns = [
        r"<tool_call>\s*(.*?)\s*</tool_call>",
        r"```(?:json)?\s*(\{.*?\"name\".*?\})\s*```",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.DOTALL):
            raw = match.group(1).strip()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                name = payload.get("name") or payload.get("tool_name")
                args = payload.get("arguments", payload.get("args", {}))
                try:
                    args = _json_loads_maybe(args)
                except json.JSONDecodeError:
                    args = {}
                if isinstance(name, str) and isinstance(args, dict):
                    calls.append({"name": name, "args": args})
    return calls


def _find_balanced_call(text: str, start: int) -> tuple[str, int] | None:
    depth = 0
    quote: str | None = None
    escape = False
    for idx in range(start, len(text)):
        char = text[idx]
        if quote:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[start: idx + 1], idx + 1
    return None


def _parse_python_tool_calls(text: str, allowed_names: set[str]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    compact = _strip_thinking(text)
    pattern = re.compile(r"\b(" + "|".join(re.escape(name) for name in sorted(allowed_names)) + r")\s*\(")
    pos = 0
    while True:
        match = pattern.search(compact, pos)
        if not match:
            break
        balanced = _find_balanced_call(compact, match.start())
        if balanced is None:
            pos = match.end()
            continue
        call_text, pos = balanced
        try:
            parsed = ast.parse(call_text, mode="eval")
        except SyntaxError:
            continue
        if not isinstance(parsed.body, ast.Call) or not isinstance(parsed.body.func, ast.Name):
            continue
        name = parsed.body.func.id
        args: dict[str, Any] = {}
        for keyword in parsed.body.keywords:
            if keyword.arg is None:
                continue
            try:
                args[keyword.arg] = ast.literal_eval(keyword.value)
            except (ValueError, SyntaxError):
                args[keyword.arg] = None
        calls.append({"name": name, "args": args})
    return calls


def parse_tool_calls(text: str, allowed_names: set[str]) -> list[dict[str, Any]]:
    calls = _parse_json_tool_calls(text)
    if calls:
        return [call for call in calls if call["name"] in allowed_names]
    return _parse_python_tool_calls(text, allowed_names)


def _load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_in_4bit = args.load_in_4bit and not args.load_in_16bit
    use_bf16 = args.precision == "bf16"
    use_fp16 = args.precision == "fp16"
    if use_bf16 and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise ValueError("--precision bf16 requested, but this CUDA device does not support bf16.")
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    quantization_config = None
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    adapter_source = args.adapter
    if adapter_source:
        adapter_path = Path(adapter_source).expanduser()
        if adapter_path.exists():
            adapter_source = str(adapter_path.resolve())

    tokenizer_source = args.model
    if adapter_source:
        adapter_path = Path(adapter_source)
        tokenizer_files = {
            "tokenizer_config.json",
            "tokenizer.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "spiece.model",
            "sentencepiece.bpe.model",
        }
        if adapter_path.exists() and any((adapter_path / name).exists() for name in tokenizer_files):
            tokenizer_source = str(adapter_path)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype if (use_bf16 or use_fp16) and not load_in_4bit else None,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter_source:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_source)
    model.eval()
    model.config.use_cache = True
    return model, tokenizer


def _print_reward(env: EcomRLVEMultiTurnEnv) -> None:
    print("\n" + "=" * 80)
    print("FINAL REWARD")
    print("=" * 80)
    print(json.dumps({
        "reward": env.reward,
        "done": env.done,
        "is_correct": env.is_correct,
        "termination_reason": env.termination_reason,
        "finish_called_by_model": env.finish_called_by_model,
        "fallback_finished": env.fallback_finished,
        "reward_breakdown": env.reward_breakdown,
    }, indent=2, default=str))


def _read_customer_message() -> str:
    while True:
        try:
            text = input("\nCustomer> ").strip()
        except EOFError:
            return ""
        if text:
            return text


def _set_visible_customer_message(env: EcomRLVEMultiTurnEnv, message: str) -> None:
    state = env.env.get_episode_state()
    if state is None:
        return
    state.conversation = [{"role": "user", "content": message}]


def _run_assistant_turn(
    *,
    env: EcomRLVEMultiTurnEnv,
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[Callable[..., Any]],
    allowed_names: set[str],
    args: argparse.Namespace,
    round_index: int,
) -> bool:
    """Run one assistant generation.

    Returns True when at least one tool was called, which means the script
    should continue the model/tool loop without asking the customer again.
    """
    print("\nAssistant> ", end="", flush=True)
    completion = _stream_generate(model, tokenizer, messages, tools, args)
    messages.append({"role": "assistant", "content": completion})

    calls = parse_tool_calls(completion, allowed_names)
    if not calls:
        return False

    for call in calls:
        name = call["name"]
        kwargs = call["args"]
        print("\n" + "-" * 80)
        print(f"Tool call: {name}")
        print(json.dumps(kwargs, indent=2, default=str))
        try:
            result = getattr(env, name)(**kwargs)
        except Exception as exc:
            result = json.dumps({"error": f"{type(exc).__name__}: {exc}"})
        print("Tool result:")
        print(result)
        messages.append({"role": "tool", "name": name, "content": result})
        if env.done:
            return True

    return True


def main() -> None:
    args = parse_args()
    valid_env_ids = set(get_collection(args.collection))
    if args.env_id not in valid_env_ids:
        raise ValueError(
            f"--env_id {args.env_id!r} is not in --collection {args.collection}: "
            f"{sorted(valid_env_ids)}"
        )

    env_config = {
        "disclose_env_id": True,
        "disclose_difficulty": True,
    }
    optional_env_config = {
        "embedding_debug": args.embedding_debug,
        "embedding_model": args.embedding_model,
        "embedding_device": args.embedding_device,
        "n_synthetic_products": args.n_synthetic_products,
        "faiss_index_factory": args.faiss_index_factory,
        "faiss_use_gpu": args.faiss_use_gpu,
        "faiss_index_path": args.faiss_index_path,
    }
    env_config.update({key: value for key, value in optional_env_config.items() if value is not None})
    _FACTORY_CONFIG.update({
        "collection": args.collection,
        "seed": args.seed,
        "env_id": args.env_id,
        "difficulty": args.difficulty,
        "config": env_config,
        "debug_rollouts": 0,
        "debug_result_chars": args.debug_result_chars,
        "trace_rollouts_dir": "",
        "trace_rollouts_limit": 0,
        "fallback_finish_reward": 0.0,
    })

    logger.info("Loading model=%s adapter=%s", args.model, args.adapter or "<none>")
    model, tokenizer = _load_model_and_tokenizer(args)

    env = EcomRLVEMultiTurnEnv()
    env.reset(env_id=args.env_id, difficulty=args.difficulty, episode_seed=args.seed)
    tools = _tool_functions(env, args.env_id)
    allowed_names = {tool.__name__ for tool in tools}

    print("\n" + "=" * 80)
    print(f"CHAT env={env.env_id} seed={env.episode_seed}")
    print("=" * 80)
    print("Type the customer request. Press Ctrl-D to stop without scoring.")

    customer_message = _read_customer_message()
    if not customer_message:
        print("\nNo customer message received.")
        return

    _set_visible_customer_message(env, customer_message)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _system_prompt_for_env(args.env_id)},
        {
            "role": "user",
            "content": (
                f"Task env: {env.env_id}\n"
                f"Difficulty: {args.difficulty if args.difficulty is not None else 'sampled'}\n\n"
                f"Customer: {customer_message}"
            ),
        },
    ]

    round_index = 0
    while not env.done and round_index < args.max_tool_rounds:
        round_index += 1
        called_tool = _run_assistant_turn(
            env=env,
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            tools=tools,
            allowed_names=allowed_names,
            args=args,
            round_index=round_index,
        )
        if env.done:
            _print_reward(env)
            return
        if called_tool:
            continue

        customer_message = _read_customer_message()
        if not customer_message:
            break
        messages.append({"role": "user", "content": customer_message})

    if not env.done and args.force_finish_on_max_rounds:
        print("\nModel did not call finish before chat ended; applying fallback scoring.")
        env.ensure_finished()
    _print_reward(env)


if __name__ == "__main__":
    main()

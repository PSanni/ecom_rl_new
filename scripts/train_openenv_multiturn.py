#!/usr/bin/env python3
"""Multi-turn EcomRLVE-GYM training with TRL environment_factory.

This script is intentionally separate from train_openenv.py.  The older
script scores one generated JSON action at a time.  This one exposes the
EcomRLVE tools through TRL's environment_factory interface so GRPO can run
interactive multi-turn episodes before computing reward.

Example:
    python scripts/train_openenv_multiturn.py \
        --model Qwen/Qwen3-1.7B \
        --collection C4 \
        --n_prompts 1000 \
        --max_steps 300 \
        --num_generations 4 \
        --output_dir outputs/ecomrlve_multiturn_c4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from ecom_rlve.server.openenv import EcomRLVEEnv
from ecom_rlve.tools.registry import ToolCall
from ecom_rlve.training.collections import COLLECTIONS, get_collection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ecomrlve.train_openenv_multiturn")


SYSTEM_PROMPT = """\
You are a helpful e-commerce shopping assistant. Use the available tools to
complete the customer's task, then call finish when the task is complete.

Important:
- Use only product IDs, variant IDs, order IDs, and line IDs that came from tools.
- Never show internal IDs in user-facing text unless the user explicitly asks.
- For cart tasks, use user_get_visit_history first when the user refers to items
  they viewed or previously selected.
- For product discovery and substitution tasks, retrieve products before
  recommending them.
- For returns and order status, inspect orders before selecting an order/line.
- For policy questions, search policy before answering.

Available tools include:
- user_get_visit_history()
- catalog_search(query, filters_json, top_k)
- catalog_rerank(query, candidate_product_ids_json, top_k)
- catalog_get_product(product_id)
- catalog_get_variants(product_id)
- cart_add(product_id, variant_id, quantity)
- cart_remove(line_id)
- cart_set_quantity(line_id, quantity)
- cart_view()
- order_list(days)
- order_get_status(order_id)
- order_checkout(shipping_address_id, payment_method_id)
- return_check_eligibility(order_id, line_id)
- return_initiate(order_id, line_id, reason_code, method)
- return_exchange(order_id, line_id, new_product_id, new_variant_id)
- policy_search(query, top_k)
- datetime_now()
- finish(assistant_message, recommended_product_ids_json, selected_order_id,
         selected_line_id, policy_answer)
"""


_FACTORY_CONFIG: dict[str, Any] = {
    "collection": "C4",
    "seed": 42,
    "env_id": None,
    "difficulty": None,
    "config": {"disclose_env_id": True, "disclose_difficulty": True},
}


def _json_loads_or_default(value: str | None, default: Any) -> Any:
    if value is None or value == "":
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _trim_result(result: Any, limit: int = 6000) -> str:
    text = json.dumps(result, indent=2, default=str)
    if len(text) > limit:
        return text[:limit] + "\n... <truncated>"
    return text


class EcomRLVEMultiTurnEnv:
    """TRL environment_factory wrapper around EcomRLVEEnv.

    Public methods become callable tools for the model.  Tool methods execute
    against the underlying EcomRLVE tool registry without terminating the
    episode.  The finish method submits the final answer through env.step(),
    which lets the existing verifiers compute reward.
    """

    def __init__(self) -> None:
        self.env = EcomRLVEEnv(
            collection=_FACTORY_CONFIG["collection"],
            seed=int(_FACTORY_CONFIG["seed"]),
            config=_FACTORY_CONFIG["config"],
        )
        self.reward = 0.0
        self.done = False
        self.env_id = ""
        self.episode_seed = 0
        self.invalid_tool = False

    def reset(self, **kwargs: Any) -> str:
        """Start a fresh EcomRLVE episode and return the first user message."""
        self.reward = 0.0
        self.done = False
        self.invalid_tool = False

        env_id = kwargs.get("env_id") or _FACTORY_CONFIG.get("env_id")
        difficulty = kwargs.get("difficulty")
        if difficulty is None:
            difficulty = _FACTORY_CONFIG.get("difficulty")
        seed = kwargs.get("episode_seed")

        obs = self.env.reset(env_id=env_id, difficulty=difficulty, seed=seed)
        state = self.env.get_episode_state()
        self.env_id = state.env_id if state is not None else (env_id or "")
        self.episode_seed = state.seed if state is not None else int(seed or 0)

        user_message = obs.conversation[0]["content"] if obs.conversation else ""
        task_bits = [f"Task env: {self.env_id}"]
        if obs.difficulty is not None:
            task_bits.append(f"Difficulty: {obs.difficulty}")
        return "\n".join(task_bits) + f"\n\nCustomer: {user_message}"

    def _execute_tool(self, name: str, args: dict[str, Any]) -> str:
        if self.done:
            raise ValueError("Episode is already done.")
        state = self.env.get_episode_state()
        if state is None:
            raise ValueError("Environment has not been reset.")

        result = self.env._tool_registry.execute(ToolCall(name=name, args=args), state=state)
        entry = {
            "name": name,
            "args": args,
            "result": result.result,
            "error": result.error,
            "duration_ms": result.duration_ms,
        }
        state.tool_results_history.append(entry)
        if result.error is not None:
            self.invalid_tool = True
        else:
            self.env._extract_seen_ids(result.result, state.seen_product_ids)
        return _trim_result(entry)

    def user_get_visit_history(self) -> str:
        """Get the customer's recently viewed products.

        Returns:
            Recently viewed products with product IDs and attributes.
        """
        return self._execute_tool("user.get_visit_history", {})

    def catalog_search(self, query: str, filters_json: str = "", top_k: int = 10) -> str:
        """Search the product catalog.

        Args:
            query: Natural-language search query.
            filters_json: Optional JSON object string for filters.
            top_k: Number of results to return.

        Returns:
            Matching products.
        """
        filters = _json_loads_or_default(filters_json, None)
        return self._execute_tool(
            "catalog.search",
            {"query": query, "filters": filters, "top_k": top_k},
        )

    def catalog_rerank(
        self,
        query: str,
        candidate_product_ids_json: str,
        top_k: int = 10,
    ) -> str:
        """Rerank candidate products for a query.

        Args:
            query: User requirement or reranking query.
            candidate_product_ids_json: JSON list of candidate product IDs.
            top_k: Number of reranked products to return.

        Returns:
            Reranked products.
        """
        ids = _json_loads_or_default(candidate_product_ids_json, [])
        return self._execute_tool(
            "catalog.rerank",
            {"query": query, "candidate_product_ids": ids, "top_k": top_k},
        )

    def catalog_get_product(self, product_id: str) -> str:
        """Get full product details.

        Args:
            product_id: Product ID returned by a previous tool.

        Returns:
            Product details.
        """
        return self._execute_tool("catalog.get_product", {"product_id": product_id})

    def catalog_get_variants(self, product_id: str) -> str:
        """Get variants for a product.

        Args:
            product_id: Product ID returned by a previous tool.

        Returns:
            Available variants.
        """
        return self._execute_tool("catalog.get_variants", {"product_id": product_id})

    def cart_add(self, product_id: str, variant_id: str | None = None, quantity: int = 1) -> str:
        """Add a product to the cart.

        Args:
            product_id: Product ID returned by a previous tool.
            variant_id: Optional variant ID returned by catalog_get_variants.
            quantity: Quantity to add.

        Returns:
            Updated cart line.
        """
        return self._execute_tool(
            "cart.add",
            {"product_id": product_id, "variant_id": variant_id, "quantity": quantity},
        )

    def cart_remove(self, line_id: str) -> str:
        """Remove a cart line.

        Args:
            line_id: Cart line ID from cart_view.

        Returns:
            Updated cart information.
        """
        return self._execute_tool("cart.remove", {"line_id": line_id})

    def cart_set_quantity(self, line_id: str, quantity: int) -> str:
        """Set cart line quantity.

        Args:
            line_id: Cart line ID from cart_view.
            quantity: New quantity. Use 0 to remove the line.

        Returns:
            Updated cart information.
        """
        return self._execute_tool(
            "cart.set_quantity",
            {"line_id": line_id, "quantity": quantity},
        )

    def cart_view(self) -> str:
        """View the current cart.

        Returns:
            Current cart contents.
        """
        return self._execute_tool("cart.view", {})

    def order_list(self, days: int = 30) -> str:
        """List recent orders.

        Args:
            days: Lookback window in days.

        Returns:
            Recent orders.
        """
        return self._execute_tool("order.list", {"days": days})

    def order_get_status(self, order_id: str) -> str:
        """Get order status.

        Args:
            order_id: Order ID returned by order_list.

        Returns:
            Order status details.
        """
        return self._execute_tool("order.get_status", {"order_id": order_id})

    def order_checkout(self, shipping_address_id: str, payment_method_id: str) -> str:
        """Checkout the current cart.

        Args:
            shipping_address_id: Shipping address ID.
            payment_method_id: Payment method ID.

        Returns:
            Checkout result.
        """
        return self._execute_tool(
            "order.checkout",
            {
                "shipping_address_id": shipping_address_id,
                "payment_method_id": payment_method_id,
            },
        )

    def return_check_eligibility(self, order_id: str, line_id: str) -> str:
        """Check whether an order line is return eligible.

        Args:
            order_id: Order ID returned by order_list.
            line_id: Line ID in the order.

        Returns:
            Return eligibility result.
        """
        return self._execute_tool(
            "return.check_eligibility",
            {"order_id": order_id, "line_id": line_id},
        )

    def return_initiate(
        self,
        order_id: str,
        line_id: str,
        reason_code: str,
        method: str = "refund",
    ) -> str:
        """Initiate a return.

        Args:
            order_id: Order ID returned by order_list.
            line_id: Line ID in the order.
            reason_code: Reason code such as damaged, wrong_item, defective.
            method: refund, store_credit, or exchange.

        Returns:
            Return initiation result.
        """
        return self._execute_tool(
            "return.initiate",
            {
                "order_id": order_id,
                "line_id": line_id,
                "reason_code": reason_code,
                "method": method,
            },
        )

    def return_exchange(
        self,
        order_id: str,
        line_id: str,
        new_product_id: str,
        new_variant_id: str | None = None,
    ) -> str:
        """Start an exchange for a new product.

        Args:
            order_id: Order ID returned by order_list.
            line_id: Line ID in the order.
            new_product_id: Replacement product ID returned by catalog tools.
            new_variant_id: Optional replacement variant ID.

        Returns:
            Exchange result.
        """
        return self._execute_tool(
            "return.exchange",
            {
                "order_id": order_id,
                "line_id": line_id,
                "new_product_id": new_product_id,
                "new_variant_id": new_variant_id,
            },
        )

    def policy_search(self, query: str, top_k: int = 5) -> str:
        """Search store policy.

        Args:
            query: Policy question or search query.
            top_k: Number of policy snippets to return.

        Returns:
            Matching policy snippets.
        """
        return self._execute_tool("policy.search", {"query": query, "top_k": top_k})

    def datetime_now(self) -> str:
        """Get the current date/time.

        Returns:
            Current date/time information.
        """
        return self._execute_tool("datetime.now", {})

    def finish(
        self,
        assistant_message: str,
        recommended_product_ids_json: str = "[]",
        selected_order_id: str = "",
        selected_line_id: str = "",
        policy_answer: str = "",
    ) -> str:
        """Submit the final answer and end the episode.

        Args:
            assistant_message: Final user-facing response.
            recommended_product_ids_json: JSON list of product IDs for recommendation tasks.
            selected_order_id: Order ID for status/return tasks when applicable.
            selected_line_id: Line ID for return tasks when applicable.
            policy_answer: Final policy answer when applicable.

        Returns:
            Final reward and reward breakdown.
        """
        answer: dict[str, Any] = {
            "env": self.env_id,
            "recommended_product_ids": _json_loads_or_default(
                recommended_product_ids_json, []
            ),
            "done": True,
        }
        if selected_order_id:
            answer["selected_order_id"] = selected_order_id
        if selected_line_id:
            answer["selected_line_id"] = selected_line_id
        if policy_answer:
            answer["policy_answer"] = policy_answer

        action = {
            "assistant_message": assistant_message,
            "tool_calls": [],
            "answer": answer,
        }
        _obs, reward, done, info = self.env.step(json.dumps(action))
        self.done = done
        self.reward = -1.0 if self.invalid_tool else float(reward)
        return _trim_result({
            "reward": self.reward,
            "done": self.done,
            "is_correct": info.get("is_correct", False),
            "termination_reason": info.get("termination_reason"),
            "reward_breakdown": info.get("reward_breakdown", {}),
        })

    def ensure_finished(self) -> None:
        """Force terminal scoring if the rollout stopped before finish."""
        if self.done:
            return
        self.finish(
            assistant_message="I have completed the task.",
            recommended_product_ids_json="[]",
        )


def reward_func(
    environments: list[EcomRLVEMultiTurnEnv] | None = None,
    **_: Any,
) -> list[float]:
    """Read final rewards from environment instances after each rollout."""
    if environments is None:
        raise RuntimeError(
            "TRL did not pass environments to reward_func. This script requires "
            "a GRPOTrainer version with environment_factory support and must not "
            "use the Unsloth-patched single-turn GRPOTrainer."
        )
    rewards = []
    for env in environments:
        env.ensure_finished()
        rewards.append(float(env.reward))
    return rewards


def build_dataset(n_prompts: int, collection: str, env_id: str | None) -> "Dataset":
    """Build prompt rows. reset(**kwargs) creates the actual task instance."""
    from datasets import Dataset

    env_ids = [env_id] if env_id else get_collection(collection)
    rows = []
    for i in range(n_prompts):
        eid = env_ids[i % len(env_ids)]
        rows.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Start the e-commerce task. Read the environment "
                        "observation, use tools as needed, and call finish "
                        "when the task is complete."
                    ),
                },
            ],
            "env_id": eid,
            "episode_seed": i * 1000 + 42,
        })
    return Dataset.from_list(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-turn EcomRLVE-GYM training with TRL environment_factory",
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument(
        "--collection",
        type=str,
        default="C4",
        choices=sorted(COLLECTIONS.keys()),
    )
    parser.add_argument(
        "--env_id",
        type=str,
        default=None,
        help="Optional single env to force, e.g. CART, PD, RETURN",
    )
    parser.add_argument("--n_prompts", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--max_completion_length", type=int, default=4096)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--load_in_16bit", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, default="outputs/ecomrlve_multiturn")
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        choices=["none", "wandb", "tensorboard", "trackio"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--disable_thinking",
        action="store_true",
        help="Pass enable_thinking=False through GRPO chat template kwargs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    valid_env_ids = set(get_collection(args.collection))
    if args.env_id and args.env_id not in valid_env_ids:
        raise ValueError(
            f"--env_id {args.env_id!r} is not in --collection {args.collection}: "
            f"{sorted(valid_env_ids)}"
        )

    _FACTORY_CONFIG.update({
        "collection": args.collection,
        "seed": args.seed,
        "env_id": args.env_id,
    })

    load_in_4bit = args.load_in_4bit and not args.load_in_16bit
    use_bf16 = (
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
        and not load_in_4bit
    )
    model_dtype = torch.bfloat16 if use_bf16 else torch.float16

    logger.info("Loading model: %s", args.model)
    logger.info("Collection=%s env_id=%s", args.collection, args.env_id or "<cycle>")
    logger.info("Precision: dtype=%s load_in_4bit=%s", model_dtype, load_in_4bit)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    quantization_config = None
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=model_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=model_dtype if not load_in_4bit else None,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.0,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = build_dataset(args.n_prompts, args.collection, args.env_id)
    logger.info("Built %d prompt rows", len(dataset))

    from trl import GRPOConfig, GRPOTrainer

    config_kwargs: dict[str, Any] = {}
    if args.disable_thinking:
        config_kwargs["chat_template_kwargs"] = {"enable_thinking": False}

    per_device_train_batch_size = max(args.batch_size, args.num_generations)
    if per_device_train_batch_size != args.batch_size:
        logger.info(
            "Adjusting per-device batch size from %d to num_generations=%d "
            "so GRPO generation_batch_size is divisible by num_generations.",
            args.batch_size,
            args.num_generations,
        )

    training_args = GRPOConfig(
        temperature=args.temperature,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        logging_steps=1,
        save_steps=args.save_steps,
        output_dir=args.output_dir,
        report_to=args.report_to,
        seed=args.seed,
        bf16=use_bf16,
        fp16=not use_bf16,
        **config_kwargs,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_func,
        args=training_args,
        train_dataset=dataset,
        environment_factory=EcomRLVEMultiTurnEnv,
    )
    trainer.train()

    final_dir = os.path.join(args.output_dir, "final")
    logger.info("Saving final LoRA adapters to %s", final_dir)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    main()

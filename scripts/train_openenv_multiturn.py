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
import inspect
import json
import logging
import os
import sys
from collections import Counter
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
You are a helpful e-commerce shopping assistant. Your goal is to help \
customers find products, manage orders, handle returns, and answer \
policy questions.

You can use the following tools:
- catalog.search(query, filters, top_k): Search the product catalog
- catalog.rerank(query, candidate_product_ids, top_k): Re-rank products
- catalog.get_product(product_id): Get full product details
- catalog.get_variants(product_id): Get product variants
- cart.add(product_id, variant_id, qty): Add item to cart
- cart.remove(line_id): Remove item from cart
- cart.view(): View current cart
- order.list(days): List recent orders
- order.get_status(order_id): Get order status
- order.checkout(shipping_address_id, payment_method_id): Checkout
- return.initiate(order_id, line_id, reason): Initiate a return
- policy.search(query, top_k): Search policy knowledge base

In this tool-calling interface, use the available Python tool names that
correspond to the tool names above, for example catalog_search for
catalog.search, catalog_rerank for catalog.rerank, cart_add for cart.add, and
policy_search for policy.search.

When you have found the answer, call the finish tool. Do not write the final
answer as plain assistant text after tool results. For product recommendations,
call finish with recommended_product_ids_json set to a JSON list of product IDs
returned by tools.
"""


_FACTORY_CONFIG: dict[str, Any] = {
    "collection": "C4",
    "seed": 42,
    "env_id": None,
    "difficulty": None,
    "config": {"disclose_env_id": True, "disclose_difficulty": True},
    "debug_rollouts": 0,
    "debug_result_chars": 1200,
    "trace_rollouts_dir": None,
    "trace_rollouts_limit": 0,
}
_DEBUG_ROLLOUT_COUNT = 0
_TRACE_ROLLOUT_COUNT = 0


def _json_loads_or_default(value: str | None, default: Any) -> Any:
    if value is None or value == "":
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _bool_arg(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"expected a boolean value, got {value!r}"
    )


def _trim_result(result: Any, limit: int = 6000) -> str:
    text = json.dumps(result, indent=2, default=str)
    if len(text) > limit:
        return text[:limit] + "\n... <truncated>"
    return text


def _jsonable(value: Any) -> Any:
    """Convert trainer/env payloads into JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "shape"):
        return {
            "type": type(value).__name__,
            "shape": list(value.shape),
            "dtype": str(getattr(value, "dtype", "")),
        }
    return str(value)


def _select_rollout_item(value: Any, index: int, total: int) -> Any:
    """Select one rollout's value from a batched TRL reward payload."""
    if isinstance(value, (list, tuple)) and len(value) == total:
        return value[index]
    if hasattr(value, "shape") and getattr(value, "shape", None):
        try:
            if len(value.shape) > 0 and value.shape[0] == total:
                return value[index]
        except Exception:
            return value
    return value


def _dump_rollout_trace(
    env: "EcomRLVEMultiTurnEnv",
    rollout_index: int,
    total_rollouts: int,
    reward_kwargs: dict[str, Any],
) -> None:
    """Persist a full rollout trace for post-hoc GRPO debugging."""
    global _TRACE_ROLLOUT_COUNT

    trace_dir = str(_FACTORY_CONFIG.get("trace_rollouts_dir") or "")
    if not trace_dir:
        return

    limit = int(_FACTORY_CONFIG.get("trace_rollouts_limit") or 0)
    if limit > 0 and _TRACE_ROLLOUT_COUNT >= limit:
        return

    trace_id = _TRACE_ROLLOUT_COUNT
    _TRACE_ROLLOUT_COUNT += 1

    state = env.env.get_episode_state()
    tool_history = state.tool_results_history if state is not None else []
    tool_counts = Counter(str(entry.get("name", "")) for entry in tool_history)
    tool_duration_ms = sum(
        float(entry.get("duration_ms") or 0.0)
        for entry in tool_history
    )

    trainer_payload: dict[str, Any] = {}
    for key, value in reward_kwargs.items():
        selected = _select_rollout_item(value, rollout_index, total_rollouts)
        trainer_payload[key] = _jsonable(selected)

    trace = env.env.get_episode_trace()
    trace["wrapper"] = {
        "debug_id": env.debug_id,
        "invalid_tool": env.invalid_tool,
        "is_correct": env.is_correct,
        "termination_reason": env.termination_reason,
        "reward": env.reward,
        "tool_count": len(tool_history),
        "tool_counts": dict(sorted(tool_counts.items())),
        "tool_duration_ms": tool_duration_ms,
    }
    trace["trl"] = {
        "trace_id": trace_id,
        "pid": os.getpid(),
        "rollout_index": rollout_index,
        "rollouts_in_reward_call": total_rollouts,
        "payload": trainer_payload,
    }

    out_dir = Path(trace_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env_id = env.env_id or trace.get("env_id") or "env"
    seed = env.episode_seed or trace.get("seed") or 0
    filename = f"rollout_{os.getpid()}_{trace_id:06d}_{env_id}_s{seed}.json"
    filepath = out_dir / filename
    with open(filepath, "w") as f:
        json.dump(trace, f, indent=2, default=str)

    summary = {
        "trace_id": trace_id,
        "path": str(filepath),
        "env_id": env_id,
        "seed": seed,
        "reward": env.reward,
        "is_correct": env.is_correct,
        "termination_reason": env.termination_reason,
        "turns": trace.get("turn"),
        "tool_count": len(tool_history),
        "tool_counts": dict(sorted(tool_counts.items())),
    }
    with open(out_dir / "summary.jsonl", "a") as f:
        f.write(json.dumps(summary, default=str) + "\n")


class EcomRLVEMultiTurnEnv:
    """TRL environment_factory wrapper around EcomRLVEEnv.

    Public methods become callable tools for the model.  Tool methods execute
    against the underlying EcomRLVE tool registry without terminating the
    episode.  The finish method submits the final answer through env.step(),
    which lets the existing verifiers compute reward.
    """

    def __init__(self) -> None:
        global _DEBUG_ROLLOUT_COUNT
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
        self.is_correct = False
        self.termination_reason = ""
        self.reward_breakdown: dict[str, Any] = {}
        self.debug_enabled = _DEBUG_ROLLOUT_COUNT < int(_FACTORY_CONFIG["debug_rollouts"])
        self.debug_id = _DEBUG_ROLLOUT_COUNT
        _DEBUG_ROLLOUT_COUNT += 1

    def _debug(self, message: str, *args: Any) -> None:
        if self.debug_enabled:
            logger.info("[rollout:%03d] " + message, self.debug_id, *args)

    def reset(self, **kwargs: Any) -> str:
        """Start a fresh EcomRLVE episode and return the first user message."""
        self.reward = 0.0
        self.done = False
        self.invalid_tool = False
        self.is_correct = False
        self.termination_reason = ""
        self.reward_breakdown = {}

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
        self._debug(
            "reset env=%s seed=%s user=%s",
            self.env_id,
            self.episode_seed,
            user_message[:300],
        )
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
        result_preview = _trim_result(entry, limit=int(_FACTORY_CONFIG["debug_result_chars"]))
        self._debug(
            "tool=%s args=%s error=%s result=%s",
            name,
            json.dumps(args, default=str),
            result.error,
            result_preview,
        )
        return _trim_result(entry)

    def _execute_catalog_search(
        self,
        query: str,
        filters: dict[str, Any] | None,
        top_k: int,
    ) -> str:
        """Run catalog.search with enough retrieval depth for post-filtering."""
        if self.done:
            raise ValueError("Episode is already done.")
        state = self.env.get_episode_state()
        if state is None:
            raise ValueError("Environment has not been reset.")

        requested_top_k = max(1, int(top_k))
        internal_top_k = max(requested_top_k, 200 if filters else requested_top_k)
        seen_before = set(state.seen_product_ids)
        result = self.env._tool_registry.execute(
            ToolCall(
                name="catalog.search",
                args={"query": query, "filters": filters, "top_k": internal_top_k},
            ),
            state=state,
        )
        returned = result.result if isinstance(result.result, list) else result.result
        if isinstance(returned, list):
            returned = returned[:requested_top_k]
            state.seen_product_ids = seen_before
            self.env._extract_seen_ids(returned, state.seen_product_ids)

        entry = {
            "name": "catalog.search",
            "args": {"query": query, "filters": filters, "top_k": requested_top_k},
            "result": returned,
            "error": result.error,
            "duration_ms": result.duration_ms,
        }
        if internal_top_k != requested_top_k:
            entry["internal_top_k"] = internal_top_k
            entry["message"] = (
                "Used a larger internal retrieval pool before applying filters; "
                "showing only the requested top_k results."
            )
        state.tool_results_history.append(entry)
        if result.error is not None:
            self.invalid_tool = True
        result_preview = _trim_result(entry, limit=int(_FACTORY_CONFIG["debug_result_chars"]))
        self._debug(
            "tool=%s args=%s error=%s result=%s",
            "catalog.search",
            json.dumps(entry["args"], default=str),
            result.error,
            result_preview,
        )
        return _trim_result(entry)

    def _record_tool_observation(
        self,
        name: str,
        args: dict[str, Any],
        result: Any,
        message: str,
    ) -> str:
        """Append a non-error tool observation without calling the tool registry."""
        state = self.env.get_episode_state()
        if state is None:
            raise ValueError("Environment has not been reset.")
        entry = {
            "name": name,
            "args": args,
            "result": result,
            "error": None,
            "message": message,
            "duration_ms": 0.0,
        }
        state.tool_results_history.append(entry)
        self._debug(
            "tool=%s args=%s error=None result=%s",
            name,
            json.dumps(args, default=str),
            _trim_result(entry, limit=int(_FACTORY_CONFIG["debug_result_chars"])),
        )
        return _trim_result(entry)

    def user_get_visit_history(self) -> str:
        """Get the customer's recently viewed products.

        Returns:
            Recently viewed products with product IDs and attributes.
        """
        return self._execute_tool("user.get_visit_history", {})

    def catalog_search(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
    ) -> str:
        """Search the product catalog.

        Args:
            query: Natural-language search query.
            filters: Optional metadata filters using catalog keys, e.g.
                {"cat": "toys/educational/stem", "price_max": 93.07, "color": "orange"}.
            top_k: Number of results to return.

        Returns:
            Matching products.
        """
        return self._execute_catalog_search(query=query, filters=filters, top_k=top_k)

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
        if not isinstance(ids, list):
            ids = []
        ids = [pid for pid in ids if isinstance(pid, str) and pid]
        if not ids:
            return self._record_tool_observation(
                "catalog.rerank",
                {"query": query, "candidate_product_ids": [], "top_k": top_k},
                [],
                "No candidate product IDs were provided. Run catalog_search again with broader query or filters.",
            )
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
        self.is_correct = bool(info.get("is_correct", False))
        self.termination_reason = str(info.get("termination_reason") or "")
        self.reward_breakdown = info.get("reward_breakdown", {})
        summary = {
            "reward": self.reward,
            "done": self.done,
            "is_correct": self.is_correct,
            "termination_reason": self.termination_reason,
            "reward_breakdown": self.reward_breakdown,
        }
        self._debug(
            "finish message=%s answer=%s summary=%s",
            assistant_message[:300],
            json.dumps(answer, default=str),
            _trim_result(summary, limit=int(_FACTORY_CONFIG["debug_result_chars"])),
        )
        return _trim_result(summary)

    def ensure_finished(self) -> None:
        """Force terminal scoring if the rollout stopped before finish."""
        if self.done:
            return
        state = self.env.get_episode_state()
        tool_history = state.tool_results_history if state is not None else []
        seen_product_ids = sorted(state.seen_product_ids) if state is not None else []
        self._debug(
            "fallback_finish invalid_tool=%s seen_product_ids=%s tool_history=%s",
            self.invalid_tool,
            json.dumps(seen_product_ids),
            _trim_result(tool_history, limit=int(_FACTORY_CONFIG["debug_result_chars"])),
        )
        self.finish(
            assistant_message="I have completed the task.",
            recommended_product_ids_json=json.dumps(seen_product_ids),
        )


def reward_func(
    environments: list[EcomRLVEMultiTurnEnv] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Read final rewards from environment instances after each rollout."""
    if environments is None:
        raise RuntimeError(
            "TRL did not pass environments to reward_func. This script requires "
            "a GRPOTrainer version with environment_factory support and must not "
            "use the Unsloth-patched single-turn GRPOTrainer."
        )
    if any(env.debug_enabled for env in environments):
        debug_kwargs: dict[str, Any] = {}
        for key, value in kwargs.items():
            if key in {"prompts", "completions", "completion_ids"}:
                text = repr(value)
                if len(text) > int(_FACTORY_CONFIG["debug_result_chars"]):
                    text = text[: int(_FACTORY_CONFIG["debug_result_chars"])] + "... <truncated>"
                debug_kwargs[key] = text
            else:
                debug_kwargs[key] = type(value).__name__
        logger.info("[reward_func] kwargs=%s", json.dumps(debug_kwargs, default=str))
    rewards = []
    for rollout_index, env in enumerate(environments):
        env.ensure_finished()
        rewards.append(float(env.reward))
        _dump_rollout_trace(
            env=env,
            rollout_index=rollout_index,
            total_rollouts=len(environments),
            reward_kwargs=kwargs,
        )
    if any(env.debug_enabled for env in environments):
        mean_reward = sum(rewards) / max(len(rewards), 1)
        reward_std = (
            sum((reward - mean_reward) ** 2 for reward in rewards) / max(len(rewards), 1)
        ) ** 0.5
        env_counts = Counter(env.env_id for env in environments)
        reason_counts = Counter(env.termination_reason or "unknown" for env in environments)
        correct_count = sum(1 for env in environments if env.is_correct)
        logger.info(
            "[reward_func] rewards=%s mean=%.4f std=%.4f correct=%d/%d envs=%s reasons=%s",
            json.dumps([round(float(r), 4) for r in rewards]),
            mean_reward,
            reward_std,
            correct_count,
            len(environments),
            json.dumps(dict(sorted(env_counts.items()))),
            json.dumps(dict(sorted(reason_counts.items()))),
        )
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
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature. Default 0.6 matches Qwen3 thinking-mode guidance.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Nucleus sampling top_p. Default 0.95 matches Qwen3 thinking-mode guidance.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Top-k sampling. Default 20 matches Qwen3 guidance.",
    )
    parser.add_argument(
        "--min_p",
        type=float,
        default=0.0,
        help="Minimum probability sampling cutoff. Default 0.0 matches Qwen3 guidance.",
    )
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--max_completion_length", type=int, default=4096)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--load_in_16bit", action="store_true", default=False)
    parser.add_argument(
        "--precision",
        type=str,
        default="none",
        choices=["none", "fp16", "bf16"],
        help=(
            "Trainer mixed precision for the native TRL path. Use none to "
            "avoid AMP GradScaler issues with 4-bit training (default: none)."
        ),
    )
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
        "--embedding_debug",
        type=_bool_arg,
        default=None,
        metavar="true|false",
        help=(
            "Override environment embedding debug mode. Use false to build "
            "a real semantic catalog index for synthetic catalogs."
        ),
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default=None,
        help="Optional SentenceTransformer model for catalog retrieval.",
    )
    parser.add_argument(
        "--embedding_device",
        type=str,
        default=None,
        help="Optional embedding device, e.g. cuda:0, mps, or cpu.",
    )
    parser.add_argument(
        "--n_synthetic_products",
        type=int,
        default=None,
        help="Optional synthetic catalog size override.",
    )
    parser.add_argument(
        "--faiss_index_factory",
        type=str,
        default=None,
        help="Optional FAISS index factory when real embeddings are enabled.",
    )
    parser.add_argument(
        "--faiss_use_gpu",
        type=_bool_arg,
        default=None,
        metavar="true|false",
        help="Move FAISS index to GPU when the installed FAISS package supports it.",
    )
    parser.add_argument(
        "--faiss_index_path",
        type=str,
        default=None,
        help="Optional path to a prebuilt FAISS index directory.",
    )
    parser.add_argument(
        "--disable_thinking",
        action="store_true",
        help=(
            "Pass enable_thinking=False through GRPO chat template kwargs. "
            "By default thinking mode is explicitly left enabled."
        ),
    )
    parser.add_argument(
        "--debug_rollouts",
        type=int,
        default=0,
        help="Log internal env/tool/reward details for the first N rollout env instances.",
    )
    parser.add_argument(
        "--debug_result_chars",
        type=int,
        default=1200,
        help="Maximum characters per debug tool result/reward payload.",
    )
    parser.add_argument(
        "--trace_rollouts_dir",
        type=str,
        default=None,
        help=(
            "Directory to save full GRPO rollout traces as JSON files. Defaults "
            "to <output_dir>/traces. Pass an empty string to disable tracing."
        ),
    )
    parser.add_argument(
        "--trace_rollouts_limit",
        type=int,
        default=0,
        help="Maximum number of rollout traces to save. Use 0 for unlimited.",
    )
    parser.add_argument(
        "--use_transformers_continuous_batching",
        type=_bool_arg,
        default=None,
        metavar="true|false",
        help=(
            "Enable TRL/Transformers continuous batching for generation when "
            "the installed TRL GRPOConfig supports it."
        ),
    )
    parser.add_argument(
        "--transformers_cb_max_memory_percent",
        type=float,
        default=None,
        help="Optional max_memory_percent for transformers continuous batching.",
    )
    parser.add_argument(
        "--transformers_cb_use_cuda_graph",
        type=_bool_arg,
        default=None,
        metavar="true|false",
        help="Optional use_cuda_graph value for transformers continuous batching.",
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
    env_config.update({
        key: value
        for key, value in optional_env_config.items()
        if value is not None
    })

    trace_rollouts_dir = (
        os.path.join(args.output_dir, "traces")
        if args.trace_rollouts_dir is None
        else args.trace_rollouts_dir
    )

    _FACTORY_CONFIG.update({
        "collection": args.collection,
        "seed": args.seed,
        "env_id": args.env_id,
        "config": env_config,
        "debug_rollouts": args.debug_rollouts,
        "debug_result_chars": args.debug_result_chars,
        "trace_rollouts_dir": trace_rollouts_dir,
        "trace_rollouts_limit": args.trace_rollouts_limit,
    })

    load_in_4bit = args.load_in_4bit and not args.load_in_16bit
    use_fp16 = args.precision == "fp16"
    use_bf16 = args.precision == "bf16"
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError("--precision bf16 requested, but this CUDA device does not support bf16.")
    model_dtype = torch.bfloat16 if use_bf16 else torch.float16

    logger.info("Loading model: %s", args.model)
    logger.info("Collection=%s env_id=%s", args.collection, args.env_id or "<cycle>")
    logger.info(
        "Precision: model dtype=%s trainer fp16=%s trainer bf16=%s load_in_4bit=%s",
        model_dtype,
        use_fp16,
        use_bf16,
        load_in_4bit,
    )
    logger.info("Environment config: %s", json.dumps(env_config, default=str))

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
    grpo_config_params = inspect.signature(GRPOConfig).parameters
    requested_generation_kwargs: dict[str, Any] = {
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
    }
    for key, value in requested_generation_kwargs.items():
        if key in grpo_config_params:
            config_kwargs[key] = value
        else:
            logger.warning(
                "Installed TRL GRPOConfig does not support %s; leaving it at TRL/model default.",
                key,
            )

    thinking_enabled = not args.disable_thinking
    if "chat_template_kwargs" in grpo_config_params:
        config_kwargs["chat_template_kwargs"] = {"enable_thinking": thinking_enabled}
    elif args.disable_thinking:
        logger.warning(
            "Installed TRL GRPOConfig does not support chat_template_kwargs; "
            "cannot pass enable_thinking=False."
        )

    if args.use_transformers_continuous_batching is not None:
        key = "use_transformers_continuous_batching"
        if key in grpo_config_params:
            config_kwargs[key] = args.use_transformers_continuous_batching
        else:
            logger.warning(
                "Installed TRL GRPOConfig does not support %s; ignoring request.",
                key,
            )

    transformers_cb_config: dict[str, Any] = {}
    if args.transformers_cb_max_memory_percent is not None:
        transformers_cb_config["max_memory_percent"] = args.transformers_cb_max_memory_percent
    if args.transformers_cb_use_cuda_graph is not None:
        transformers_cb_config["use_cuda_graph"] = args.transformers_cb_use_cuda_graph
    if transformers_cb_config:
        key = "transformers_continuous_batching_config"
        if key in grpo_config_params:
            config_kwargs[key] = transformers_cb_config
        else:
            logger.warning(
                "Installed TRL GRPOConfig does not support %s; ignoring request.",
                key,
            )

    logger.info(
        "Generation: temperature=%s top_p=%s top_k=%s min_p=%s thinking=%s "
        "transformers_continuous_batching=%s",
        args.temperature,
        args.top_p,
        args.top_k,
        args.min_p,
        "enabled" if thinking_enabled else "disabled",
        config_kwargs.get("use_transformers_continuous_batching", "<default>"),
    )

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
        max_grad_norm=args.max_grad_norm,
        logging_steps=1,
        save_steps=args.save_steps,
        output_dir=args.output_dir,
        report_to=args.report_to,
        seed=args.seed,
        bf16=use_bf16,
        fp16=use_fp16,
        remove_unused_columns=False,
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

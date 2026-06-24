#!/usr/bin/env python3
"""Generate CART trajectories using OpenAI or OpenAI-compatible endpoints.

This reuses the Ollama trajectory runner's environment loop and recording
format, but swaps the agent LLM call to the OpenAI SDK.

Usage:
    OPENAI_API_KEY=... uv run python scripts/generate_trajectories_openai.py

Optional:
    OPENAI_BASE_URL=https://api.openai.com/v1
    OPENAI_MODEL=gpt-4.1-mini

Azure/OpenAI-compatible aliases are also supported:
    AZURE_OPENAI_API_KEY=...
    AZURE_OPENAI_ENDPOINT=https://customersupport.openai.azure.com/openai/v1/
    AZURE_OPENAI_DEPLOYMENT_NAME=gpt-5.4-nano
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE pairs from a .env file without extra dependencies."""
    if not path.exists():
        return

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_env_file(PROJECT_ROOT / ".env")

import generate_trajectories as trajectory_runner  # noqa: E402
from ecom_rlve.data.catalog_loader import load_catalog  # noqa: E402
from ecom_rlve.server.openenv import EcomRLVEEnv  # noqa: E402

DEFAULT_MODEL = (
    os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    or os.getenv("OPENAI_DEPLOYMENT_NAME")
    or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
)
DEFAULT_BASE_URL = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("OPENAI_BASE_URL")
if DEFAULT_BASE_URL:
    DEFAULT_BASE_URL = DEFAULT_BASE_URL.rstrip("/")
DEFAULT_API_KEY = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
TIMEOUT = 60
DEFAULT_CATALOG = "owlgebra-ai/Amazebay-catalog-2M"

_client: OpenAI | None = None


def save_results_atomic(results: dict, output_path: Path) -> None:
    """Write results atomically so interrupted runs keep the last checkpoint."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    tmp_path.replace(output_path)


def get_openai_client() -> OpenAI:
    """Create the OpenAI client from environment configuration."""
    global _client
    if _client is not None:
        return _client

    kwargs = {"timeout": TIMEOUT}
    if DEFAULT_BASE_URL:
        kwargs["base_url"] = DEFAULT_BASE_URL
        kwargs["api_key"] = DEFAULT_API_KEY or "not-needed"
    _client = OpenAI(**kwargs)
    return _client


def openai_chat(
    messages: list[dict[str, str]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    seed: int = 42,
) -> str | None:
    """Call an OpenAI chat-completions endpoint and return assistant content."""
    try:
        client = get_openai_client()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,
                seed=seed,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            # Some OpenAI-compatible endpoints do not implement seed or JSON mode.
            print(f"  [WARN] Retrying without seed/json mode after OpenAI call failed: {exc}")
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )

        text = (response.choices[0].message.content or "").strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return text if text else None
    except Exception as exc:
        print(f"  [WARN] OpenAI call failed: {exc}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CART trajectories with OpenAI agent")
    parser.add_argument("--output", default="data/cart_trajectories_openai.json", help="Output JSON file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model name")
    parser.add_argument("--min-d", type=int, default=0, help="Min difficulty")
    parser.add_argument("--max-d", type=int, default=10, help="Max difficulty")
    parser.add_argument(
        "--num-per-d",
        type=int,
        default=1,
        help="Number of trajectories to generate per difficulty level",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    parser.add_argument(
        "--catalog",
        default=DEFAULT_CATALOG,
        help="Hugging Face dataset name or local dataset path for real catalog",
    )
    parser.add_argument(
        "--catalog-size",
        type=int,
        default=5000,
        help="Max real catalog items to load, or synthetic size with --synthetic",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use generated synthetic products instead of loading a real catalog",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Overwrite output instead of resuming/skipping completed samples",
    )
    args = parser.parse_args()

    hosted_openai = not DEFAULT_BASE_URL or "api.openai.com" in DEFAULT_BASE_URL
    if hosted_openai and not DEFAULT_API_KEY:
        print("ERROR: Set AZURE_OPENAI_API_KEY or OPENAI_API_KEY in .env.")
        return

    # Swap the imported runner's LLM client. run_trajectory resolves this
    # module global at call time, so no copy of the rollout logic is needed.
    trajectory_runner.ollama_chat = openai_chat

    env_config = {
        "disclose_env_id": True,
        "disclose_difficulty": True,
    }
    catalog = None
    catalog_source = "synthetic"
    if args.synthetic:
        env_config["n_synthetic_products"] = args.catalog_size
    else:
        print(f"Loading real catalog from {args.catalog} (max_items={args.catalog_size})")
        products = load_catalog(args.catalog, max_items=args.catalog_size, seed=args.seed)
        catalog = (products, [])
        catalog_source = args.catalog
        env_config["embedding_debug"] = False

    env = EcomRLVEEnv(
        collection="C4",
        catalog=catalog,
        seed=args.seed,
        config=env_config,
    )
    print(f"Environment ready ({catalog_source}: {args.catalog_size} products)")
    print(f"OpenAI endpoint ready, model: {args.model}")

    output_path = Path(args.output)
    results = {
        "metadata": {
            "env_id": "CART",
            "model": args.model,
            "provider": "openai",
            "base_url": DEFAULT_BASE_URL or "https://api.openai.com/v1",
            "catalog_source": catalog_source,
            "min_difficulty": args.min_d,
            "max_difficulty": args.max_d,
            "num_per_difficulty": args.num_per_d,
            "base_seed": args.seed,
            "catalog_size": args.catalog_size,
        },
        "trajectories": [],
    }
    if output_path.exists() and not args.no_resume:
        try:
            existing = json.loads(output_path.read_text())
            if isinstance(existing, dict) and isinstance(existing.get("trajectories"), list):
                results = existing
                results.setdefault("metadata", {}).update({
                    "model": args.model,
                    "provider": "openai",
                    "base_url": DEFAULT_BASE_URL or "https://api.openai.com/v1",
                    "catalog_source": catalog_source,
                    "min_difficulty": args.min_d,
                    "max_difficulty": args.max_d,
                    "num_per_difficulty": args.num_per_d,
                    "base_seed": args.seed,
                    "catalog_size": args.catalog_size,
                })
                print(f"Resuming from {output_path} ({len(results['trajectories'])} saved)")
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Could not resume from {output_path}: {exc}. Starting fresh.")

    completed = {
        (traj.get("difficulty"), traj.get("sample_index", 0))
        for traj in results.get("trajectories", [])
    }

    total_t0 = time.time()

    for d in range(args.min_d, args.max_d + 1):
        for sample_idx in range(args.num_per_d):
            if (d, sample_idx) in completed:
                print(f"Skipping Difficulty {d} | Sample {sample_idx + 1}/{args.num_per_d} (already saved)")
                continue

            print(f"\n{'=' * 60}")
            print(f"  Difficulty {d} | Sample {sample_idx + 1}/{args.num_per_d}")
            print(f"{'=' * 60}")

            traj_seed = args.seed * 100000 + d * 1000 + sample_idx
            traj = trajectory_runner.run_trajectory(
                env,
                difficulty=d,
                seed=traj_seed,
                model=args.model,
            )
            traj["sample_index"] = sample_idx
            results["trajectories"].append(traj)
            completed.add((d, sample_idx))

            n_steps = len(traj["steps"])
            n_tools = sum(len(s["_meta"]["tool_calls"]) for s in traj["steps"])
            n_parse_ok = sum(1 for s in traj["steps"] if s["_meta"]["parse_success"])
            print(f"  Steps: {n_steps}, Tool calls: {n_tools}, Parse OK: {n_parse_ok}/{n_steps}")
            print(f"  Reward: {traj['final_reward']}, IsCorrect: {traj['is_correct']}")
            print(f"  Time: {traj['generation_time_s']}s")

            for step in traj["steps"]:
                tool_names = [tc["name"] for tc in step["_meta"]["tool_calls"]]
                answer = " [ANSWER]" if step["_meta"]["submitted_answer"] else ""
                print(
                    f"    t={step['turn']}: "
                    f"{', '.join(tool_names) or 'no tools'}{answer} -> "
                    f"r={step['_meta']['reward']}"
                )

            results["metadata"]["completed_trajectories"] = len(results["trajectories"])
            results["metadata"]["last_checkpoint_time_s"] = round(time.time() - total_t0, 1)
            save_results_atomic(results, output_path)
            print(f"  Checkpoint saved to {output_path}")

    total_time = round(time.time() - total_t0, 1)
    results["metadata"]["total_generation_time_s"] = total_time

    save_results_atomic(results, output_path)

    print(f"\n{'=' * 60}")
    print(f"  Saved {len(results['trajectories'])} trajectories to {output_path}")
    print(f"  Total time: {total_time}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

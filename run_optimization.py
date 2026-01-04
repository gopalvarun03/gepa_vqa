#!/usr/bin/env python3
"""
GEPA VQA Optimization Script.

This script optimizes VQA system prompts using the GEPA framework
with InternVL as the VLM and VQAv2/TextVQA as the dataset.

Usage:
    python run_optimization.py --model_path /path/to/InternVL \
        --dataset vqav2 --image_dir /path/to/images \
        --max_metric_calls 100 --reflection_lm openai/gpt-4
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load .env file
load_dotenv(Path(__file__).parent / ".env")

# Add GEPA to path
GEPA_PATH = Path(__file__).parent.parent / "gepa" / "src"
sys.path.insert(0, str(GEPA_PATH))

import gepa
from vqa_adapter import VQAAdapter, MockVLM
from vqa_dataset import load_vqa_dataset, create_mock_dataset
from vlm_wrapper import InternVLWrapper
from openrouter_vlm import OpenRouterVLM
from vllm_vlm import VLLMWrapper


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Optimize VQA system prompts using GEPA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model configuration
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to InternVL model. If not provided, uses mock VLM for testing.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run model on",
    )
    parser.add_argument(
        "--use_openrouter",
        action="store_true",
        help="Use OpenRouter API instead of local model",
    )
    parser.add_argument(
        "--openrouter_model",
        type=str,
        default="qwen/qwen-2.5-vl-7b-instruct:free",
        help="OpenRouter model to use (e.g., 'qwen/qwen-2.5-vl-7b-instruct:free')",
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="Use local vLLM server",
    )
    parser.add_argument(
        "--vllm_url",
        type=str,
        default="http://localhost:8001",
        help="vLLM server URL",
    )
    parser.add_argument(
        "--vllm_model",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Model name for vLLM server",
    )

    # Dataset configuration
    parser.add_argument(
        "--dataset",
        type=str,
        default="mock",
        help="Dataset name: 'vqav2', 'textvqa', 'mock', or path to JSONL file",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Directory containing dataset images",
    )
    parser.add_argument(
        "--train_samples",
        type=int,
        default=500,
        help="Number of training samples to use",
    )
    parser.add_argument(
        "--val_samples",
        type=int,
        default=100,
        help="Number of validation samples to use",
    )

    # GEPA configuration
    parser.add_argument(
        "--reflection_lm",
        type=str,
        default=None,
        help="LLM to use for reflection. If not set and --use_openrouter is set, uses the same OpenRouter model.",
    )
    parser.add_argument(
        "--max_metric_calls",
        type=int,
        default=100,
        help="Maximum number of evaluation calls during optimization",
    )
    parser.add_argument(
        "--reflection_minibatch_size",
        type=int,
        default=3,
        help="Number of examples per reflection batch",
    )
    parser.add_argument(
        "--candidate_selection",
        type=str,
        default="pareto",
        choices=["pareto", "current_best", "epsilon_greedy"],
        help="Candidate selection strategy",
    )
    parser.add_argument(
        "--use_merge",
        action="store_true",
        help="Enable merge strategy for combining candidates",
    )

    # Seed prompt
    parser.add_argument(
        "--seed_prompt",
        type=str,
        default="Answer the question about the image using a single word or phrase.",
        help="Initial system prompt to optimize from",
    )

    # Output configuration
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Directory to save results. Defaults to runs/TIMESTAMP",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    return parser.parse_args()


def main():
    """Main optimization routine."""
    args = parse_args()

    print("=" * 60)
    print("GEPA VQA Prompt Optimization")
    print("=" * 60)

    # Setup run directory
    if args.run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_dir = f"runs/gepa_vqa_{timestamp}"
    os.makedirs(args.run_dir, exist_ok=True)
    print(f"\nRun directory: {args.run_dir}")

    # Load VLM
    print("\n1. Loading VLM...")
    if args.use_vllm:
        print(f"   Using vLLM server at: {args.vllm_url}")
        print(f"   Model: {args.vllm_model}")
        vlm = VLLMWrapper(base_url=args.vllm_url, model=args.vllm_model)

        # Set reflection LM to use same vLLM server if not specified
        if args.reflection_lm is None:
            # litellm uses format: openai/<model> with api_base
            # For vLLM, we use hosted_vllm format
            args.reflection_lm = f"hosted_vllm/{args.vllm_model}"
            os.environ["HOSTED_VLLM_API_BASE"] = f"{args.vllm_url}/v1"
            print(f"   Reflection LM: {args.reflection_lm}")
    elif args.use_openrouter:
        print(f"   Using OpenRouter with model: {args.openrouter_model}")
        vlm = OpenRouterVLM(model=args.openrouter_model)

        # Set reflection LM to use OpenRouter if not specified
        if args.reflection_lm is None:
            # litellm uses format: openrouter/<model>
            args.reflection_lm = f"openrouter/{args.openrouter_model}"
            print(f"   Reflection LM: {args.reflection_lm}")
    elif args.model_path:
        vlm = InternVLWrapper(
            model_path=args.model_path,
            device=args.device,
        )
        if args.reflection_lm is None:
            args.reflection_lm = "openai/gpt-4"
    else:
        print("   Using mock VLM for testing (no --model_path provided)")
        vlm = MockVLM()  # type: ignore
        if args.reflection_lm is None:
            args.reflection_lm = "openai/gpt-4"

    # Load dataset
    print("\n2. Loading dataset...")
    if args.dataset == "mock":
        print("   Using mock dataset for testing")
        trainset = create_mock_dataset(args.train_samples)
        valset = create_mock_dataset(args.val_samples)
    else:
        trainset = load_vqa_dataset(
            dataset_name=args.dataset,
            split="train",
            image_dir=args.image_dir,
            max_samples=args.train_samples,
        )
        valset = load_vqa_dataset(
            dataset_name=args.dataset,
            split="validation",
            image_dir=args.image_dir,
            max_samples=args.val_samples,
        )

    print(f"   Training samples: {len(trainset)}")
    print(f"   Validation samples: {len(valset)}")

    # Create adapter
    print("\n3. Creating VQA adapter...")
    adapter = VQAAdapter(vlm=vlm)

    # Define seed candidate
    seed_candidate = {
        "system_prompt": args.seed_prompt,
    }
    print(f"\n4. Seed prompt: '{args.seed_prompt}'")

    # Run GEPA optimization
    print("\n5. Starting GEPA optimization...")
    print(f"   Reflection LM: {args.reflection_lm}")
    print(f"   Max metric calls: {args.max_metric_calls}")
    print(f"   Candidate selection: {args.candidate_selection}")
    print(f"   Use merge: {args.use_merge}")

    try:
        result = gepa.optimize(
            seed_candidate=seed_candidate,
            trainset=trainset,
            valset=valset,
            adapter=adapter,
            reflection_lm=args.reflection_lm,
            max_metric_calls=args.max_metric_calls,
            reflection_minibatch_size=args.reflection_minibatch_size,
            candidate_selection_strategy=args.candidate_selection,
            use_merge=args.use_merge,
            run_dir=args.run_dir,
            use_wandb=args.use_wandb,
            seed=args.seed,
            display_progress_bar=True,
        )

        # Print results
        print("\n" + "=" * 60)
        print("OPTIMIZATION COMPLETE")
        print("=" * 60)

        print(f"\nTotal candidates evaluated: {len(result.candidates)}")
        print(f"Total metric calls: {result.total_metric_calls}")

        print(f"\nBest candidate index: {result.best_idx}")
        print(f"Best validation score: {result.val_aggregate_scores[result.best_idx]:.4f}")

        print("\nBest prompt:")
        print("-" * 40)
        print(result.best_candidate["system_prompt"])
        print("-" * 40)

        # Save best prompt to file
        best_prompt_path = os.path.join(args.run_dir, "best_prompt.txt")
        with open(best_prompt_path, "w") as f:
            f.write(result.best_candidate["system_prompt"])
        print(f"\nBest prompt saved to: {best_prompt_path}")

        # Print score progression
        print("\nScore progression:")
        for i, score in enumerate(result.val_aggregate_scores):
            print(f"  Candidate {i}: {score:.4f}")

        # Print lineage of best candidate using parents attribute
        if len(result.candidates) > 1:
            best_idx = result.best_idx
            lineage = [best_idx]
            current = best_idx
            while current > 0 and result.parents[current]:
                parent = result.parents[current][0]  # Get first parent
                if parent is not None:
                    lineage.append(parent)
                    current = parent
                else:
                    break
            lineage.reverse()
            if len(lineage) > 1:
                print(f"\nBest candidate lineage: {' -> '.join(map(str, lineage))}")

    except Exception as e:
        print(f"\nError during optimization: {e}")
        raise

    print("\nDone!")


if __name__ == "__main__":
    main()

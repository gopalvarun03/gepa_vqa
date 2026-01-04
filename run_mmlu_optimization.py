#!/usr/bin/env python3
"""
GEPA MMLU Optimization Script.

This script optimizes multiple choice QA prompts using the GEPA framework
with MMLU as the benchmark dataset.

Usage:
    python run_mmlu_optimization.py --vllm_url http://localhost:8001 \
        --max_metric_calls 100 --train_samples 200 --val_samples 100
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
from mmlu_adapter import MMLUAdapter, VLLMTextWrapper
from mmlu_dataset import load_mmlu, get_mmlu_subjects


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Optimize MMLU prompts using GEPA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model configuration
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
        "--data_path",
        type=str,
        default="/home/shivank_g/projects/vlm_reason/varun_codes/vqa_datasets/mmlu",
        help="Path to MMLU dataset",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        nargs="+",
        default=None,
        help="Filter to specific subjects (e.g., 'abstract_algebra' 'anatomy')",
    )
    parser.add_argument(
        "--train_samples",
        type=int,
        default=200,
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
        help="LLM for reflection. If not set, uses same vLLM server.",
    )
    parser.add_argument(
        "--max_metric_calls",
        type=int,
        default=200,
        help="Maximum number of evaluation calls",
    )
    parser.add_argument(
        "--reflection_minibatch_size",
        type=int,
        default=5,
        help="Number of examples per reflection batch",
    )
    parser.add_argument(
        "--candidate_selection",
        type=str,
        default="pareto",
        choices=["pareto", "current_best", "epsilon_greedy"],
        help="Candidate selection strategy",
    )

    # Seed prompt
    parser.add_argument(
        "--seed_prompt",
        type=str,
        default="You are an expert at answering multiple choice questions. Analyze the question carefully and select the correct answer. Respond with just the letter (A, B, C, or D).",
        help="Initial system prompt to optimize from",
    )

    # Output configuration
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Directory to save results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser.parse_args()


def main():
    """Main optimization routine."""
    args = parse_args()

    print("=" * 60)
    print("GEPA MMLU Prompt Optimization")
    print("=" * 60)

    # Setup run directory
    if args.run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_dir = f"runs/gepa_mmlu_{timestamp}"
    os.makedirs(args.run_dir, exist_ok=True)
    print(f"\nRun directory: {args.run_dir}")

    # Load LLM
    print("\n1. Loading LLM...")
    print(f"   vLLM server: {args.vllm_url}")
    print(f"   Model: {args.vllm_model}")
    llm = VLLMTextWrapper(base_url=args.vllm_url, model=args.vllm_model)

    # Set reflection LM
    if args.reflection_lm is None:
        args.reflection_lm = f"hosted_vllm/{args.vllm_model}"
        os.environ["HOSTED_VLLM_API_BASE"] = f"{args.vllm_url}/v1"
        print(f"   Reflection LM: {args.reflection_lm}")

    # Load dataset
    print("\n2. Loading MMLU dataset...")
    if args.subjects:
        print(f"   Filtering to subjects: {args.subjects}")

    trainset = load_mmlu(
        split="auxiliary_train",
        subjects=args.subjects,
        max_samples=args.train_samples,
        data_path=args.data_path,
    )
    valset = load_mmlu(
        split="test",
        subjects=args.subjects,
        max_samples=args.val_samples,
        data_path=args.data_path,
    )

    print(f"   Training samples: {len(trainset)}")
    print(f"   Validation samples: {len(valset)}")

    # Create adapter
    print("\n3. Creating MMLU adapter...")
    adapter = MMLUAdapter(llm=llm)

    # Define seed candidate
    seed_candidate = {
        "system_prompt": args.seed_prompt,
    }
    print(f"\n4. Seed prompt: '{args.seed_prompt[:80]}...'")

    # Run GEPA optimization
    print("\n5. Starting GEPA optimization...")
    print(f"   Reflection LM: {args.reflection_lm}")
    print(f"   Max metric calls: {args.max_metric_calls}")
    print(f"   Candidate selection: {args.candidate_selection}")

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
            run_dir=args.run_dir,
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

        # Save best prompt
        best_prompt_path = os.path.join(args.run_dir, "best_prompt.txt")
        with open(best_prompt_path, "w") as f:
            f.write(result.best_candidate["system_prompt"])
        print(f"\nBest prompt saved to: {best_prompt_path}")

        # Print score progression
        print("\nScore progression:")
        for i, score in enumerate(result.val_aggregate_scores):
            marker = " <-- BEST" if i == result.best_idx else ""
            print(f"  Candidate {i}: {score:.4f}{marker}")

    except Exception as e:
        print(f"\nError during optimization: {e}")
        raise

    print("\nDone!")


if __name__ == "__main__":
    main()

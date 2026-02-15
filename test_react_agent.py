#!/usr/bin/env python3
"""Test the ReAct VQA Agent with DocVQA samples."""

from vqa_react_agent import VQAReActAgent
from vqa_dataset import load_docvqa

def main():
    print("=" * 60)
    print("ReAct VQA Agent Test")
    print("=" * 60)

    # Load DocVQA samples
    print("\nLoading DocVQA samples...")
    data = load_docvqa(split="validation", max_samples=3)

    # Initialize agent
    print("\nInitializing ReAct agent...")
    agent = VQAReActAgent(
        base_url="http://localhost:8000",
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        max_tool_calls=3,
        verbose=True,
    )

    correct = 0
    total = len(data)

    # Test each sample
    for i, sample in enumerate(data):
        print(f"\n{'='*60}")
        print(f"Sample {i+1}/{total}")
        print(f"Question: {sample['question']}")
        print(f"Expected: {sample['answer']}")
        print("-" * 40)

        answer = agent.query(
            image_path=sample['image_path'],
            question=sample['question'],
        )

        is_correct = answer.lower().strip() == sample['answer'].lower().strip()
        if is_correct:
            correct += 1

        print(f"\n>>> Agent answer: {answer}")
        print(f">>> Correct: {is_correct}")

    print(f"\n{'='*60}")
    print(f"Results: {correct}/{total} correct ({100*correct/total:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Test the VQA Agent with DocVQA samples."""

from vqa_agent import VQAAgent
from vqa_dataset import load_docvqa

def main():
    print("=" * 60)
    print("VQA Agent Test")
    print("=" * 60)

    # Load a few DocVQA samples
    print("\nLoading DocVQA samples...")
    data = load_docvqa(split="validation", max_samples=3)

    # Initialize agent
    print("\nInitializing agent...")
    agent = VQAAgent(
        base_url="http://localhost:8000",
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        max_tool_calls=3,
    )

    # Test each sample
    print("\n" + "=" * 60)
    for i, sample in enumerate(data):
        print(f"\n--- Sample {i+1} ---")
        print(f"Question: {sample['question']}")
        print(f"Expected: {sample['answer']}")
        print("\nAgent reasoning:")

        answer = agent.query(
            image_path=sample['image_path'],
            question=sample['question'],
            verbose=True,
        )

        print(f"\nAgent answer: {answer}")
        print(f"Correct: {answer.lower().strip() == sample['answer'].lower().strip()}")
        print("-" * 40)

    print("\nDone!")


if __name__ == "__main__":
    main()

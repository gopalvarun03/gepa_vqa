"""
MMLU Dataset Loading Utilities.

This module provides utilities for loading the MMLU dataset
for use with GEPA optimization.
"""

import os
from typing import TypedDict, Optional
from datasets import load_from_disk, load_dataset


class MMLUDataInst(TypedDict):
    """Data instance format for MMLU examples."""
    question: str
    choices: list[str]
    answer_idx: int           # Index of correct answer (0-3)
    answer: str               # Text of correct answer
    subject: str              # Subject category
    question_id: str          # Unique identifier
    additional_context: dict


def format_choices(choices: list[str]) -> str:
    """Format choices as A, B, C, D options."""
    labels = ['A', 'B', 'C', 'D']
    return '\n'.join(f"{labels[i]}. {choice}" for i, choice in enumerate(choices))


def parse_answer(response: str, choices: list[str]) -> int:
    """
    Parse model response to get answer index.

    Returns:
        Index (0-3) of the selected answer, or -1 if not found
    """
    response = response.strip().upper()

    # Check for letter answer (A, B, C, D)
    for i, label in enumerate(['A', 'B', 'C', 'D']):
        if response.startswith(label) or response == label:
            return i

    # Check if response matches choice text
    response_lower = response.lower()
    for i, choice in enumerate(choices):
        if choice.lower() in response_lower or response_lower in choice.lower():
            return i

    return -1


def compute_mmlu_accuracy(response: str, answer_idx: int, choices: list[str]) -> float:
    """
    Compute accuracy for MMLU response.

    Args:
        response: Model's response
        answer_idx: Index of correct answer (0-3)
        choices: List of choice texts

    Returns:
        1.0 if correct, 0.0 otherwise
    """
    predicted_idx = parse_answer(response, choices)
    return 1.0 if predicted_idx == answer_idx else 0.0


def load_mmlu(
    split: str = "test",
    subjects: Optional[list[str]] = None,
    max_samples: Optional[int] = None,
    data_path: Optional[str] = None,
) -> list[MMLUDataInst]:
    """
    Load MMLU dataset.

    Args:
        split: Dataset split ("test", "validation", "dev", "auxiliary_train")
        subjects: Optional list of subjects to filter by
        max_samples: Maximum number of samples to load
        data_path: Path to saved dataset (if None, loads from HuggingFace)

    Returns:
        List of MMLUDataInst dictionaries
    """
    print(f"Loading MMLU {split} split...")

    # Load dataset
    if data_path and os.path.exists(data_path):
        dataset = load_from_disk(data_path)
        data = dataset[split]
    else:
        dataset = load_dataset("cais/mmlu", "all", split=split)
        data = dataset

    # Filter by subjects if specified
    if subjects:
        data = data.filter(lambda x: x['subject'] in subjects)
        print(f"  Filtered to subjects: {subjects}")

    if max_samples:
        data = data.select(range(min(max_samples, len(data))))

    data_instances = []
    for idx, item in enumerate(data):
        choices = item['choices']
        answer_idx = item['answer']

        data_inst = MMLUDataInst(
            question=item['question'],
            choices=choices,
            answer_idx=answer_idx,
            answer=choices[answer_idx],
            subject=item['subject'],
            question_id=f"{item['subject']}_{idx}",
            additional_context={
                "dataset": "mmlu",
                "split": split,
            },
        )
        data_instances.append(data_inst)

    print(f"Loaded {len(data_instances)} MMLU examples.")
    return data_instances


def get_mmlu_subjects(data_path: Optional[str] = None) -> list[str]:
    """Get list of all MMLU subjects."""
    if data_path and os.path.exists(data_path):
        dataset = load_from_disk(data_path)
        data = dataset['test']
    else:
        data = load_dataset("cais/mmlu", "all", split="test")

    subjects = list(set(data['subject']))
    return sorted(subjects)


if __name__ == "__main__":
    # Test the loader
    data_path = "/home/shivank_g/projects/vlm_reason/varun_codes/vqa_datasets/mmlu"

    print("Available subjects:")
    subjects = get_mmlu_subjects(data_path)
    print(f"  {len(subjects)} subjects: {subjects[:5]}...")

    print("\nLoading sample data...")
    data = load_mmlu(split="test", max_samples=5, data_path=data_path)

    for item in data:
        print(f"\nSubject: {item['subject']}")
        print(f"Question: {item['question'][:100]}...")
        print(f"Choices: {item['choices']}")
        print(f"Answer: {item['answer']} (index {item['answer_idx']})")

"""
VQA Dataset Loading Utilities.

This module provides utilities for loading VQA datasets in formats
compatible with GEPA optimization.

Supported datasets:
- VQAv2
- TextVQA
- Custom JSONL format
"""

import json
import os
from pathlib import Path
from typing import TypedDict, Optional
from collections import Counter

from datasets import load_dataset


class VQADataInst(TypedDict):
    """
    Data instance format for VQA examples.

    This is the DataInst type used by the GEPA adapter.
    """
    image_path: str          # Path to the image file
    question: str            # Question about the image
    answer: str              # Ground truth answer (most common or first)
    answers: list[str]       # All ground truth answers (for VQA accuracy)
    question_id: str         # Unique identifier for the question
    additional_context: dict # Extra metadata (dataset name, etc.)


def normalize_answer(answer: str) -> str:
    """
    Normalize an answer string for comparison.

    Args:
        answer: Raw answer string

    Returns:
        Normalized answer (lowercase, stripped)
    """
    return answer.lower().strip()


def compute_vqa_accuracy(prediction: str, ground_truth_answers: list[str]) -> float:
    """
    Compute VQA accuracy score.

    For VQAv2, the accuracy is: min(1, # humans that gave that answer / 3)
    This is approximated by checking if the prediction matches any of the answers.

    Args:
        prediction: Model's predicted answer
        ground_truth_answers: List of human-provided answers

    Returns:
        Accuracy score between 0.0 and 1.0
    """
    pred_norm = normalize_answer(prediction)
    answer_counts = Counter(normalize_answer(a) for a in ground_truth_answers)

    if pred_norm in answer_counts:
        # VQA accuracy formula: min(1, count / 3)
        count = answer_counts[pred_norm]
        return min(1.0, count / 3.0)

    # Check for substring match as fallback
    for ans in answer_counts:
        if pred_norm in ans or ans in pred_norm:
            return 0.5  # Partial credit for substring match

    return 0.0


def load_vqav2(
    split: str = "validation",
    image_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> list[VQADataInst]:
    """
    Load VQAv2 dataset from HuggingFace.

    Args:
        split: Dataset split ("train", "validation", "test")
        image_dir: Directory containing COCO images. If None, will use
                   the default path or download.
        max_samples: Maximum number of samples to load (for testing)

    Returns:
        List of VQADataInst dictionaries
    """
    print(f"Loading VQAv2 {split} split...")

    # Load from HuggingFace
    dataset = load_dataset("HuggingFaceM4/VQAv2", split=split)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    data_instances = []
    for item in dataset:
        # VQAv2 has multiple answers per question
        answers = item.get("answers", [item.get("answer", "")])
        if isinstance(answers, str):
            answers = [answers]

        # Get most common answer
        answer_counts = Counter(answers)
        most_common = answer_counts.most_common(1)[0][0] if answer_counts else ""

        # Construct image path
        image_id = str(item.get("image_id", item.get("id", "")))
        if image_dir:
            # VQAv2 images follow COCO naming convention
            image_path = os.path.join(image_dir, f"COCO_val2014_{image_id.zfill(12)}.jpg")
        else:
            # Use image directly if available
            image_path = item.get("image", {}).get("path", f"image_{image_id}.jpg")

        data_inst = VQADataInst(
            image_path=image_path,
            question=item["question"],
            answer=most_common,
            answers=answers,
            question_id=str(item.get("question_id", image_id)),
            additional_context={"dataset": "vqav2", "split": split},
        )
        data_instances.append(data_inst)

    print(f"Loaded {len(data_instances)} VQAv2 examples.")
    return data_instances


def load_docvqa(
    split: str = "validation",
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> list[VQADataInst]:
    """
    Load DocVQA dataset from HuggingFace.

    Note: lmms-lab/DocVQA only has 'validation' and 'test' splits.
    For 'train', we use 'validation' split.

    Args:
        split: Dataset split ("train", "validation", "test")
        max_samples: Maximum number of samples to load
        cache_dir: Directory to cache images (if None, keeps PIL images in memory)

    Returns:
        List of VQADataInst dictionaries (with PIL images stored directly)
    """
    # lmms-lab/DocVQA only has validation and test splits
    # Map train to validation for this dataset
    split_map = {"train": "validation", "validation": "validation", "val": "validation", "test": "test"}
    hf_split = split_map.get(split, "validation")

    print(f"Loading DocVQA {hf_split} split (requested: {split})...")

    # Load from HuggingFace
    dataset = load_dataset("lmms-lab/DocVQA", "DocVQA", split=hf_split)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    data_instances = []
    for idx, item in enumerate(dataset):
        answers = item.get("answers", [])
        if isinstance(answers, str):
            answers = [answers]

        # Get most common answer
        answer_counts = Counter(answers)
        most_common = answer_counts.most_common(1)[0][0] if answer_counts else ""

        # Store PIL image directly (OpenRouter wrapper can handle it)
        pil_image = item["image"]

        # Optionally save to cache directory
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            image_path = os.path.join(cache_dir, f"docvqa_{item['questionId']}.png")
            if not os.path.exists(image_path):
                pil_image.save(image_path)
            image_source = image_path
        else:
            image_source = pil_image  # Store PIL image directly

        data_inst = VQADataInst(
            image_path=image_source,  # Can be path or PIL Image
            question=item["question"],
            answer=most_common,
            answers=answers,
            question_id=str(item.get("questionId", idx)),
            additional_context={
                "dataset": "docvqa",
                "split": split,
                "question_types": item.get("question_types", []),
            },
        )
        data_instances.append(data_inst)

    print(f"Loaded {len(data_instances)} DocVQA examples.")
    return data_instances


def load_textvqa(
    split: str = "validation",
    image_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> list[VQADataInst]:
    """
    Load TextVQA dataset from HuggingFace.

    Args:
        split: Dataset split ("train", "validation", "test")
        image_dir: Directory containing TextVQA images
        max_samples: Maximum number of samples to load

    Returns:
        List of VQADataInst dictionaries
    """
    print(f"Loading TextVQA {split} split...")

    # Load from HuggingFace
    dataset = load_dataset("textvqa", split=split)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    data_instances = []
    for item in dataset:
        answers = item.get("answers", [item.get("answer", "")])
        if isinstance(answers, str):
            answers = [answers]

        answer_counts = Counter(answers)
        most_common = answer_counts.most_common(1)[0][0] if answer_counts else ""

        # Construct image path
        image_id = item.get("image_id", str(item.get("question_id", "")))
        if image_dir:
            image_path = os.path.join(image_dir, f"{image_id}.jpg")
        else:
            image_path = item.get("image_path", f"image_{image_id}.jpg")

        data_inst = VQADataInst(
            image_path=image_path,
            question=item["question"],
            answer=most_common,
            answers=answers,
            question_id=str(item.get("question_id", image_id)),
            additional_context={"dataset": "textvqa", "split": split},
        )
        data_instances.append(data_inst)

    print(f"Loaded {len(data_instances)} TextVQA examples.")
    return data_instances


def load_jsonl(
    jsonl_path: str,
    image_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> list[VQADataInst]:
    """
    Load VQA data from a JSONL file.

    Expected JSONL format (one JSON object per line):
    {
        "image": "image_path.jpg",
        "question": "What is this?",
        "answer": "a cat",
        "question_id": "123"  # optional
    }

    Args:
        jsonl_path: Path to the JSONL file
        image_dir: Base directory for relative image paths
        max_samples: Maximum number of samples to load

    Returns:
        List of VQADataInst dictionaries
    """
    print(f"Loading JSONL dataset from {jsonl_path}...")

    data_instances = []
    with open(jsonl_path, "r") as f:
        for idx, line in enumerate(f):
            if max_samples and idx >= max_samples:
                break

            item = json.loads(line.strip())

            # Get image path
            image_path = item.get("image", item.get("image_path", ""))
            if image_dir and not os.path.isabs(image_path):
                image_path = os.path.join(image_dir, image_path)

            # Get answers
            answer = item.get("answer", "")
            answers = item.get("answers", [answer])
            if isinstance(answers, str):
                answers = [answers]

            data_inst = VQADataInst(
                image_path=image_path,
                question=item["question"],
                answer=answer,
                answers=answers,
                question_id=str(item.get("question_id", idx)),
                additional_context={"dataset": "custom", "source": jsonl_path},
            )
            data_instances.append(data_inst)

    print(f"Loaded {len(data_instances)} examples from JSONL.")
    return data_instances


def load_vqa_dataset(
    dataset_name: str,
    split: str = "validation",
    image_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
    **kwargs,
) -> list[VQADataInst]:
    """
    Load a VQA dataset by name.

    Args:
        dataset_name: Name of the dataset ("vqav2", "textvqa") or path to JSONL
        split: Dataset split
        image_dir: Directory containing images
        max_samples: Maximum number of samples to load
        **kwargs: Additional arguments passed to the loader

    Returns:
        List of VQADataInst dictionaries
    """
    dataset_name_lower = dataset_name.lower()

    if dataset_name_lower == "vqav2":
        return load_vqav2(split=split, image_dir=image_dir, max_samples=max_samples)
    elif dataset_name_lower == "textvqa":
        return load_textvqa(split=split, image_dir=image_dir, max_samples=max_samples)
    elif dataset_name_lower == "docvqa":
        cache_dir = kwargs.get("cache_dir", image_dir)
        return load_docvqa(split=split, max_samples=max_samples, cache_dir=cache_dir)
    elif os.path.exists(dataset_name):
        # Assume it's a path to a JSONL file
        return load_jsonl(jsonl_path=dataset_name, image_dir=image_dir, max_samples=max_samples)
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            "Supported: 'vqav2', 'textvqa', 'docvqa', or path to JSONL file."
        )


def create_mock_dataset(num_samples: int = 10) -> list[VQADataInst]:
    """
    Create a mock dataset for testing.

    Args:
        num_samples: Number of mock samples to create

    Returns:
        List of mock VQADataInst dictionaries
    """
    mock_data = [
        {"question": "What color is the car?", "answer": "red"},
        {"question": "How many people are there?", "answer": "3"},
        {"question": "What is the animal?", "answer": "dog"},
        {"question": "Is it sunny?", "answer": "yes"},
        {"question": "What is the person wearing?", "answer": "hat"},
        {"question": "What time of day is it?", "answer": "morning"},
        {"question": "Where is this photo taken?", "answer": "beach"},
        {"question": "What sport is being played?", "answer": "tennis"},
        {"question": "What food is on the table?", "answer": "pizza"},
        {"question": "What color is the sky?", "answer": "blue"},
    ]

    data_instances = []
    for i in range(num_samples):
        item = mock_data[i % len(mock_data)]
        data_inst = VQADataInst(
            image_path=f"mock_image_{i}.jpg",
            question=item["question"],
            answer=item["answer"],
            answers=[item["answer"]],
            question_id=str(i),
            additional_context={"dataset": "mock"},
        )
        data_instances.append(data_inst)

    return data_instances


if __name__ == "__main__":
    # Test the dataset loaders
    print("Testing mock dataset...")
    mock = create_mock_dataset(5)
    for item in mock:
        print(f"  Q: {item['question']} -> A: {item['answer']}")

    print("\nTesting VQA accuracy computation...")
    test_cases = [
        ("cat", ["cat", "cat", "cat"], 1.0),
        ("cat", ["cat", "dog", "bird"], 0.333),
        ("cats", ["cat", "cat", "cat"], 0.5),  # Partial match
        ("dog", ["cat", "cat", "cat"], 0.0),
    ]
    for pred, answers, expected in test_cases:
        score = compute_vqa_accuracy(pred, answers)
        print(f"  predict='{pred}', answers={answers} -> score={score:.3f} (expected {expected:.3f})")

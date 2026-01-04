#!/usr/bin/env python3
"""
Download and analyze DocVQA dataset from HuggingFace.

This script downloads the DocVQA dataset, saves it locally,
and provides detailed analysis of its structure.
"""

import os
import json
from pathlib import Path
from collections import Counter
from datasets import load_dataset
import pickle

# Dataset paths
BASE_DIR = Path(__file__).parent.parent / "vqa_datasets" / "DocVQA"
BASE_DIR.mkdir(parents=True, exist_ok=True)


def download_and_save_dataset():
    """Download DocVQA dataset from HuggingFace and save locally."""
    print("=" * 60)
    print("Downloading DocVQA Dataset")
    print("=" * 60)

    # Load dataset from HuggingFace
    print("\n1. Loading dataset from HuggingFace...")
    print("   Dataset: lmms-lab/DocVQA")
    print("   Config: DocVQA")

    # DocVQA has train, validation, and test splits
    dataset = load_dataset("lmms-lab/DocVQA", "DocVQA")

    print(f"\n2. Dataset loaded successfully!")
    print(f"   Available splits: {list(dataset.keys())}")

    # Print split sizes
    for split_name, split_data in dataset.items():
        print(f"   - {split_name}: {len(split_data)} examples")

    # Save dataset locally
    print(f"\n3. Saving dataset to {BASE_DIR}...")

    for split_name, split_data in dataset.items():
        split_dir = BASE_DIR / split_name
        split_dir.mkdir(exist_ok=True)

        print(f"\n   Processing {split_name} split...")

        # Save as pickle for fast loading
        pickle_path = split_dir / "data.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(split_data, f)
        print(f"   ✓ Saved pickle: {pickle_path}")

        # Save as JSONL for easy inspection
        jsonl_path = split_dir / "data.jsonl"
        with open(jsonl_path, "w") as f:
            for idx, example in enumerate(split_data):
                # Convert to JSON-serializable format
                image_info = ""
                if "image" in example:
                    img = example["image"]
                    if hasattr(img, 'filename'):
                        image_info = img.filename
                    elif hasattr(img, 'size'):
                        image_info = f"image_{example.get('questionId', idx)}.png"

                json_example = {
                    "question_id": example.get("questionId", idx),
                    "question": example.get("question", ""),
                    "answers": example.get("answers", []),
                    "data_split": example.get("data_split", ""),
                    "image_filename": image_info,
                }
                f.write(json.dumps(json_example) + "\n")
        print(f"   ✓ Saved JSONL: {jsonl_path}")

        # Save images metadata
        images_dir = split_dir / "images"
        images_dir.mkdir(exist_ok=True)

        # Create image index
        image_paths = []
        for idx, example in enumerate(split_data):
            if "image" in example:
                img = example["image"]
                # Save image if it's a PIL Image
                if hasattr(img, 'save'):
                    img_filename = f"{example.get('questionId', idx)}.png"
                    img_path = images_dir / img_filename
                    img.save(img_path)
                    image_paths.append(str(img_path))

                    if (idx + 1) % 100 == 0:
                        print(f"   Saved {idx + 1} images...")

        print(f"   ✓ Saved {len(image_paths)} images to {images_dir}")

    print("\n" + "=" * 60)
    print("Dataset download complete!")
    print("=" * 60)

    return dataset


def analyze_dataset(dataset):
    """Analyze the dataset structure in detail."""
    print("\n" + "=" * 60)
    print("Dataset Structure Analysis")
    print("=" * 60)

    # Get first example from validation split for analysis
    val_split = dataset.get("validation") or dataset.get("val") or list(dataset.values())[0]
    first_example = val_split[0]

    print("\n1. DATASET FEATURES")
    print("-" * 60)
    print(f"Feature names: {list(first_example.keys())}")

    # Detailed feature analysis
    print("\n2. FEATURE DETAILS")
    print("-" * 60)

    for key, value in first_example.items():
        print(f"\n   [{key}]")
        print(f"   Type: {type(value)}")

        if isinstance(value, str):
            print(f"   Sample: '{value[:100]}...' " if len(value) > 100 else f"   Sample: '{value}'")
        elif isinstance(value, list):
            print(f"   Length: {len(value)}")
            if value:
                print(f"   First item type: {type(value[0])}")
                print(f"   Sample: {value}")
        elif isinstance(value, dict):
            print(f"   Keys: {list(value.keys())}")
            print(f"   Sample: {value}")
        elif hasattr(value, 'size'):  # PIL Image
            print(f"   Image size: {value.size}")
            print(f"   Image mode: {value.mode}")
        else:
            print(f"   Value: {value}")

    # Statistics across dataset
    print("\n3. DATASET STATISTICS")
    print("-" * 60)

    for split_name, split_data in dataset.items():
        print(f"\n   [{split_name.upper()} SPLIT]")
        print(f"   Total examples: {len(split_data)}")

        # Question length statistics
        question_lengths = [len(ex["question"].split()) for ex in split_data]
        print(f"   Question length (words):")
        print(f"     - Min: {min(question_lengths)}")
        print(f"     - Max: {max(question_lengths)}")
        print(f"     - Avg: {sum(question_lengths) / len(question_lengths):.1f}")

        # Answer statistics
        all_answers = []
        answer_counts = []
        for ex in split_data:
            answers = ex.get("answers", [])
            if answers:
                all_answers.extend(answers)
                answer_counts.append(len(answers))

        if answer_counts:
            print(f"   Answers per question:")
            print(f"     - Min: {min(answer_counts)}")
            print(f"     - Max: {max(answer_counts)}")
            print(f"     - Avg: {sum(answer_counts) / len(answer_counts):.1f}")

        if all_answers:
            # Answer length statistics
            answer_lengths = [len(str(ans).split()) for ans in all_answers]
            print(f"   Answer length (words):")
            print(f"     - Min: {min(answer_lengths)}")
            print(f"     - Max: {max(answer_lengths)}")
            print(f"     - Avg: {sum(answer_lengths) / len(answer_lengths):.1f}")

            # Most common answers
            answer_counter = Counter(all_answers)
            print(f"   Most common answers:")
            for ans, count in answer_counter.most_common(10):
                print(f"     - '{ans}': {count} times")

    # Sample examples
    print("\n4. SAMPLE EXAMPLES")
    print("-" * 60)

    for i in range(min(3, len(val_split))):
        example = val_split[i]
        print(f"\n   Example {i + 1}:")
        print(f"   Question ID: {example.get('questionId', 'N/A')}")
        print(f"   Question: {example.get('question', 'N/A')}")
        print(f"   Answers: {example.get('answers', 'N/A')}")
        if 'image' in example:
            img = example['image']
            if hasattr(img, 'size'):
                print(f"   Image: {img.size} pixels, {img.mode} mode")

    print("\n" + "=" * 60)


def save_analysis_report(dataset):
    """Save detailed analysis report to file."""
    report_path = BASE_DIR / "dataset_analysis.txt"

    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("DocVQA Dataset Analysis Report\n")
        f.write("=" * 60 + "\n\n")

        f.write("DATASET OVERVIEW\n")
        f.write("-" * 60 + "\n")
        f.write(f"Source: lmms-lab/DocVQA (HuggingFace)\n")
        f.write(f"Splits: {list(dataset.keys())}\n\n")

        for split_name, split_data in dataset.items():
            f.write(f"{split_name.upper()}: {len(split_data)} examples\n")

        f.write("\n")
        f.write("FEATURES\n")
        f.write("-" * 60 + "\n")

        val_split = dataset.get("validation") or list(dataset.values())[0]
        first_example = val_split[0]

        for key in first_example.keys():
            f.write(f"- {key}\n")

        f.write("\n")
        f.write("SAVED FILES\n")
        f.write("-" * 60 + "\n")
        f.write(f"Location: {BASE_DIR}\n")
        f.write(f"Format: JSONL + Pickle + Images\n")

    print(f"\n✓ Analysis report saved to: {report_path}")


def main():
    """Main execution function."""
    # Download and save dataset
    dataset = download_and_save_dataset()

    # Analyze dataset
    analyze_dataset(dataset)

    # Save analysis report
    save_analysis_report(dataset)

    print("\n" + "=" * 60)
    print("All operations completed successfully!")
    print(f"Dataset location: {BASE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

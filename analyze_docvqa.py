#!/usr/bin/env python3
"""
Analyze DocVQA dataset structure without downloading all images.

This script loads the DocVQA dataset, analyzes its structure,
and saves metadata while keeping images in HuggingFace cache.
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


def analyze_dataset_structure():
    """Load and analyze DocVQA dataset structure."""
    print("=" * 70)
    print("DocVQA Dataset Structure Analysis")
    print("=" * 70)

    # Load dataset from HuggingFace
    print("\n1. Loading dataset from HuggingFace...")
    print("   Dataset: lmms-lab/DocVQA")
    print("   Config: DocVQA")

    dataset = load_dataset("lmms-lab/DocVQA", "DocVQA")

    print(f"\n2. Dataset loaded successfully!")
    print(f"   Available splits: {list(dataset.keys())}")

    # Print split sizes
    for split_name, split_data in dataset.items():
        print(f"   - {split_name}: {len(split_data)} examples")

    # Analyze each split
    print("\n" + "=" * 70)
    print("DETAILED FEATURE ANALYSIS")
    print("=" * 70)

    for split_name, split_data in dataset.items():
        print(f"\n{'=' * 70}")
        print(f"{split_name.upper()} SPLIT")
        print(f"{'=' * 70}")

        # Get first example for structure analysis
        if len(split_data) == 0:
            print("   (Empty split)")
            continue

        first_example = split_data[0]

        print(f"\n1. Available Features/Columns:")
        print("-" * 70)
        for key in first_example.keys():
            print(f"   - {key}")

        print(f"\n2. Feature Details:")
        print("-" * 70)

        for key, value in first_example.items():
            print(f"\n   [{key}]")
            print(f"   Type: {type(value).__name__}")

            if isinstance(value, str):
                print(f"   Length: {len(value)} characters")
                preview = value[:150] + "..." if len(value) > 150 else value
                print(f"   Sample: '{preview}'")

            elif isinstance(value, list):
                print(f"   Length: {len(value)} items")
                if value:
                    print(f"   First item type: {type(value[0]).__name__}")
                    print(f"   Sample items: {value[:5]}")

            elif isinstance(value, dict):
                print(f"   Dictionary keys: {list(value.keys())}")
                for k, v in list(value.items())[:3]:
                    print(f"     - {k}: {type(v).__name__} = {v}")

            elif hasattr(value, 'size'):  # PIL Image
                print(f"   Image size: {value.size} pixels")
                print(f"   Image mode: {value.mode}")

            elif isinstance(value, (int, float, bool)):
                print(f"   Value: {value}")

            else:
                print(f"   Preview: {str(value)[:100]}")

        # Statistical analysis
        print(f"\n3. Dataset Statistics:")
        print("-" * 70)

        print(f"   Total examples: {len(split_data)}")

        # Question length stats
        questions = [ex["question"] for ex in split_data if "question" in ex]
        if questions:
            q_lengths = [len(q.split()) for q in questions]
            q_char_lengths = [len(q) for q in questions]

            print(f"\n   Question Statistics:")
            print(f"     Word count:")
            print(f"       - Min: {min(q_lengths)} words")
            print(f"       - Max: {max(q_lengths)} words")
            print(f"       - Average: {sum(q_lengths) / len(q_lengths):.1f} words")
            print(f"       - Median: {sorted(q_lengths)[len(q_lengths) // 2]} words")
            print(f"     Character count:")
            print(f"       - Min: {min(q_char_lengths)} chars")
            print(f"       - Max: {max(q_char_lengths)} chars")
            print(f"       - Average: {sum(q_char_lengths) / len(q_char_lengths):.1f} chars")

        # Answer statistics
        all_answers = []
        answers_per_question = []
        for ex in split_data:
            answers = ex.get("answers", [])
            if answers:
                all_answers.extend(answers)
                answers_per_question.append(len(answers))

        if all_answers:
            ans_lengths_words = [len(str(ans).split()) for ans in all_answers]
            ans_lengths_chars = [len(str(ans)) for ans in all_answers]

            print(f"\n   Answer Statistics:")
            print(f"     Answers per question:")
            print(f"       - Min: {min(answers_per_question)}")
            print(f"       - Max: {max(answers_per_question)}")
            print(f"       - Average: {sum(answers_per_question) / len(answers_per_question):.1f}")

            print(f"     Answer length (words):")
            print(f"       - Min: {min(ans_lengths_words)} words")
            print(f"       - Max: {max(ans_lengths_words)} words")
            print(f"       - Average: {sum(ans_lengths_words) / len(ans_lengths_words):.1f} words")

            print(f"     Answer length (characters):")
            print(f"       - Min: {min(ans_lengths_chars)} chars")
            print(f"       - Max: {max(ans_lengths_chars)} chars")
            print(f"       - Average: {sum(ans_lengths_chars) / len(ans_lengths_chars):.1f} chars")

            # Most common answers
            answer_counts = Counter(all_answers)
            print(f"\n     Most common answers (Top 15):")
            for i, (ans, count) in enumerate(answer_counts.most_common(15), 1):
                percentage = (count / len(all_answers)) * 100
                print(f"       {i:2}. '{ans}' - {count} times ({percentage:.2f}%)")

        # Image statistics (check first few)
        images = [ex["image"] for ex in split_data[:100] if "image" in ex]
        if images:
            img_sizes = [img.size for img in images if hasattr(img, 'size')]
            img_modes = [img.mode for img in images if hasattr(img, 'mode')]

            print(f"\n   Image Statistics (sample of 100):")
            if img_sizes:
                widths = [size[0] for size in img_sizes]
                heights = [size[1] for size in img_sizes]

                print(f"     Width:")
                print(f"       - Min: {min(widths)}px")
                print(f"       - Max: {max(widths)}px")
                print(f"       - Average: {sum(widths) / len(widths):.1f}px")

                print(f"     Height:")
                print(f"       - Min: {min(heights)}px")
                print(f"       - Max: {max(heights)}px")
                print(f"       - Average: {sum(heights) / len(heights):.1f}px")

            if img_modes:
                mode_counts = Counter(img_modes)
                print(f"     Image modes: {dict(mode_counts)}")

        # Sample examples
        print(f"\n4. Sample Examples:")
        print("-" * 70)

        num_samples = min(5, len(split_data))
        for i in range(num_samples):
            ex = split_data[i]
            print(f"\n   Example {i + 1}:")
            print(f"   -----------")

            for key, value in ex.items():
                if key == "image":
                    if hasattr(value, 'size'):
                        print(f"     {key}: PIL Image ({value.size}, {value.mode})")
                elif isinstance(value, str):
                    preview = value[:100] + "..." if len(value) > 100 else value
                    print(f"     {key}: '{preview}'")
                elif isinstance(value, list):
                    print(f"     {key}: {value}")
                elif isinstance(value, dict):
                    print(f"     {key}: {value}")
                else:
                    print(f"     {key}: {value}")

    return dataset


def save_metadata_and_samples(dataset):
    """Save dataset metadata and sample images."""
    print("\n" + "=" * 70)
    print("SAVING METADATA AND SAMPLES")
    print("=" * 70)

    for split_name, split_data in dataset.items():
        split_dir = BASE_DIR / split_name
        split_dir.mkdir(exist_ok=True)

        print(f"\nProcessing {split_name} split...")

        # Save metadata as JSONL (without images)
        metadata_path = split_dir / "metadata.jsonl"
        with open(metadata_path, "w") as f:
            for idx, ex in enumerate(split_data):
                metadata = {
                    "index": idx,
                    "question_id": ex.get("questionId", ""),
                    "question": ex.get("question", ""),
                    "answers": ex.get("answers", []),
                    "data_split": ex.get("data_split", ""),
                }

                # Add image info if available
                if "image" in ex and hasattr(ex["image"], 'size'):
                    metadata["image_size"] = ex["image"].size
                    metadata["image_mode"] = ex["image"].mode

                f.write(json.dumps(metadata) + "\n")

        print(f"   ✓ Saved metadata: {metadata_path}")

        # Save a few sample images
        samples_dir = split_dir / "sample_images"
        samples_dir.mkdir(exist_ok=True)

        num_samples = min(10, len(split_data))
        for i in range(num_samples):
            ex = split_data[i]
            if "image" in ex and hasattr(ex["image"], 'save'):
                img_path = samples_dir / f"sample_{i:03d}.png"
                ex["image"].save(img_path)

        print(f"   ✓ Saved {num_samples} sample images to: {samples_dir}")

    print(f"\n✓ All metadata and samples saved to: {BASE_DIR}")


def save_analysis_report(dataset):
    """Save comprehensive analysis report."""
    report_path = BASE_DIR / "DATASET_ANALYSIS.md"

    with open(report_path, "w") as f:
        f.write("# DocVQA Dataset Analysis\n\n")
        f.write("## Dataset Overview\n\n")
        f.write("**Source**: [lmms-lab/DocVQA](https://huggingface.co/datasets/lmms-lab/DocVQA)\n\n")
        f.write("**Task**: Document Visual Question Answering\n\n")
        f.write("**Description**: DocVQA is a dataset for Visual Question Answering on document images. ")
        f.write("Questions are asked about the content of document images (forms, receipts, invoices, etc.).\n\n")

        f.write("### Splits\n\n")
        for split_name, split_data in dataset.items():
            f.write(f"- **{split_name}**: {len(split_data):,} examples\n")

        f.write("\n### Features\n\n")

        # Get structure from first split
        first_split = list(dataset.values())[0]
        if len(first_split) > 0:
            example = first_split[0]

            f.write("| Feature | Type | Description |\n")
            f.write("|---------|------|-------------|\n")

            for key in example.keys():
                value = example[key]
                type_name = type(value).__name__

                if key == "questionId":
                    desc = "Unique identifier for the question"
                elif key == "question":
                    desc = "The question text"
                elif key == "answers":
                    desc = "List of ground truth answers"
                elif key == "data_split":
                    desc = "Original data split name"
                elif key == "image":
                    desc = "Document image (PIL Image)"
                else:
                    desc = ""

                f.write(f"| `{key}` | {type_name} | {desc} |\n")

        f.write("\n### Example\n\n")
        f.write("```python\n")
        if len(first_split) > 0:
            ex = first_split[0]
            f.write("{\n")
            for key, value in ex.items():
                if key == "image":
                    if hasattr(value, 'size'):
                        f.write(f"  '{key}': PIL.Image ({value.size}),\n")
                elif isinstance(value, str):
                    preview = value[:80] + "..." if len(value) > 80 else value
                    f.write(f"  '{key}': '{preview}',\n")
                else:
                    f.write(f"  '{key}': {value},\n")
            f.write("}\n")
        f.write("```\n\n")

        f.write("## Usage Notes\n\n")
        f.write("1. **Images**: Document images are stored as PIL Images in HuggingFace cache\n")
        f.write("2. **Answers**: Multiple ground truth answers per question\n")
        f.write("3. **Format**: Documents include forms, invoices, receipts, and other structured documents\n\n")

        f.write("## Loading the Dataset\n\n")
        f.write("```python\n")
        f.write("from datasets import load_dataset\n\n")
        f.write('dataset = load_dataset("lmms-lab/DocVQA", "DocVQA")\n')
        f.write("validation = dataset['validation']\n")
        f.write("test = dataset['test']\n")
        f.write("```\n")

    print(f"\n✓ Analysis report saved to: {report_path}")


def main():
    """Main execution function."""
    # Analyze dataset structure
    dataset = analyze_dataset_structure()

    # Save metadata and samples
    save_metadata_and_samples(dataset)

    # Save analysis report
    save_analysis_report(dataset)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\nDataset metadata saved to: {BASE_DIR}")
    print(f"Images are available in HuggingFace cache")
    print(f"Load with: load_dataset('lmms-lab/DocVQA', 'DocVQA')")
    print("=" * 70)


if __name__ == "__main__":
    main()

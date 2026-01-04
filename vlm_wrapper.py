"""
InternVL Wrapper for VQA Inference.

This module provides a clean interface to InternVL for VQA tasks,
handling model loading, image preprocessing, and inference.
"""

import sys
import os
from typing import Optional
from pathlib import Path

import torch
from PIL import Image

# Add InternVL to path
INTERNVL_PATH = Path(__file__).parent.parent / "InternVL" / "internvl_chat"
sys.path.insert(0, str(INTERNVL_PATH))

from transformers import AutoModel, AutoTokenizer
from internvl.train.dataset import build_transform, dynamic_preprocess


class InternVLWrapper:
    """
    Wrapper for InternVL model to perform VQA inference.

    Usage:
        vlm = InternVLWrapper(model_path="path/to/InternVL3_5-8B")
        answer = vlm.query(
            image_path="image.jpg",
            question="What is in this image?",
            system_prompt="Answer briefly."
        )
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        use_flash_attn: bool = True,
        max_tiles: int = 12,
        image_size: int = 448,
        max_new_tokens: int = 100,
    ):
        """
        Initialize the InternVL wrapper.

        Args:
            model_path: Path to the InternVL model checkpoint
            device: Device to load model on ("cuda" or "cpu")
            dtype: Data type for model weights
            use_flash_attn: Whether to use flash attention
            max_tiles: Maximum number of image tiles (for dynamic resolution)
            image_size: Size to resize image tiles to
            max_new_tokens: Default max tokens for generation
        """
        self.device = device
        self.dtype = dtype
        self.max_tiles = max_tiles
        self.image_size = image_size
        self.max_new_tokens = max_new_tokens

        print(f"Loading InternVL model from {model_path}...")

        # Load model
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_flash_attn=use_flash_attn,
            trust_remote_code=True,
        ).to(device).eval()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
        )

        # Build image transform
        self.transform = build_transform(is_train=False, input_size=image_size)

        print("InternVL model loaded successfully.")

    def _load_image(self, image_path: str) -> torch.Tensor:
        """
        Load and preprocess an image for the model.

        Args:
            image_path: Path to the image file

        Returns:
            Tensor of preprocessed image patches
        """
        image = Image.open(image_path).convert("RGB")

        # Dynamic preprocessing - splits image into tiles
        images = dynamic_preprocess(
            image,
            image_size=self.image_size,
            use_thumbnail=True,
            max_num=self.max_tiles,
        )

        # Apply transforms and stack
        pixel_values = torch.stack([self.transform(img) for img in images])
        pixel_values = pixel_values.to(self.dtype).to(self.device)

        return pixel_values

    def query(
        self,
        image_path: str,
        question: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        do_sample: bool = False,
        temperature: float = 0.0,
    ) -> str:
        """
        Query the VLM with an image and question.

        Args:
            image_path: Path to the image file
            question: Question to ask about the image
            system_prompt: Optional system prompt to prepend
            max_new_tokens: Maximum tokens to generate (defaults to self.max_new_tokens)
            do_sample: Whether to sample (False = greedy decoding)
            temperature: Sampling temperature (only used if do_sample=True)

        Returns:
            Model's text response
        """
        # Load and preprocess image
        pixel_values = self._load_image(image_path)

        # Construct the full prompt
        if system_prompt:
            full_question = f"{system_prompt}\n\n{question}"
        else:
            full_question = question

        # Generation config
        generation_config = {
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            generation_config["temperature"] = temperature

        # Run inference
        with torch.no_grad():
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                full_question,
                generation_config=generation_config,
                num_patches_list=[pixel_values.shape[0]],
                history=None,
                return_history=False,
            )

        return response

    def batch_query(
        self,
        image_paths: list[str],
        questions: list[str],
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> list[str]:
        """
        Query the VLM with multiple image-question pairs.

        Note: This processes sequentially as InternVL's chat() doesn't support
        true batching. For better performance with large batches, consider
        using distributed inference.

        Args:
            image_paths: List of paths to image files
            questions: List of questions (same length as image_paths)
            system_prompt: Optional system prompt to prepend to all questions
            max_new_tokens: Maximum tokens to generate

        Returns:
            List of model responses
        """
        assert len(image_paths) == len(questions), \
            "image_paths and questions must have the same length"

        responses = []
        for image_path, question in zip(image_paths, questions):
            response = self.query(
                image_path=image_path,
                question=question,
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens,
            )
            responses.append(response)

        return responses


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Test InternVL wrapper")
    parser.add_argument("--model_path", type=str, required=True, help="Path to InternVL model")
    parser.add_argument("--image", type=str, required=True, help="Path to test image")
    parser.add_argument("--question", type=str, default="What is in this image?")
    args = parser.parse_args()

    vlm = InternVLWrapper(model_path=args.model_path)

    response = vlm.query(
        image_path=args.image,
        question=args.question,
        system_prompt="Answer the question briefly using a single word or phrase.",
    )

    print(f"Question: {args.question}")
    print(f"Response: {response}")

"""
vLLM VLM Wrapper for VQA Inference.

This module provides an interface to locally hosted vLLM servers
for vision-language models like Qwen2.5-VL.
"""

import base64
import requests
from typing import Optional
from pathlib import Path
from PIL import Image
import io


class VLLMWrapper:
    """
    Wrapper for vLLM server to perform VQA inference with vision models.

    Usage:
        vlm = VLLMWrapper(base_url="http://localhost:8001")
        answer = vlm.query(
            image_path="image.jpg",
            question="What is in this image?",
            system_prompt="Answer briefly."
        )
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        max_tokens: int = 256,
    ):
        """
        Initialize the vLLM wrapper.

        Args:
            base_url: URL of the vLLM server (e.g., http://localhost:8001)
            model: Model name (used in API calls)
            max_tokens: Maximum tokens to generate
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.api_url = f"{self.base_url}/v1/chat/completions"

        print(f"Initialized vLLM wrapper at {self.base_url} with model: {model}")

    def _encode_image(self, image_source) -> str:
        """Encode an image to base64."""
        if isinstance(image_source, str):
            with open(image_source, "rb") as f:
                image_bytes = f.read()
        elif isinstance(image_source, Image.Image):
            buffer = io.BytesIO()
            # Convert to RGB if necessary
            if image_source.mode != "RGB":
                image_source = image_source.convert("RGB")
            image_source.save(buffer, format="JPEG", quality=85)
            image_bytes = buffer.getvalue()
        elif isinstance(image_source, bytes):
            image_bytes = image_source
        else:
            raise ValueError(f"Unsupported image source type: {type(image_source)}")

        return base64.b64encode(image_bytes).decode("utf-8")

    def _get_mime_type(self, image_source) -> str:
        """Determine MIME type of image."""
        if isinstance(image_source, str):
            ext = Path(image_source).suffix.lower()
            mime_map = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".webp": "image/webp",
            }
            return mime_map.get(ext, "image/jpeg")
        return "image/jpeg"

    def query(
        self,
        image_path,
        question: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        max_new_tokens: Optional[int] = None,  # Alias for compatibility
    ) -> str:
        """
        Query the VLM with an image and question.

        Args:
            image_path: Path to image file, PIL Image, or bytes
            question: Question to ask about the image
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate

        Returns:
            Model's text response
        """
        image_base64 = self._encode_image(image_path)
        mime_type = self._get_mime_type(image_path)

        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{image_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": question
                }
            ]
        })

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or max_new_tokens or self.max_tokens,
            "temperature": 0.1,  # Small temperature for slight variation
        }

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=120,
            )
            response.raise_for_status()

            result = response.json()
            content = result["choices"][0]["message"]["content"]
            return content.strip() if content else ""

        except requests.exceptions.RequestException as e:
            print(f"vLLM API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text[:500]}")
            return ""

    def batch_query(
        self,
        image_paths: list,
        questions: list[str],
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> list[str]:
        """Query the VLM with multiple image-question pairs."""
        assert len(image_paths) == len(questions)

        responses = []
        for image_path, question in zip(image_paths, questions):
            response = self.query(
                image_path=image_path,
                question=question,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
            )
            responses.append(response)

        return responses


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test vLLM wrapper")
    parser.add_argument("--base_url", type=str, default="http://localhost:8001")
    parser.add_argument("--image", type=str, required=True, help="Path to test image")
    parser.add_argument("--question", type=str, default="What is in this image?")
    args = parser.parse_args()

    vlm = VLLMWrapper(base_url=args.base_url)

    response = vlm.query(
        image_path=args.image,
        question=args.question,
        system_prompt="Answer the question briefly using a single word or phrase.",
    )

    print(f"Question: {args.question}")
    print(f"Response: {response}")

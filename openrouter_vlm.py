"""
OpenRouter VLM Wrapper for VQA Inference.

This module provides an interface to vision-language models via OpenRouter API,
supporting models like Qwen-2.5-VL for VQA tasks.
"""

import os
import base64
import requests
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
import io


class OpenRouterVLM:
    """
    Wrapper for OpenRouter API to perform VQA inference with vision models.

    Usage:
        vlm = OpenRouterVLM(model="qwen/qwen-2.5-vl-7b-instruct:free")
        answer = vlm.query(
            image_path="image.jpg",
            question="What is in this image?",
            system_prompt="Answer briefly."
        )
    """

    def __init__(
        self,
        model: str = "qwen/qwen-2.5-vl-7b-instruct:free",
        api_key: Optional[str] = None,
        max_tokens: int = 512,
    ):
        """
        Initialize the OpenRouter VLM wrapper.

        Args:
            model: OpenRouter model identifier
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            max_tokens: Maximum tokens to generate
        """
        # Load .env file from the same directory
        env_path = Path(__file__).parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)

        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. Set OPENROUTER_API_KEY env var or pass api_key."
            )

        self.model = model
        self.max_tokens = max_tokens
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

        print(f"Initialized OpenRouter VLM with model: {model}")

    def _encode_image(self, image_source) -> str:
        """
        Encode an image to base64.

        Args:
            image_source: Path to image file, PIL Image, or bytes

        Returns:
            Base64 encoded image string
        """
        if isinstance(image_source, str):
            # It's a file path
            with open(image_source, "rb") as f:
                image_bytes = f.read()
        elif isinstance(image_source, Image.Image):
            # It's a PIL Image
            buffer = io.BytesIO()
            image_source.save(buffer, format="PNG")
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
            return mime_map.get(ext, "image/png")
        return "image/png"

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
            system_prompt: Optional system prompt to prepend
            max_tokens: Maximum tokens to generate

        Returns:
            Model's text response
        """
        # Encode image
        image_base64 = self._encode_image(image_path)
        mime_type = self._get_mime_type(image_path)

        # Build messages
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        # User message with image and question
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

        # Make API request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or max_new_tokens or self.max_tokens,
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=60,
                )
                response.raise_for_status()

                result = response.json()
                if "choices" not in result:
                    print(f"Unexpected API response: {result}")
                    return ""

                message = result["choices"][0]["message"]
                content = message.get("content", "")

                # Some models (like Nemotron) put the answer in reasoning field
                if not content and "reasoning" in message:
                    # Extract the last sentence or key answer from reasoning
                    reasoning = message["reasoning"]
                    content = reasoning  # Use reasoning as fallback

                return content.strip() if content else ""

            except requests.exceptions.RequestException as e:
                is_retriable = False
                if hasattr(e, 'response') and e.response is not None:
                    status = e.response.status_code
                    is_retriable = status in [429, 500, 502, 503, 504]

                if is_retriable and attempt < max_retries - 1:
                    import time
                    wait_time = 2 ** attempt
                    print(f"API error (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                print(f"API request failed: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    print(f"Response: {e.response.text[:500]}")
                return ""

        return ""

    def batch_query(
        self,
        image_paths: list,
        questions: list[str],
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> list[str]:
        """
        Query the VLM with multiple image-question pairs.

        Args:
            image_paths: List of paths to image files or PIL Images
            questions: List of questions (same length as image_paths)
            system_prompt: Optional system prompt for all queries
            max_tokens: Maximum tokens to generate

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
                max_tokens=max_tokens,
            )
            responses.append(response)

        return responses


if __name__ == "__main__":
    # Test the wrapper
    import argparse

    parser = argparse.ArgumentParser(description="Test OpenRouter VLM wrapper")
    parser.add_argument("--image", type=str, required=True, help="Path to test image")
    parser.add_argument("--question", type=str, default="What is in this image?")
    parser.add_argument("--model", type=str, default="qwen/qwen-2.5-vl-7b-instruct:free")
    args = parser.parse_args()

    vlm = OpenRouterVLM(model=args.model)

    response = vlm.query(
        image_path=args.image,
        question=args.question,
        system_prompt="Answer the question briefly using a single word or phrase.",
    )

    print(f"Question: {args.question}")
    print(f"Response: {response}")

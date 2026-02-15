"""
Agentic VQA System with Tool Calling.

This module implements a multi-turn VQA agent that can use tools like
cropping bounding boxes to focus on specific regions of documents.
"""

import base64
import io
import json
import requests
from typing import Optional, Any
from dataclasses import dataclass
from PIL import Image


# Tool definitions for the VLM
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "crop_region",
            "description": "Crop a rectangular region from the document image to focus on a specific area. Use this when you need to look more closely at a table, chart, text block, or any specific region. Coordinates are relative (0-1) where (0,0) is top-left and (1,1) is bottom-right.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x1": {
                        "type": "number",
                        "description": "Left edge of crop region (0-1)"
                    },
                    "y1": {
                        "type": "number",
                        "description": "Top edge of crop region (0-1)"
                    },
                    "x2": {
                        "type": "number",
                        "description": "Right edge of crop region (0-1)"
                    },
                    "y2": {
                        "type": "number",
                        "description": "Bottom edge of crop region (0-1)"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Brief explanation of why you're cropping this region"
                    }
                },
                "required": ["x1", "y1", "x2", "y2"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "zoom_quadrant",
            "description": "Zoom into one of 4 quadrants of the image for a closer look. Use when the document has multiple sections and you need to focus on one area.",
            "parameters": {
                "type": "object",
                "properties": {
                    "quadrant": {
                        "type": "string",
                        "enum": ["top_left", "top_right", "bottom_left", "bottom_right"],
                        "description": "Which quadrant to zoom into"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Brief explanation of why you're zooming here"
                    }
                },
                "required": ["quadrant"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "reset_view",
            "description": "Reset to the full original image view. Use this after cropping if you need to see the whole document again.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Provide your final answer to the question. Use this when you have enough information to answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Your final answer to the question (be concise, use single word or short phrase when possible)"
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "How confident are you in this answer"
                    }
                },
                "required": ["answer"]
            }
        }
    }
]


@dataclass
class AgentState:
    """Tracks the state of the VQA agent during execution."""
    original_image: Image.Image
    current_image: Image.Image
    question: str
    messages: list
    tool_calls_made: int = 0
    max_tool_calls: int = 5
    crop_history: list = None

    def __post_init__(self):
        if self.crop_history is None:
            self.crop_history = []


class VQAAgent:
    """
    Agentic VQA system that uses tools to analyze documents.

    The agent can:
    - Crop specific regions of the document
    - Zoom into quadrants
    - Reset to full view
    - Provide final answers

    This enables multi-turn reasoning where the model can focus on
    specific parts of complex documents before answering.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        max_tool_calls: int = 5,
        max_tokens: int = 512,
        image_max_size: int = 768,
    ):
        """
        Initialize the VQA agent.

        Args:
            base_url: vLLM server URL
            model: Model name
            max_tool_calls: Maximum number of tool calls per question
            max_tokens: Max tokens for each response
            image_max_size: Max dimension for images sent to model
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tool_calls = max_tool_calls
        self.max_tokens = max_tokens
        self.image_max_size = image_max_size
        self.api_url = f"{self.base_url}/v1/chat/completions"

    def _load_image(self, image_source) -> Image.Image:
        """Load and preprocess image."""
        if isinstance(image_source, str):
            image = Image.open(image_source)
        elif isinstance(image_source, Image.Image):
            image = image_source
        else:
            raise ValueError(f"Unsupported image type: {type(image_source)}")

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def _resize_image(self, image: Image.Image) -> Image.Image:
        """Resize image for model input."""
        width, height = image.size
        if width <= self.image_max_size and height <= self.image_max_size:
            return image

        if width > height:
            new_width = self.image_max_size
            new_height = int(height * self.image_max_size / width)
        else:
            new_height = self.image_max_size
            new_width = int(width * self.image_max_size / height)

        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def _encode_image(self, image: Image.Image) -> str:
        """Encode image to base64."""
        image = self._resize_image(image)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _execute_tool(self, tool_name: str, args: dict, state: AgentState) -> tuple[str, Image.Image]:
        """
        Execute a tool and return the result.

        Returns:
            Tuple of (result_message, updated_image)
        """
        if tool_name == "crop_region":
            return self._crop_region(args, state)
        elif tool_name == "zoom_quadrant":
            return self._zoom_quadrant(args, state)
        elif tool_name == "reset_view":
            return self._reset_view(state)
        elif tool_name == "final_answer":
            return self._final_answer(args, state)
        else:
            return f"Unknown tool: {tool_name}", state.current_image

    def _crop_region(self, args: dict, state: AgentState) -> tuple[str, Image.Image]:
        """Crop a rectangular region from the image."""
        x1 = max(0, min(1, args.get("x1", 0)))
        y1 = max(0, min(1, args.get("y1", 0)))
        x2 = max(0, min(1, args.get("x2", 1)))
        y2 = max(0, min(1, args.get("y2", 1)))

        # Ensure valid bounds
        if x2 <= x1 or y2 <= y1:
            return "Invalid crop coordinates. x2 must be > x1 and y2 must be > y1.", state.current_image

        width, height = state.original_image.size
        crop_box = (
            int(x1 * width),
            int(y1 * height),
            int(x2 * width),
            int(y2 * height)
        )

        cropped = state.original_image.crop(crop_box)
        state.crop_history.append({
            "type": "crop",
            "coords": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "reason": args.get("reason", "")
        })

        reason = args.get("reason", "focusing on this region")
        return f"Cropped region ({x1:.2f}, {y1:.2f}) to ({x2:.2f}, {y2:.2f}): {reason}. Here is the cropped view:", cropped

    def _zoom_quadrant(self, args: dict, state: AgentState) -> tuple[str, Image.Image]:
        """Zoom into a quadrant of the image."""
        quadrant = args.get("quadrant", "top_left")

        quadrant_coords = {
            "top_left": (0, 0, 0.5, 0.5),
            "top_right": (0.5, 0, 1, 0.5),
            "bottom_left": (0, 0.5, 0.5, 1),
            "bottom_right": (0.5, 0.5, 1, 1)
        }

        if quadrant not in quadrant_coords:
            return f"Invalid quadrant: {quadrant}. Use: top_left, top_right, bottom_left, bottom_right", state.current_image

        x1, y1, x2, y2 = quadrant_coords[quadrant]
        width, height = state.original_image.size
        crop_box = (
            int(x1 * width),
            int(y1 * height),
            int(x2 * width),
            int(y2 * height)
        )

        zoomed = state.original_image.crop(crop_box)
        state.crop_history.append({
            "type": "zoom_quadrant",
            "quadrant": quadrant,
            "reason": args.get("reason", "")
        })

        reason = args.get("reason", "examining this area")
        return f"Zoomed into {quadrant} quadrant: {reason}. Here is the zoomed view:", zoomed

    def _reset_view(self, state: AgentState) -> tuple[str, Image.Image]:
        """Reset to the full original image."""
        state.crop_history.append({"type": "reset"})
        return "Reset to full document view. Here is the complete image:", state.original_image

    def _final_answer(self, args: dict, state: AgentState) -> tuple[str, Image.Image]:
        """Return the final answer (special handling in agent loop)."""
        answer = args.get("answer", "")
        confidence = args.get("confidence", "medium")
        return f"FINAL_ANSWER:{answer}|CONFIDENCE:{confidence}", state.current_image

    def _call_model(
        self,
        messages: list,
        current_image: Image.Image,
        use_tools: bool = True,
    ) -> dict:
        """Call the VLM with messages and current image."""
        # Build the last user message with current image
        image_base64 = self._encode_image(current_image)

        # Check if last message is user message that needs image
        formatted_messages = []
        for msg in messages:
            if msg["role"] == "user" and "ADD_CURRENT_IMAGE" in str(msg.get("content", "")):
                # This message needs the current image
                text_content = msg["content"].replace("ADD_CURRENT_IMAGE", "").strip()
                formatted_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                        },
                        {"type": "text", "text": text_content}
                    ]
                })
            else:
                formatted_messages.append(msg)

        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "max_tokens": self.max_tokens,
            "temperature": 0.1,
        }

        if use_tools:
            payload["tools"] = TOOLS
            payload["tool_choice"] = "auto"

        response = requests.post(self.api_url, json=payload, timeout=300)
        response.raise_for_status()
        return response.json()

    def query(
        self,
        image_path,
        question: str,
        system_prompt: Optional[str] = None,
        verbose: bool = False,
        max_new_tokens: int = None,  # For compatibility
        max_tokens: int = None,  # For compatibility
    ) -> str:
        """
        Query the VQA agent with an image and question.

        The agent will use tools as needed to analyze the document
        before providing a final answer.

        Args:
            image_path: Path to image or PIL Image
            question: Question about the image
            system_prompt: Optional system prompt to guide behavior
            verbose: Print agent reasoning steps

        Returns:
            Final answer string
        """
        # Load image
        original_image = self._load_image(image_path)

        # Initialize state
        state = AgentState(
            original_image=original_image,
            current_image=original_image,
            question=question,
            messages=[],
            max_tool_calls=self.max_tool_calls
        )

        # Build initial system prompt
        default_system = """You are a document analysis assistant. You can use tools to examine specific regions of documents.

When answering questions:
1. First look at the full document to understand its structure
2. If needed, use crop_region or zoom_quadrant to focus on relevant areas (tables, specific text, etc.)
3. When you have enough information, use final_answer to provide your response

Be precise and concise in your answers. For questions asking for specific values, dates, or names, give just that value."""

        if system_prompt:
            full_system = f"{system_prompt}\n\n{default_system}"
        else:
            full_system = default_system

        state.messages.append({"role": "system", "content": full_system})

        # Initial user message with image
        state.messages.append({
            "role": "user",
            "content": f"ADD_CURRENT_IMAGE\n\nQuestion: {question}\n\nAnalyze the document and answer the question. Use tools if needed to focus on specific regions."
        })

        # Agent loop
        final_answer = None

        while state.tool_calls_made < state.max_tool_calls:
            try:
                result = self._call_model(state.messages, state.current_image, use_tools=True)

                choice = result["choices"][0]
                message = choice["message"]

                # Check for tool calls
                tool_calls = message.get("tool_calls", [])

                if not tool_calls:
                    # No tool calls - model gave direct response
                    content = message.get("content", "")
                    if verbose:
                        print(f"  [Agent] Direct response: {content[:100]}...")

                    # Check if it's trying to give an answer without using final_answer tool
                    if content.strip():
                        final_answer = content.strip()
                    break

                # Process tool calls
                state.messages.append(message)  # Add assistant message with tool calls

                for tool_call in tool_calls:
                    tool_name = tool_call["function"]["name"]
                    try:
                        tool_args = json.loads(tool_call["function"]["arguments"])
                    except json.JSONDecodeError:
                        tool_args = {}

                    if verbose:
                        print(f"  [Agent] Tool call: {tool_name}({tool_args})")

                    # Execute tool
                    result_msg, new_image = self._execute_tool(tool_name, tool_args, state)
                    state.current_image = new_image
                    state.tool_calls_made += 1

                    # Check for final answer
                    if result_msg.startswith("FINAL_ANSWER:"):
                        parts = result_msg.split("|")
                        final_answer = parts[0].replace("FINAL_ANSWER:", "")
                        if verbose:
                            confidence = parts[1].replace("CONFIDENCE:", "") if len(parts) > 1 else "unknown"
                            print(f"  [Agent] Final answer: {final_answer} (confidence: {confidence})")
                        break

                    # Add tool result to messages
                    # For tool results with new images, we need to include the image
                    if "cropped" in result_msg.lower() or "zoomed" in result_msg.lower() or "reset" in result_msg.lower():
                        state.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": f"ADD_CURRENT_IMAGE\n\n{result_msg}"
                        })
                    else:
                        state.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": result_msg
                        })

                if final_answer is not None:
                    break

            except Exception as e:
                if verbose:
                    print(f"  [Agent] Error: {e}")
                break

        # If no final answer from tools, try to extract from last response
        if final_answer is None:
            if verbose:
                print("  [Agent] No final answer from tools, using direct response")
            # Make one more call without tools to get direct answer
            try:
                state.messages.append({
                    "role": "user",
                    "content": "Please provide your final answer to the question in a single word or short phrase."
                })
                result = self._call_model(state.messages, state.current_image, use_tools=False)
                final_answer = result["choices"][0]["message"].get("content", "").strip()
            except:
                final_answer = ""

        return final_answer

    def batch_query(
        self,
        image_paths: list,
        questions: list[str],
        system_prompt: Optional[str] = None,
        verbose: bool = False,
        max_tokens: Optional[int] = None,
    ) -> list[str]:
        """Query multiple image-question pairs."""
        responses = []
        for image_path, question in zip(image_paths, questions):
            response = self.query(
                image_path=image_path,
                question=question,
                system_prompt=system_prompt,
                verbose=verbose,
            )
            responses.append(response)
        return responses


# Simplified wrapper that matches VLLMWrapper interface
class VQAAgentWrapper:
    """
    Wrapper that provides the same interface as VLLMWrapper but uses the agent.

    This allows drop-in replacement in the GEPA optimization pipeline.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        max_tool_calls: int = 3,
        max_tokens: int = 256,
        verbose: bool = False,
    ):
        self.base_url = base_url
        self.model = model
        self.agent = VQAAgent(
            base_url=base_url,
            model=model,
            max_tool_calls=max_tool_calls,
            max_tokens=max_tokens,
        )
        self.verbose = verbose
        print(f"Initialized VQA Agent at {base_url} with model: {model}")
        print(f"  Tools enabled: crop_region, zoom_quadrant, reset_view, final_answer")
        print(f"  Max tool calls per question: {max_tool_calls}")

    def query(
        self,
        image_path,
        question: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """Query with agent (compatible with VLLMWrapper interface)."""
        return self.agent.query(
            image_path=image_path,
            question=question,
            system_prompt=system_prompt,
            verbose=self.verbose,
        )

    def batch_query(
        self,
        image_paths: list,
        questions: list[str],
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> list[str]:
        """Batch query with agent."""
        return self.agent.batch_query(
            image_paths=image_paths,
            questions=questions,
            system_prompt=system_prompt,
            verbose=self.verbose,
        )


if __name__ == "__main__":
    # Test the agent
    print("Testing VQA Agent...")

    agent = VQAAgent(
        base_url="http://localhost:8000",
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        max_tool_calls=3,
    )

    # Create a test image
    from PIL import Image, ImageDraw, ImageFont

    # Create a simple document-like image for testing
    img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img)

    # Add some text
    draw.text((50, 50), "INVOICE", fill='black')
    draw.text((50, 100), "Date: 2024-01-15", fill='black')
    draw.text((50, 150), "Amount: $1,234.56", fill='black')
    draw.text((50, 200), "Customer: John Doe", fill='black')

    # Add a simple table
    draw.rectangle([50, 250, 400, 350], outline='black')
    draw.line([50, 280, 400, 280], fill='black')
    draw.line([200, 250, 200, 350], fill='black')
    draw.text((60, 255), "Item", fill='black')
    draw.text((210, 255), "Price", fill='black')
    draw.text((60, 290), "Widget A", fill='black')
    draw.text((210, 290), "$500.00", fill='black')
    draw.text((60, 320), "Widget B", fill='black')
    draw.text((210, 320), "$734.56", fill='black')

    # Save test image
    test_img_path = "/tmp/test_invoice.png"
    img.save(test_img_path)

    print(f"\nTest image saved to: {test_img_path}")
    print("\nTesting agent with question: 'What is the total amount?'")

    try:
        answer = agent.query(
            image_path=test_img_path,
            question="What is the total amount on this invoice?",
            verbose=True,
        )
        print(f"\nFinal Answer: {answer}")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure vLLM server is running with tool calling enabled:")
        print("  python -m vllm.entrypoints.openai.api_server \\")
        print("    --model Qwen/Qwen2.5-VL-7B-Instruct \\")
        print("    --trust-remote-code --port 8000")

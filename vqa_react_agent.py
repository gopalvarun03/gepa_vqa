"""
ReAct-style VQA Agent with Tool Calling via Prompting.

This agent uses explicit prompting to get the model to use tools,
rather than relying on native function calling which may not work
reliably with all VLMs.
"""

import base64
import io
import json
import re
import requests
from typing import Optional, Any
from dataclasses import dataclass, field
from PIL import Image


@dataclass
class AgentState:
    """Tracks the state of the VQA agent during execution."""
    original_image: Image.Image
    current_image: Image.Image
    question: str
    history: list = field(default_factory=list)
    tool_calls_made: int = 0
    max_tool_calls: int = 5


class VQAReActAgent:
    """
    ReAct-style VQA Agent that uses prompting for tool calling.

    The agent follows a Thought -> Action -> Observation loop:
    1. Thought: Analyze what's needed
    2. Action: Either use a tool or give final answer
    3. Observation: See tool result
    4. Repeat until answer is found
    """

    SYSTEM_PROMPT = """You are a document analysis assistant that can examine images closely using tools.

Available tools:
1. CROP(x1, y1, x2, y2) - Crop a region. Coordinates are 0-1 relative (0,0 is top-left, 1,1 is bottom-right)
   Example: CROP(0.0, 0.5, 0.5, 1.0) crops the bottom-left quarter

2. ZOOM(quadrant) - Zoom into a quadrant: "top_left", "top_right", "bottom_left", "bottom_right"
   Example: ZOOM(bottom_right)

3. RESET() - Go back to full image view

4. ANSWER(your_answer) - Give your final answer
   Example: ANSWER($1,234.56)

Response format - you MUST follow this EXACTLY:

THOUGHT: [Your reasoning about what you see and what to do next]
ACTION: [One of: CROP(x1,y1,x2,y2) | ZOOM(quadrant) | RESET() | ANSWER(answer)]

Rules:
- Give ONLY ONE action per response
- For ANSWER, be concise - single word or short phrase when possible
- Use tools to zoom into tables, small text, or specific regions
- After 2-3 tool uses, you should have enough info to ANSWER"""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        max_tool_calls: int = 5,
        max_tokens: int = 512,
        image_max_size: int = 768,
        verbose: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tool_calls = max_tool_calls
        self.max_tokens = max_tokens
        self.image_max_size = image_max_size
        self.verbose = verbose
        self.api_url = f"{self.base_url}/v1/chat/completions"

    def _load_image(self, image_source) -> Image.Image:
        """Load image from various sources."""
        if isinstance(image_source, str):
            image = Image.open(image_source)
        elif isinstance(image_source, Image.Image):
            image = image_source.copy()
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

    def _parse_action(self, response: str) -> tuple[str, Any]:
        """Parse the action from model response."""
        # Look for ACTION: line
        action_match = re.search(r'ACTION:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if not action_match:
            # Try to find any tool-like pattern
            for pattern in [r'ANSWER\((.+?)\)', r'CROP\((.+?)\)', r'ZOOM\((.+?)\)', r'RESET\(\)']:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    action_str = match.group(0)
                    break
            else:
                return "NONE", None
        else:
            action_str = action_match.group(1).strip()

        # Parse specific actions
        # ANSWER
        answer_match = re.search(r'ANSWER\((.+?)\)', action_str, re.IGNORECASE)
        if answer_match:
            return "ANSWER", answer_match.group(1).strip().strip('"\'')

        # CROP
        crop_match = re.search(r'CROP\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\)', action_str, re.IGNORECASE)
        if crop_match:
            coords = [float(x) for x in crop_match.groups()]
            return "CROP", coords

        # ZOOM
        zoom_match = re.search(r'ZOOM\(\s*["\']?(\w+)["\']?\s*\)', action_str, re.IGNORECASE)
        if zoom_match:
            return "ZOOM", zoom_match.group(1).lower()

        # RESET
        if 'RESET' in action_str.upper():
            return "RESET", None

        return "NONE", None

    def _execute_action(self, action: str, args: Any, state: AgentState) -> tuple[str, Image.Image]:
        """Execute an action and return observation."""
        if action == "CROP":
            x1, y1, x2, y2 = args
            x1, y1, x2, y2 = max(0, min(1, x1)), max(0, min(1, y1)), max(0, min(1, x2)), max(0, min(1, y2))

            if x2 <= x1 or y2 <= y1:
                return "Invalid crop coordinates. Make sure x2 > x1 and y2 > y1.", state.current_image

            w, h = state.original_image.size
            crop_box = (int(x1*w), int(y1*h), int(x2*w), int(y2*h))
            cropped = state.original_image.crop(crop_box)

            return f"Cropped to region ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}). Here is the cropped view:", cropped

        elif action == "ZOOM":
            quadrant = args
            coords = {
                "top_left": (0, 0, 0.5, 0.5),
                "top_right": (0.5, 0, 1, 0.5),
                "bottom_left": (0, 0.5, 0.5, 1),
                "bottom_right": (0.5, 0.5, 1, 1),
            }
            if quadrant not in coords:
                return f"Invalid quadrant '{quadrant}'. Use: top_left, top_right, bottom_left, bottom_right", state.current_image

            x1, y1, x2, y2 = coords[quadrant]
            w, h = state.original_image.size
            crop_box = (int(x1*w), int(y1*h), int(x2*w), int(y2*h))
            zoomed = state.original_image.crop(crop_box)

            return f"Zoomed into {quadrant} quadrant. Here is the zoomed view:", zoomed

        elif action == "RESET":
            return "Reset to full document view:", state.original_image

        return "Unknown action", state.current_image

    def _call_model(self, state: AgentState) -> str:
        """Call the model with current state."""
        # Build message with current image
        image_b64 = self._encode_image(state.current_image)

        # Build conversation history
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]

        # First turn: image + question
        if not state.history:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    {"type": "text", "text": f"Question: {state.question}\n\nAnalyze the document and answer the question."}
                ]
            })
        else:
            # Add history
            for entry in state.history:
                if entry["role"] == "assistant":
                    messages.append({"role": "assistant", "content": entry["content"]})
                elif entry["role"] == "observation":
                    # Observation with new image
                    obs_image_b64 = self._encode_image(entry["image"])
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{obs_image_b64}"}},
                            {"type": "text", "text": f"OBSERVATION: {entry['content']}"}
                        ]
                    })

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": 0.1,
        }

        response = requests.post(self.api_url, json=payload, timeout=300)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def query(
        self,
        image_path,
        question: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = None,
        max_tokens: int = None,
    ) -> str:
        """
        Query the agent with an image and question.

        Args:
            image_path: Path to image or PIL Image
            question: Question about the image
            system_prompt: Optional additional instructions (prepended to question)

        Returns:
            Final answer string
        """
        original_image = self._load_image(image_path)

        # Prepend system prompt to question if provided
        full_question = question
        if system_prompt:
            full_question = f"{system_prompt}\n\n{question}"

        state = AgentState(
            original_image=original_image,
            current_image=original_image,
            question=full_question,
            max_tool_calls=self.max_tool_calls,
        )

        if self.verbose:
            print(f"\n  [Agent] Question: {question}")

        # Agent loop
        while state.tool_calls_made < state.max_tool_calls:
            try:
                # Get model response
                response = self._call_model(state)

                if self.verbose:
                    # Print thought
                    thought_match = re.search(r'THOUGHT:\s*(.+?)(?=ACTION:|$)', response, re.IGNORECASE | re.DOTALL)
                    if thought_match:
                        print(f"  [Thought] {thought_match.group(1).strip()[:100]}...")

                # Parse action
                action, args = self._parse_action(response)

                if self.verbose:
                    print(f"  [Action] {action}({args})")

                # Check for final answer
                if action == "ANSWER":
                    if self.verbose:
                        print(f"  [Answer] {args}")
                    return args

                # Handle no action
                if action == "NONE":
                    # Model didn't follow format - try to extract answer from response
                    state.history.append({"role": "assistant", "content": response})
                    state.history.append({
                        "role": "observation",
                        "content": "Please follow the format. Use ANSWER(your_answer) to provide your final answer.",
                        "image": state.current_image,
                    })
                    state.tool_calls_made += 1
                    continue

                # Execute tool
                observation, new_image = self._execute_action(action, args, state)
                state.current_image = new_image
                state.tool_calls_made += 1

                if self.verbose:
                    print(f"  [Observation] {observation[:50]}...")

                # Add to history
                state.history.append({"role": "assistant", "content": response})
                state.history.append({
                    "role": "observation",
                    "content": observation,
                    "image": new_image,
                })

            except Exception as e:
                if self.verbose:
                    print(f"  [Error] {e}")
                break

        # If we ran out of tool calls, ask for final answer
        if self.verbose:
            print("  [Agent] Max tool calls reached, requesting final answer...")

        state.history.append({
            "role": "observation",
            "content": "You've used all available tool calls. Please provide your final ANSWER now.",
            "image": state.current_image,
        })

        try:
            response = self._call_model(state)
            action, args = self._parse_action(response)
            if action == "ANSWER":
                return args
            # Try to extract any answer-like content
            return response.split('\n')[0].strip()[:100]
        except:
            return ""

    def batch_query(
        self,
        image_paths: list,
        questions: list[str],
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> list[str]:
        """Query multiple image-question pairs."""
        responses = []
        for image_path, question in zip(image_paths, questions):
            response = self.query(image_path, question, system_prompt)
            responses.append(response)
        return responses


class VQAReActAgentWrapper:
    """
    Wrapper that provides VLLMWrapper-compatible interface.
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
        self.agent = VQAReActAgent(
            base_url=base_url,
            model=model,
            max_tool_calls=max_tool_calls,
            max_tokens=max_tokens,
            verbose=verbose,
        )
        self.verbose = verbose
        print(f"Initialized ReAct VQA Agent at {base_url}")
        print(f"  Model: {model}")
        print(f"  Tools: CROP, ZOOM, RESET, ANSWER")
        print(f"  Max tool calls: {max_tool_calls}")

    def query(
        self,
        image_path,
        question: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        return self.agent.query(image_path, question, system_prompt)

    def batch_query(
        self,
        image_paths: list,
        questions: list[str],
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> list[str]:
        return self.agent.batch_query(image_paths, questions, system_prompt)


if __name__ == "__main__":
    from vqa_dataset import load_docvqa

    print("=" * 60)
    print("ReAct VQA Agent Test")
    print("=" * 60)

    # Load samples
    print("\nLoading DocVQA samples...")
    data = load_docvqa(split="validation", max_samples=2)

    # Initialize agent
    agent = VQAReActAgent(
        base_url="http://localhost:8000",
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        max_tool_calls=3,
        verbose=True,
    )

    # Test
    for i, sample in enumerate(data):
        print(f"\n{'='*60}")
        print(f"Sample {i+1}")
        print(f"Expected answer: {sample['answer']}")

        answer = agent.query(
            image_path=sample['image_path'],
            question=sample['question'],
        )

        print(f"\nFinal: {answer}")
        print(f"Match: {answer.lower().strip() == sample['answer'].lower().strip()}")

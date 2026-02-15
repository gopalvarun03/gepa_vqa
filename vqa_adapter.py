"""
GEPA Adapter for VQA Optimization with Multimodal Reflection.

This module implements the GEPAAdapter protocol for Visual Question Answering,
allowing GEPA to optimize VQA system prompts using images in reflection.
"""

import sys
import base64
import io
from pathlib import Path
from typing import Any, TypedDict, Optional
from collections.abc import Mapping, Sequence

from PIL import Image

# Add GEPA to path
GEPA_PATH = Path(__file__).parent.parent / "gepa" / "src"
sys.path.insert(0, str(GEPA_PATH))

from gepa.core.adapter import EvaluationBatch, GEPAAdapter

from vqa_dataset import VQADataInst, compute_vqa_accuracy
from vlm_wrapper import InternVLWrapper


class VQATrajectory(TypedDict):
    """
    Trajectory data captured during VQA evaluation.

    Contains all information needed for reflection, including the image.
    """
    data: VQADataInst           # Original input data
    system_prompt: str          # System prompt used
    vlm_response: str           # Raw VLM response
    score: float                # VQA accuracy score
    image: Any                  # PIL Image or path for reflection


class VQARolloutOutput(TypedDict):
    """
    Output from a single VQA evaluation.
    """
    predicted_answer: str       # Model's predicted answer
    question_id: str            # Question identifier


class VQAReflectiveRecord(TypedDict):
    """
    Record format for the reflective dataset.

    This is what gets passed to the reflection LLM.
    """
    Inputs: str                 # Question and image context
    Generated_Outputs: str      # VLM response (using underscore to match GEPA convention)
    Feedback: str               # Feedback on correctness
    image: Any                  # PIL Image for multimodal reflection


class VQAAdapter(GEPAAdapter[VQADataInst, VQATrajectory, VQARolloutOutput]):
    """
    GEPA Adapter for VQA prompt optimization with multimodal reflection.

    This adapter:
    1. Takes a candidate system prompt
    2. Runs the VLM on VQA examples with that prompt
    3. Computes VQA accuracy scores
    4. Generates reflective datasets WITH IMAGES for prompt improvement
    5. Uses a VLM for reflection to understand visual patterns

    Usage:
        vlm = VLLMWrapper(base_url="http://localhost:8000")
        adapter = VQAAdapter(vlm=vlm, reflection_vlm=vlm)

        result = gepa.optimize(
            seed_candidate={"system_prompt": "Answer briefly."},
            trainset=train_data,
            valset=val_data,
            adapter=adapter,
            reflection_lm="hosted_vllm/model",  # Still needed but custom propose_new_texts will be used
            max_metric_calls=100,
        )
    """

    # Tool descriptions for agentic VLM
    TOOL_DESCRIPTIONS = """
The VLM being optimized has access to these tools:
1. CROP(x1, y1, x2, y2) - Crop a rectangular region from the document (coordinates 0-1, where 0,0 is top-left)
2. ZOOM(quadrant) - Zoom into a quadrant: "top_left", "top_right", "bottom_left", "bottom_right"
3. RESET() - Reset to full document view
4. ANSWER(response) - Provide the final answer

The system prompt should guide the VLM on WHEN and HOW to use these tools effectively.
For example: "For questions about specific values in tables, use CROP to focus on the relevant table section before answering."
"""

    def __init__(
        self,
        vlm,
        component_name: str = "system_prompt",
        max_new_tokens: int = 50,
        reflection_vlm=None,
        reflection_max_tokens: int = 1024,
        is_agentic: bool = False,
    ):
        """
        Initialize the VQA adapter.

        Args:
            vlm: VLM wrapper instance for evaluation
            component_name: Name of the text component being optimized
            max_new_tokens: Maximum tokens for VLM generation
            reflection_vlm: VLM wrapper for reflection (if None, uses same as vlm)
            reflection_max_tokens: Max tokens for reflection response
            is_agentic: Whether the VLM has tool calling capabilities
        """
        self.vlm = vlm
        self.component_name = component_name
        self.max_new_tokens = max_new_tokens
        self.reflection_vlm = reflection_vlm if reflection_vlm is not None else vlm
        self.reflection_max_tokens = reflection_max_tokens
        self.is_agentic = is_agentic

    def evaluate(
        self,
        batch: list[VQADataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[VQATrajectory, VQARolloutOutput]:
        """
        Evaluate a candidate prompt on a batch of VQA examples.

        Args:
            batch: List of VQA data instances
            candidate: Mapping from component name to text (e.g., {"system_prompt": "..."})
            capture_traces: Whether to capture execution traces for reflection

        Returns:
            EvaluationBatch containing outputs, scores, and optionally trajectories
        """
        # Extract system prompt from candidate
        system_prompt = candidate.get(self.component_name, "")

        outputs: list[VQARolloutOutput] = []
        scores: list[float] = []
        trajectories: list[VQATrajectory] | None = [] if capture_traces else None

        for data in batch:
            try:
                # Query the VLM
                vlm_response = self.vlm.query(
                    image_path=data["image_path"],
                    question=data["question"],
                    system_prompt=system_prompt,
                    max_new_tokens=self.max_new_tokens,
                )

                # Compute VQA accuracy
                score = compute_vqa_accuracy(vlm_response, data["answers"])

            except Exception as e:
                # Handle failures gracefully - return score 0.0
                vlm_response = f"[Error: {str(e)}]"
                score = 0.0

            # Create output
            output = VQARolloutOutput(
                predicted_answer=vlm_response,
                question_id=data["question_id"],
            )
            outputs.append(output)
            scores.append(score)

            # Capture trajectory if needed (including image for reflection)
            if trajectories is not None:
                trajectory = VQATrajectory(
                    data=data,
                    system_prompt=system_prompt,
                    vlm_response=vlm_response,
                    score=score,
                    image=data["image_path"],  # Store image/path for reflection
                )
                trajectories.append(trajectory)

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[VQATrajectory, VQARolloutOutput],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """
        Build a reflective dataset for prompt improvement WITH IMAGES.

        This dataset is fed to the reflection VLM to propose improved prompts.

        Args:
            candidate: Current candidate prompt
            eval_batch: Results from evaluate() with capture_traces=True
            components_to_update: List of component names to update

        Returns:
            Mapping from component name to list of reflective records (including images)
        """
        assert eval_batch.trajectories is not None, \
            "Trajectories are required to build a reflective dataset."

        result: dict[str, list[dict[str, Any]]] = {}

        for component in components_to_update:
            records: list[dict[str, Any]] = []

            for trajectory in eval_batch.trajectories:
                data = trajectory["data"]
                vlm_response = trajectory["vlm_response"]
                score = trajectory["score"]
                image = trajectory["image"]

                # Build input description
                inputs = f"Question: {data['question']}"

                # Build feedback based on score
                if score >= 1.0:
                    feedback = (
                        f"CORRECT. The model correctly answered '{vlm_response}'. "
                        f"Expected answer: '{data['answer']}'."
                    )
                elif score > 0.0:
                    feedback = (
                        f"PARTIALLY CORRECT (score: {score:.2f}). "
                        f"The model answered '{vlm_response}'. "
                        f"Expected answer: '{data['answer']}'."
                    )
                else:
                    feedback = (
                        f"INCORRECT. The model answered '{vlm_response}'. "
                        f"Expected answer: '{data['answer']}'. "
                        f"The response does not match any expected answer."
                    )

                    # Add hints for common error patterns
                    if len(vlm_response) > 100:
                        feedback += " Note: Response was too verbose - encourage shorter answers."
                    if "[Error" in vlm_response:
                        feedback += " Note: An error occurred during inference."

                record: dict[str, Any] = {
                    "Inputs": inputs,
                    "Generated Outputs": vlm_response,
                    "Feedback": feedback,
                    "image": image,  # Include image for multimodal reflection
                }
                records.append(record)

            result[component] = records

        if not any(result.values()):
            raise ValueError("No valid predictions found for any component.")

        return result

    def _encode_image_for_reflection(self, image_source) -> tuple[str, str]:
        """Encode image to base64 for reflection VLM."""
        if isinstance(image_source, str):
            # It's a file path
            with open(image_source, "rb") as f:
                image_bytes = f.read()
            ext = Path(image_source).suffix.lower()
            mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
            mime_type = mime_map.get(ext, "image/jpeg")
        elif isinstance(image_source, Image.Image):
            # It's a PIL Image
            buffer = io.BytesIO()
            if image_source.mode != "RGB":
                image_source = image_source.convert("RGB")
            # Resize for reflection (don't need full resolution)
            max_size = 512
            width, height = image_source.size
            if width > max_size or height > max_size:
                if width > height:
                    new_width = max_size
                    new_height = int(height * max_size / width)
                else:
                    new_height = max_size
                    new_width = int(width * max_size / height)
                image_source = image_source.resize((new_width, new_height), Image.Resampling.LANCZOS)
            image_source.save(buffer, format="JPEG", quality=85)
            image_bytes = buffer.getvalue()
            mime_type = "image/jpeg"
        else:
            raise ValueError(f"Unsupported image type: {type(image_source)}")

        return base64.b64encode(image_bytes).decode("utf-8"), mime_type

    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        """
        Propose new prompt texts using multimodal reflection with images.

        This method sends images along with the feedback to the reflection VLM,
        allowing it to understand visual patterns in successes/failures.

        Args:
            candidate: Current candidate prompt
            reflective_dataset: Dataset with images from make_reflective_dataset
            components_to_update: Components to update

        Returns:
            Dict mapping component names to new proposed texts
        """
        new_texts: dict[str, str] = {}

        for component in components_to_update:
            if component not in reflective_dataset or not reflective_dataset[component]:
                continue

            current_instruction = candidate[component]
            records = reflective_dataset[component]

            # Build multimodal prompt for reflection
            # We'll create a structured prompt with images inline
            reflection_prompt = self._build_multimodal_reflection_prompt(
                current_instruction, records
            )

            # Call reflection VLM with images
            new_instruction = self._call_reflection_vlm(reflection_prompt, records)

            if new_instruction:
                new_texts[component] = new_instruction

        return new_texts

    def _build_multimodal_reflection_prompt(
        self,
        current_instruction: str,
        records: Sequence[Mapping[str, Any]],
    ) -> str:
        """Build the text part of the multimodal reflection prompt."""
        # Add tool descriptions if agentic mode
        tool_section = ""
        if self.is_agentic:
            tool_section = f"""
IMPORTANT - AGENTIC VLM WITH TOOLS:
{self.TOOL_DESCRIPTIONS}

When optimizing the prompt, consider:
- When should the model use CROP vs ZOOM vs direct answering?
- What visual patterns indicate the need to zoom in?
- How to guide tool usage for different document types (tables, forms, charts)?

"""

        prompt = f"""You are helping optimize a system prompt for a Visual Question Answering (VQA) task.
{tool_section}
Current system prompt being used:
```
{current_instruction}
```

Below are examples showing the document images, questions asked, model responses, and feedback on correctness.
Look at each image carefully to understand WHY the model succeeded or failed.

"""
        for i, record in enumerate(records, 1):
            prompt += f"""
---
EXAMPLE {i}:
[Image {i} is shown above/below]
Question: {record['Inputs']}
Model's Answer: {record['Generated Outputs']}
{record['Feedback']}
"""

        if self.is_agentic:
            prompt += """
---

Based on analyzing the images and the model's performance:

1. Identify visual patterns where the model struggles (tables, small text, dense layouts)
2. Identify what types of questions need tool usage (specific values, dates in tables, small text)
3. Consider WHEN the model should use CROP/ZOOM vs answering directly

Write an improved system prompt that guides the model on:
- When to use CROP to focus on specific regions (e.g., "For table values, CROP the relevant row")
- When to use ZOOM for quadrant-level focus
- When to answer directly without tools
- How to identify the right region to examine

The prompt should be actionable and specific about tool usage patterns.

Provide your new prompt within ``` blocks:
```
[Your improved prompt here]
```"""
        else:
            prompt += """
---

Based on analyzing the images and the model's performance:

1. Identify visual patterns in documents where the model struggles (e.g., tables, handwriting, small text, multiple columns)
2. Identify what types of questions are harder (e.g., questions about specific values, dates, names)
3. Consider what instructions would help the model focus on the right parts of the document

Write an improved system prompt that will help the model answer questions more accurately.
The prompt should be concise but include specific guidance based on the visual patterns you observed.

Provide your new prompt within ``` blocks:
```
[Your improved prompt here]
```"""

        return prompt

    def _call_reflection_vlm(
        self,
        text_prompt: str,
        records: Sequence[Mapping[str, Any]],
    ) -> Optional[str]:
        """
        Call the reflection VLM with images and text prompt.

        This method handles sending multiple images to the VLM for reflection.
        """
        import requests

        # Check if reflection_vlm has the necessary interface
        if not hasattr(self.reflection_vlm, 'base_url'):
            # Fallback: use single image reflection or text-only
            print("  [Reflection] Warning: reflection_vlm doesn't support multimodal. Using first image only.")
            if records and 'image' in records[0]:
                return self._single_image_reflection(text_prompt, records[0]['image'])
            return None

        # Build multimodal message with all images
        content_parts = []

        # Add images first
        for i, record in enumerate(records):
            if 'image' not in record:
                continue
            try:
                base64_img, mime_type = self._encode_image_for_reflection(record['image'])
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_img}"
                    }
                })
            except Exception as e:
                print(f"  [Reflection] Warning: Failed to encode image {i}: {e}")

        # Add text prompt
        content_parts.append({
            "type": "text",
            "text": text_prompt
        })

        # Send to vLLM
        messages = [{"role": "user", "content": content_parts}]

        payload = {
            "model": self.reflection_vlm.model,
            "messages": messages,
            "max_tokens": self.reflection_max_tokens,
            "temperature": 0.7,  # Some creativity for prompt generation
        }

        try:
            response = requests.post(
                f"{self.reflection_vlm.base_url}/v1/chat/completions",
                json=payload,
                timeout=300,
            )
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]

            # Extract instruction from ``` blocks
            return self._extract_instruction(content)

        except Exception as e:
            print(f"  [Reflection] VLM call failed: {e}")
            return None

    def _single_image_reflection(self, text_prompt: str, image) -> Optional[str]:
        """Fallback: reflection with single image."""
        try:
            response = self.reflection_vlm.query(
                image_path=image,
                question=text_prompt,
                max_tokens=self.reflection_max_tokens,
            )
            return self._extract_instruction(response)
        except Exception as e:
            print(f"  [Reflection] Single image reflection failed: {e}")
            return None

    def _extract_instruction(self, lm_output: str) -> Optional[str]:
        """Extract instruction text from ``` blocks."""
        import re

        # Find content between ``` blocks
        start = lm_output.find("```")
        if start == -1:
            # No code blocks, return stripped output
            return lm_output.strip() if lm_output.strip() else None

        start += 3
        # Skip optional language specifier
        if lm_output[start:start+1] != '\n':
            newline_pos = lm_output.find('\n', start)
            if newline_pos != -1:
                start = newline_pos + 1

        end = lm_output.rfind("```")
        if end <= start:
            # Incomplete block
            return lm_output[start:].strip()

        return lm_output[start:end].strip()


class MockVLM:
    """
    Mock VLM for testing the adapter without loading a real model.
    """

    def query(
        self,
        image_path: str,
        question: str,
        system_prompt: str = None,
        max_new_tokens: int = 50,
        max_tokens: int = None,
    ) -> str:
        """Generate a mock response based on keywords in the question."""
        question_lower = question.lower()

        if "color" in question_lower:
            return "red"
        elif "how many" in question_lower or "count" in question_lower:
            return "3"
        elif "what is" in question_lower and "animal" in question_lower:
            return "dog"
        elif "yes" in question_lower or "no" in question_lower:
            return "yes"
        elif "where" in question_lower:
            return "outside"
        else:
            return "unknown"


if __name__ == "__main__":
    from vqa_dataset import create_mock_dataset

    print("Testing VQA Adapter with mock VLM...")

    # Create mock components
    mock_vlm = MockVLM()
    adapter = VQAAdapter(vlm=mock_vlm)  # type: ignore

    # Create mock data
    mock_data = create_mock_dataset(5)

    # Test candidate
    candidate = {"system_prompt": "Answer the question briefly using a single word or phrase."}

    # Test evaluate
    print("\n1. Testing evaluate()...")
    eval_result = adapter.evaluate(mock_data, candidate, capture_traces=True)
    print(f"   Outputs: {len(eval_result.outputs)}")
    print(f"   Scores: {eval_result.scores}")
    print(f"   Average score: {sum(eval_result.scores) / len(eval_result.scores):.2f}")

    # Test make_reflective_dataset
    print("\n2. Testing make_reflective_dataset()...")
    reflective_data = adapter.make_reflective_dataset(
        candidate=candidate,
        eval_batch=eval_result,
        components_to_update=["system_prompt"],
    )

    print(f"   Components: {list(reflective_data.keys())}")
    print(f"   Records per component: {len(reflective_data['system_prompt'])}")
    print("\n   Sample record:")
    sample = reflective_data["system_prompt"][0]
    for key, value in sample.items():
        if key != "image":
            print(f"     {key}: {value[:80]}..." if len(str(value)) > 80 else f"     {key}: {value}")
        else:
            print(f"     {key}: [image data]")

    print("\nAdapter test complete!")

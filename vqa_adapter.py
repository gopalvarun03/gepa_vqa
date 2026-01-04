"""
GEPA Adapter for VQA Optimization.

This module implements the GEPAAdapter protocol for Visual Question Answering,
allowing GEPA to optimize VQA system prompts.
"""

import sys
from pathlib import Path
from typing import Any, TypedDict
from collections.abc import Mapping, Sequence

# Add GEPA to path
GEPA_PATH = Path(__file__).parent.parent / "gepa" / "src"
sys.path.insert(0, str(GEPA_PATH))

from gepa.core.adapter import EvaluationBatch, GEPAAdapter

from vqa_dataset import VQADataInst, compute_vqa_accuracy
from vlm_wrapper import InternVLWrapper


class VQATrajectory(TypedDict):
    """
    Trajectory data captured during VQA evaluation.

    Contains all information needed for reflection.
    """
    data: VQADataInst           # Original input data
    system_prompt: str          # System prompt used
    vlm_response: str           # Raw VLM response
    score: float                # VQA accuracy score


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


class VQAAdapter(GEPAAdapter[VQADataInst, VQATrajectory, VQARolloutOutput]):
    """
    GEPA Adapter for VQA prompt optimization.

    This adapter:
    1. Takes a candidate system prompt
    2. Runs the VLM on VQA examples with that prompt
    3. Computes VQA accuracy scores
    4. Generates reflective datasets for prompt improvement

    Usage:
        vlm = InternVLWrapper(model_path="path/to/model")
        adapter = VQAAdapter(vlm=vlm)

        result = gepa.optimize(
            seed_candidate={"system_prompt": "Answer briefly."},
            trainset=train_data,
            valset=val_data,
            adapter=adapter,
            reflection_lm="openai/gpt-4",
            max_metric_calls=100,
        )
    """

    def __init__(
        self,
        vlm: InternVLWrapper,
        component_name: str = "system_prompt",
        max_new_tokens: int = 50,
        include_image_description: bool = True,
    ):
        """
        Initialize the VQA adapter.

        Args:
            vlm: InternVL wrapper instance
            component_name: Name of the text component being optimized
            max_new_tokens: Maximum tokens for VLM generation
            include_image_description: Whether to include image path in reflective data
        """
        self.vlm = vlm
        self.component_name = component_name
        self.max_new_tokens = max_new_tokens
        self.include_image_description = include_image_description

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

            # Capture trajectory if needed
            if trajectories is not None:
                trajectory = VQATrajectory(
                    data=data,
                    system_prompt=system_prompt,
                    vlm_response=vlm_response,
                    score=score,
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
        Build a reflective dataset for prompt improvement.

        This dataset is fed to the reflection LLM to propose improved prompts.

        Args:
            candidate: Current candidate prompt
            eval_batch: Results from evaluate() with capture_traces=True
            components_to_update: List of component names to update

        Returns:
            Mapping from component name to list of reflective records
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

                # Build input description
                if self.include_image_description:
                    inputs = f"Question: {data['question']}\nImage: {data['image_path']}"
                else:
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
                }
                records.append(record)

            result[component] = records

        if not any(result.values()):
            raise ValueError("No valid predictions found for any component.")

        return result


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
        print(f"     {key}: {value[:80]}..." if len(str(value)) > 80 else f"     {key}: {value}")

    print("\nAdapter test complete!")

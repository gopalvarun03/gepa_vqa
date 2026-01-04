"""
GEPA Adapter for MMLU Optimization.

This module implements the GEPAAdapter protocol for MMLU,
allowing GEPA to optimize prompts for multiple choice QA.
"""

import sys
from pathlib import Path
from typing import Any, TypedDict, Protocol
from collections.abc import Mapping, Sequence

# Add GEPA to path
GEPA_PATH = Path(__file__).parent.parent / "gepa" / "src"
sys.path.insert(0, str(GEPA_PATH))

from gepa.core.adapter import EvaluationBatch, GEPAAdapter

from mmlu_dataset import MMLUDataInst, compute_mmlu_accuracy, format_choices


class LLMProtocol(Protocol):
    """Protocol for LLM that can answer questions."""
    def query(self, prompt: str) -> str:
        ...


class MMLUTrajectory(TypedDict):
    """Trajectory data captured during MMLU evaluation."""
    data: MMLUDataInst
    system_prompt: str
    formatted_question: str
    llm_response: str
    score: float


class MMLURolloutOutput(TypedDict):
    """Output from a single MMLU evaluation."""
    predicted_answer: str
    question_id: str


class MMLUAdapter(GEPAAdapter[MMLUDataInst, MMLUTrajectory, MMLURolloutOutput]):
    """
    GEPA Adapter for MMLU prompt optimization.

    This adapter:
    1. Takes a candidate system prompt
    2. Runs the LLM on MMLU questions with that prompt
    3. Computes accuracy scores
    4. Generates reflective datasets for prompt improvement
    """

    def __init__(
        self,
        llm: LLMProtocol,
        component_name: str = "system_prompt",
    ):
        """
        Initialize the MMLU adapter.

        Args:
            llm: LLM instance with a query(prompt) -> str method
            component_name: Name of the text component being optimized
        """
        self.llm = llm
        self.component_name = component_name

    def _format_question(self, data: MMLUDataInst, system_prompt: str) -> str:
        """Format the full prompt for the LLM."""
        choices_str = format_choices(data['choices'])

        prompt = f"""{system_prompt}

Question: {data['question']}

{choices_str}

Answer:"""
        return prompt

    def evaluate(
        self,
        batch: list[MMLUDataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[MMLUTrajectory, MMLURolloutOutput]:
        """
        Evaluate a candidate prompt on a batch of MMLU examples.
        """
        system_prompt = candidate.get(self.component_name, "")

        outputs: list[MMLURolloutOutput] = []
        scores: list[float] = []
        trajectories: list[MMLUTrajectory] | None = [] if capture_traces else None

        for data in batch:
            try:
                # Format and query
                formatted_question = self._format_question(data, system_prompt)
                llm_response = self.llm.query(formatted_question)

                # Compute accuracy
                score = compute_mmlu_accuracy(
                    llm_response,
                    data['answer_idx'],
                    data['choices']
                )

            except Exception as e:
                formatted_question = ""
                llm_response = f"[Error: {str(e)}]"
                score = 0.0

            output = MMLURolloutOutput(
                predicted_answer=llm_response,
                question_id=data['question_id'],
            )
            outputs.append(output)
            scores.append(score)

            if trajectories is not None:
                trajectory = MMLUTrajectory(
                    data=data,
                    system_prompt=system_prompt,
                    formatted_question=formatted_question,
                    llm_response=llm_response,
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
        eval_batch: EvaluationBatch[MMLUTrajectory, MMLURolloutOutput],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """
        Build a reflective dataset for prompt improvement.
        """
        assert eval_batch.trajectories is not None

        result: dict[str, list[dict[str, Any]]] = {}

        for component in components_to_update:
            records: list[dict[str, Any]] = []

            for trajectory in eval_batch.trajectories:
                data = trajectory["data"]
                llm_response = trajectory["llm_response"]
                score = trajectory["score"]

                # Build input description
                choices_str = format_choices(data['choices'])
                inputs = f"Subject: {data['subject']}\nQuestion: {data['question']}\n{choices_str}"

                # Build feedback
                correct_letter = ['A', 'B', 'C', 'D'][data['answer_idx']]
                if score >= 1.0:
                    feedback = (
                        f"CORRECT. The model answered '{llm_response.strip()}'. "
                        f"The correct answer is {correct_letter}. {data['answer']}."
                    )
                else:
                    feedback = (
                        f"INCORRECT. The model answered '{llm_response.strip()}'. "
                        f"The correct answer is {correct_letter}. {data['answer']}. "
                        f"The model should have selected option {correct_letter}."
                    )

                record: dict[str, Any] = {
                    "Inputs": inputs,
                    "Generated Outputs": llm_response,
                    "Feedback": feedback,
                }
                records.append(record)

            result[component] = records

        return result


class VLLMTextWrapper:
    """Wrapper to use vLLM for text-only queries (no images)."""

    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        max_tokens: int = 64,
    ):
        import requests
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.api_url = f"{self.base_url}/v1/chat/completions"
        self._requests = requests

        print(f"Initialized vLLM text wrapper at {self.base_url}")

    def query(self, prompt: str) -> str:
        """Query the LLM with a text prompt."""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
        }

        try:
            response = self._requests.post(
                self.api_url,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"LLM query failed: {e}")
            return ""


if __name__ == "__main__":
    from mmlu_dataset import load_mmlu

    print("Testing MMLU Adapter...")

    # Load data
    data_path = "/home/shivank_g/projects/vlm_reason/varun_codes/vqa_datasets/mmlu"
    data = load_mmlu(split="test", max_samples=3, data_path=data_path)

    # Create LLM wrapper
    llm = VLLMTextWrapper(base_url="http://localhost:8001")

    # Create adapter
    adapter = MMLUAdapter(llm=llm)

    # Test candidate
    candidate = {
        "system_prompt": "You are an expert at answering multiple choice questions. Select the correct answer by responding with just the letter (A, B, C, or D)."
    }

    # Evaluate
    print("\nEvaluating...")
    result = adapter.evaluate(data, candidate, capture_traces=True)

    print(f"\nScores: {result.scores}")
    print(f"Average: {sum(result.scores) / len(result.scores):.2%}")

    for i, traj in enumerate(result.trajectories):
        print(f"\n--- Example {i} ---")
        print(f"Subject: {traj['data']['subject']}")
        print(f"Question: {traj['data']['question'][:80]}...")
        print(f"Correct: {['A','B','C','D'][traj['data']['answer_idx']]}. {traj['data']['answer']}")
        print(f"Response: {traj['llm_response']}")
        print(f"Score: {traj['score']}")

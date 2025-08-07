"""
Core operations for Graph of Thoughts reasoning.

This module implements the fundamental operation types that form the building blocks
of reasoning graphs, inspired by the ETH Zurich Graph of Thoughts research.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class OperationType(Enum):
    """Operation types for Graph of Thoughts reasoning."""

    GENERATE = "generate"
    SCORE = "score"
    AGGREGATE = "aggregate"
    IMPROVE = "improve"
    VALIDATE = "validate"
    SELECT = "select"
    BRANCH = "branch"
    MERGE = "merge"


@dataclass
class Thought:
    """
    Represents a single thought state in the reasoning graph.

    A thought encapsulates the current state, score, and metadata
    at a specific point in the reasoning process.
    """

    state: dict[str, Any]
    score: float | None = None
    metadata: dict[str, Any] | None = None
    parent_thoughts: list['Thought'] | None = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}
        if self.parent_thoughts is None:
            self.parent_thoughts = []

    def to_dict(self) -> dict[str, Any]:
        """Convert thought to dictionary representation."""
        return {
            "state": self.state,
            "score": self.score,
            "metadata": self.metadata,
            "has_parents": len(self.parent_thoughts) > 0 if self.parent_thoughts is not None else False
        }


class Operation(ABC):
    """
    Abstract base class for all Graph of Thoughts operations.

    Operations are the fundamental building blocks that transform thoughts
    through the reasoning process.
    """

    def __init__(self, operation_type: OperationType):
        self.operation_type = operation_type
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

    @abstractmethod
    async def execute(
        self,
        thoughts: list[Thought],
        context: Any,
        **kwargs: Any
    ) -> list[Thought]:
        """
        Execute the operation on input thoughts.

        Args:
            thoughts: Input thoughts to process
            context: MCP context for sampling and interaction
            **kwargs: Additional operation-specific parameters

        Returns:
            List of output thoughts after operation
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.operation_type.value})"


class GenerateOperation(Operation):
    """
    Generate new thoughts through LLM sampling.

    This operation uses MCP sampling to generate new reasoning states
    based on the current thoughts and a generation prompt.
    """

    def __init__(
        self,
        num_thoughts: int = 1,
        prompt_template: str | None = None
    ):
        super().__init__(OperationType.GENERATE)
        self.num_thoughts = num_thoughts
        self.prompt_template = prompt_template or self._default_prompt_template()

    def _default_prompt_template(self) -> str:
        return """
Based on the current reasoning state, generate the next logical step in solving this problem.

Current State: {current_state}
Problem Context: {problem_context}

Provide your reasoning and the next step:
"""

    async def execute(
        self,
        thoughts: list[Thought],
        context: Any,
        **kwargs: Any
    ) -> list[Thought]:
        """Generate new thoughts using MCP sampling."""
        try:
            generated_thoughts = []

            for thought in thoughts:
                for i in range(self.num_thoughts):
                    # Format prompt with current thought state
                    prompt = self.prompt_template.format(
                        current_state=json.dumps(thought.state, indent=2),
                        problem_context=thought.state.get("problem", "")
                    )

                    self.logger.info(f"Generating thought {i+1}/{self.num_thoughts}")

                    # Use MCP sampling to generate new reasoning
                    response = await context.sample(prompt)

                    # Create new thought with generated content
                    new_thought = Thought(
                        state={
                            **thought.state,
                            "reasoning_step": response,
                            "generation_id": f"{id(thought)}_{i}"
                        },
                        parent_thoughts=[thought],
                        metadata={
                            "operation": "generate",
                            "prompt_used": prompt[:100] + "...",
                            "generation_index": i
                        }
                    )

                    generated_thoughts.append(new_thought)

            self.logger.info(f"Generated {len(generated_thoughts)} new thoughts")
            return generated_thoughts

        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"GenerateOperation failed: {e}") from e


class ScoreOperation(Operation):
    """
    Score thoughts based on quality metrics.

    This operation evaluates thoughts using either LLM-based scoring
    or custom scoring functions.
    """

    def __init__(
        self,
        scoring_function: Callable[[Thought], float] | None = None,
        use_llm_scoring: bool = True
    ):
        super().__init__(OperationType.SCORE)
        self.scoring_function = scoring_function
        self.use_llm_scoring = use_llm_scoring

    async def execute(
        self,
        thoughts: list[Thought],
        context: Any,
        **kwargs: Any
    ) -> list[Thought]:
        """Score thoughts using the configured method."""
        try:
            scored_thoughts = []

            for thought in thoughts:
                if self.scoring_function:
                    # Use custom scoring function
                    score = self.scoring_function(thought)
                elif self.use_llm_scoring:
                    # Use LLM-based scoring
                    score = await self._llm_score(thought, context)
                else:
                    # Default simple scoring
                    score = 0.5

                # Create new thought with score
                scored_thought = Thought(
                    state=thought.state,
                    score=score,
                    parent_thoughts=thought.parent_thoughts,
                    metadata={
                        **(thought.metadata or {}),
                        "scoring_method": "llm" if self.use_llm_scoring else "function",
                        "scored_at": "score_operation"
                    }
                )

                scored_thoughts.append(scored_thought)

            self.logger.info(f"Scored {len(scored_thoughts)} thoughts")
            return scored_thoughts

        except Exception as e:
            self.logger.error(f"Scoring failed: {e}")
            raise RuntimeError(f"ScoreOperation failed: {e}") from e

    async def _llm_score(self, thought: Thought, context: Any) -> float:
        """Use LLM to score a thought's quality."""
        scoring_prompt = f"""
Rate the quality and correctness of this reasoning step on a scale of 0.0 to 1.0.

Reasoning: {thought.state.get('reasoning_step', 'No reasoning provided')}
Problem Context: {thought.state.get('problem', 'No context')}

Consider:
- Logical consistency
- Relevance to the problem
- Clarity of reasoning
- Progress toward solution

Respond with just a number between 0.0 and 1.0:
"""

        response = await context.sample(scoring_prompt)

        try:
            # Extract score from response
            score_text = response.strip()
            score = float(score_text)
            return max(0.0, min(1.0, score))  # Clamp to valid range
        except ValueError:
            self.logger.warning(f"Could not parse score from: {response}")
            return 0.5  # Default score


class AggregateOperation(Operation):
    """
    Aggregate multiple thoughts into a unified representation.

    This operation combines insights from multiple reasoning paths
    to create a more comprehensive understanding.
    """

    def __init__(self, aggregation_strategy: str = "consensus"):
        super().__init__(OperationType.AGGREGATE)
        self.aggregation_strategy = aggregation_strategy

    async def execute(
        self,
        thoughts: list[Thought],
        context: Any,
        **kwargs: Any
    ) -> list[Thought]:
        """Aggregate thoughts using the specified strategy."""
        try:
            if not thoughts:
                return []

            if len(thoughts) == 1:
                return thoughts

            # Create aggregation prompt
            reasoning_steps = []
            for i, thought in enumerate(thoughts):
                reasoning_steps.append(
                    f"Reasoning {i+1}: {thought.state.get('reasoning_step', 'No reasoning')}"
                )

            aggregation_prompt = f"""
Analyze and synthesize the following reasoning steps into a unified, coherent conclusion:

{chr(10).join(reasoning_steps)}

Problem Context: {thoughts[0].state.get('problem', 'No context')}

Provide a synthesized reasoning that:
1. Incorporates the best insights from each approach
2. Resolves any contradictions or inconsistencies
3. Presents a unified path forward
4. Maintains logical coherence

Synthesized reasoning:
"""

            self.logger.info(f"Aggregating {len(thoughts)} thoughts")

            aggregated_response = await context.sample(aggregation_prompt)

            # Calculate aggregated score (average of input scores)
            scores = [t.score for t in thoughts if t.score is not None]
            avg_score = sum(scores) / len(scores) if scores else None

            # Create aggregated thought
            aggregated_thought = Thought(
                state={
                    **thoughts[0].state,  # Base state from first thought
                    "reasoning_step": aggregated_response,
                    "aggregated_from": len(thoughts)
                },
                score=avg_score,
                parent_thoughts=thoughts,
                metadata={
                    "operation": "aggregate",
                    "strategy": self.aggregation_strategy,
                    "source_thoughts": len(thoughts)
                }
            )

            self.logger.info("Successfully aggregated thoughts")
            return [aggregated_thought]

        except Exception as e:
            self.logger.error(f"Aggregation failed: {e}")
            raise RuntimeError(f"AggregateOperation failed: {e}") from e


class SelectOperation(Operation):
    """
    Select the best thoughts based on scoring criteria.

    This operation filters thoughts to keep only the highest-quality
    reasoning paths for continued processing.
    """

    def __init__(self, selection_count: int = 1, selection_strategy: str = "top_score"):
        super().__init__(OperationType.SELECT)
        self.selection_count = selection_count
        self.selection_strategy = selection_strategy

    async def execute(
        self,
        thoughts: list[Thought],
        context: Any,
        **kwargs: Any
    ) -> list[Thought]:
        """Select the best thoughts according to the strategy."""
        try:
            if not thoughts:
                return []

            if len(thoughts) <= self.selection_count:
                return thoughts

            if self.selection_strategy == "top_score":
                # Sort by score (descending) and take top N
                scored_thoughts = [t for t in thoughts if t.score is not None]
                if not scored_thoughts:
                    return thoughts[:self.selection_count]

                selected = sorted(
                    scored_thoughts,
                    key=lambda t: t.score if t.score is not None else 0.0,
                    reverse=True
                )[:self.selection_count]
            else:
                # Default: take first N thoughts
                selected = thoughts[:self.selection_count]

            self.logger.info(f"Selected {len(selected)} from {len(thoughts)} thoughts")
            return selected

        except Exception as e:
            self.logger.error(f"Selection failed: {e}")
            raise RuntimeError(f"SelectOperation failed: {e}") from e

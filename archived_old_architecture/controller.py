"""
Controller for Graph of Thoughts execution.

This module implements the core controller that manages the execution flow
of reasoning graphs, coordinating between operations, thoughts, and the MCP context.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from .operations import Operation, OperationType, Thought

logger = logging.getLogger(__name__)


@dataclass
class ReasoningGraph:
    """
    Represents a complete reasoning graph structure.

    A reasoning graph defines the sequence and relationships between
    operations that form a complete reasoning process.
    """

    operations: list[Operation] = field(default_factory=list)
    initial_state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_operation(self, operation: Operation) -> None:
        """Add an operation to the reasoning graph."""
        self.operations.append(operation)
        logger.debug(f"Added operation: {operation}")

    def to_dict(self) -> dict[str, Any]:
        """Convert graph to dictionary representation."""
        return {
            "operations": [op.operation_type.value for op in self.operations],
            "initial_state": self.initial_state,
            "metadata": self.metadata,
            "operation_count": len(self.operations)
        }


class ReasoningController:
    """
    Controller for executing Graph of Thoughts reasoning processes.

    The controller manages the execution flow, coordinates between operations,
    and maintains the reasoning state throughout the process.
    """

    def __init__(self, context: Any):
        """
        Initialize the reasoning controller.

        Args:
            context: MCP context for sampling and interaction
        """
        self.context = context
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.execution_history: list[dict[str, Any]] = []
        self.current_thoughts: list[Thought] = []

    async def execute_graph(
        self,
        graph: ReasoningGraph,
        problem_description: str,
        initial_data: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Execute a complete reasoning graph.

        Args:
            graph: The reasoning graph to execute
            problem_description: Description of the problem to solve
            initial_data: Optional initial data for the reasoning process

        Returns:
            dictionary containing execution results and final thoughts
        """
        try:
            self.logger.info(f"Starting graph execution with {len(graph.operations)} operations")

            # Initialize with starting thought
            initial_state = {
                "problem": problem_description,
                "initial_data": initial_data or {},
                **graph.initial_state
            }

            starting_thought = Thought(
                state=initial_state,
                metadata={"operation": "initial", "step": 0}
            )

            self.current_thoughts = [starting_thought]
            self.execution_history = []

            # Execute each operation in sequence
            for i, operation in enumerate(graph.operations):
                step_result = await self._execute_operation(operation, i)
                self.execution_history.append(step_result)

                if not self.current_thoughts:
                    raise RuntimeError(f"No thoughts remaining after operation {i}: {operation}")

            # Compile final results
            final_results = self._compile_results(graph, problem_description)

            self.logger.info("Graph execution completed successfully")
            return final_results

        except Exception as e:
            self.logger.error(f"Graph execution failed: {e}")
            error_result = {
                "success": False,
                "error": str(e),
                "problem": problem_description,
                "steps_completed": len(self.execution_history),
                "final_thoughts": [t.to_dict() for t in self.current_thoughts]
            }
            return error_result

    async def _execute_operation(self, operation: Operation, step_index: int) -> dict[str, Any]:
        """Execute a single operation and update current thoughts."""
        try:
            self.logger.info(f"Executing step {step_index}: {operation}")

            # Record pre-execution state
            pre_thoughts = len(self.current_thoughts)

            # Execute the operation
            result_thoughts = await operation.execute(
                self.current_thoughts,
                self.context
            )

            # Update current thoughts
            self.current_thoughts = result_thoughts

            # Record execution results
            step_result = {
                "step": step_index,
                "operation": operation.operation_type.value,
                "pre_thought_count": pre_thoughts,
                "post_thought_count": len(result_thoughts),
                "thoughts": [t.to_dict() for t in result_thoughts[:3]]  # Limit for brevity
            }

            self.logger.info(
                f"Step {step_index} completed: {pre_thoughts} → {len(result_thoughts)} thoughts"
            )

            return step_result

        except Exception as e:
            self.logger.error(f"Operation execution failed at step {step_index}: {e}")
            raise RuntimeError(f"Step {step_index} failed: {e}") from e

    def _compile_results(
        self,
        graph: ReasoningGraph,
        problem_description: str
    ) -> dict[str, Any]:
        """Compile final execution results."""

        # Get best final thought (highest scored if available)
        best_thought = self._get_best_thought(self.current_thoughts)

        # Extract final reasoning and solution
        final_reasoning = best_thought.state.get("reasoning_step", "No reasoning available")
        final_score = best_thought.score

        # Compile comprehensive results
        results = {
            "success": True,
            "problem": problem_description,
            "final_reasoning": final_reasoning,
            "final_score": final_score,
            "graph_metadata": graph.to_dict(),
            "execution_steps": len(self.execution_history),
            "final_thought_count": len(self.current_thoughts),
            "best_thought": best_thought.to_dict(),
            "execution_history": self.execution_history[-5:],  # Last 5 steps for brevity
            "all_final_thoughts": [t.to_dict() for t in self.current_thoughts]
        }

        return results

    def _get_best_thought(self, thoughts: list[Thought]) -> Thought:
        """Get the best thought from a list based on scoring."""
        if not thoughts:
            raise ValueError("No thoughts available")

        # If thoughts have scores, return highest scored
        scored_thoughts = [t for t in thoughts if t.score is not None]
        if scored_thoughts:
            return max(scored_thoughts, key=lambda t: t.score if t.score is not None else float('-inf'))

        # Otherwise return the last thought (most recent)
        return thoughts[-1]


class GraphBuilder:
    """
    Builder for constructing Graph of Thoughts reasoning graphs.

    Provides a fluent interface for building complex reasoning workflows
    with multiple operations and configurations.
    """

    def __init__(self):
        self.graph = ReasoningGraph()

    def generate(
        self,
        num_thoughts: int = 1,
        prompt_template: str | None = None
    ) -> 'GraphBuilder':
        """Add a generation operation to the graph."""
        from .operations import GenerateOperation

        operation = GenerateOperation(
            num_thoughts=num_thoughts,
            prompt_template=prompt_template
        )
        self.graph.add_operation(operation)
        return self

    def score(
        self,
        scoring_function: Any | None = None,
        use_llm_scoring: bool = True
    ) -> 'GraphBuilder':
        """Add a scoring operation to the graph."""
        from .operations import ScoreOperation

        operation = ScoreOperation(
            scoring_function=scoring_function,
            use_llm_scoring=use_llm_scoring
        )
        self.graph.add_operation(operation)
        return self

    def aggregate(self, strategy: str = "consensus") -> 'GraphBuilder':
        """Add an aggregation operation to the graph."""
        from .operations import AggregateOperation

        operation = AggregateOperation(aggregation_strategy=strategy)
        self.graph.add_operation(operation)
        return self

    def select(
        self,
        count: int = 1,
        strategy: str = "top_score"
    ) -> 'GraphBuilder':
        """Add a selection operation to the graph."""
        from .operations import SelectOperation

        operation = SelectOperation(
            selection_count=count,
            selection_strategy=strategy
        )
        self.graph.add_operation(operation)
        return self

    def with_initial_state(self, state: dict[str, Any]) -> 'GraphBuilder':
        """Set initial state for the reasoning graph."""
        self.graph.initial_state.update(state)
        return self

    def with_metadata(self, metadata: dict[str, Any]) -> 'GraphBuilder':
        """Add metadata to the reasoning graph."""
        self.graph.metadata.update(metadata)
        return self

    def build(self) -> ReasoningGraph:
        """Build and return the completed reasoning graph."""
        if not self.graph.operations:
            raise ValueError("Graph must contain at least one operation")

        return self.graph


# Pre-built reasoning patterns inspired by ETH Zurich GoT research
class ReasoningPatterns:
    """
    Pre-built reasoning patterns for common problem types.

    These patterns are inspired by the Graph of Thoughts research
    and provide ready-to-use reasoning workflows.
    """

    @staticmethod
    def chain_of_thought() -> ReasoningGraph:
        """Simple chain-of-thought reasoning pattern."""
        return (GraphBuilder()
                .generate(num_thoughts=1)
                .score(use_llm_scoring=True)
                .build())

    @staticmethod
    def tree_of_thoughts() -> ReasoningGraph:
        """Tree-of-thoughts with branching and selection."""
        return (GraphBuilder()
                .generate(num_thoughts=3)
                .score(use_llm_scoring=True)
                .select(count=2, strategy="top_score")
                .generate(num_thoughts=2)
                .score(use_llm_scoring=True)
                .select(count=1, strategy="top_score")
                .build())

    @staticmethod
    def graph_of_thoughts() -> ReasoningGraph:
        """Full graph-of-thoughts with aggregation."""
        return (GraphBuilder()
                .generate(num_thoughts=4)
                .score(use_llm_scoring=True)
                .select(count=3, strategy="top_score")
                .aggregate(strategy="consensus")
                .score(use_llm_scoring=True)
                .generate(num_thoughts=2)
                .score(use_llm_scoring=True)
                .select(count=1, strategy="top_score")
                .build())

    @staticmethod
    def iterative_refinement() -> ReasoningGraph:
        """Iterative refinement pattern for improvement."""
        return (GraphBuilder()
                .generate(num_thoughts=1)
                .score(use_llm_scoring=True)
                .generate(num_thoughts=2)  # Generate improvements
                .score(use_llm_scoring=True)
                .select(count=1, strategy="top_score")
                .generate(num_thoughts=1)  # Final refinement
                .score(use_llm_scoring=True)
                .build())

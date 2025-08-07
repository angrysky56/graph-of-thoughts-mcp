"""
Graph of Thoughts MCP Server

A sophisticated Model Context Protocol server that provides Graph of Thoughts
reasoning frameworks and methodologies for complex problem-solving through
structured reasoning patterns.

This server provides tools and guidance to help LLMs apply Graph of Thoughts
reasoning patterns effectively, rather than attempting to perform reasoning itself.
"""

import atexit
import json
import logging
import signal
import sys
import traceback
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import Context, FastMCP

# Configure logging to stderr for MCP servers
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("graph-of-thoughts")


@mcp.tool()
async def generate_reasoning_framework(
    ctx: Context,
    problem_description: str,
    reasoning_pattern: str = "graph_of_thoughts",
    complexity_level: str = "moderate",
    domain_context: str | None = None
) -> str:
    """
    Generate a structured reasoning framework for complex problem-solving.

    This tool provides a complete reasoning framework based on Graph of Thoughts
    methodology, giving you a structured approach to tackle complex problems
    through interconnected reasoning processes.

    Args:
        problem_description: The complex problem or question to create a framework for
        reasoning_pattern: Reasoning pattern to use:
            - "chain_of_thought": Linear step-by-step reasoning
            - "tree_of_thoughts": Branching exploration with selection
            - "graph_of_thoughts": Full interconnected reasoning with synthesis
            - "iterative_refinement": Progressive improvement approach
        complexity_level: Problem complexity ("simple", "moderate", "complex", "highly_complex")
        domain_context: Optional domain-specific context (e.g., "business", "scientific", "technical")

    Returns:
        Structured reasoning framework with step-by-step guidance
    """
    try:
        logger.info(f"Generating {reasoning_pattern} framework for problem complexity: {complexity_level}")

        # Get pattern-specific framework
        framework = _generate_pattern_framework(
            reasoning_pattern,
            problem_description,
            complexity_level,
            domain_context
        )

        # Add meta-cognitive guidance
        meta_guidance = _generate_meta_cognitive_guidance(reasoning_pattern, complexity_level)

        # Compile comprehensive framework
        response = {
            "reasoning_framework": reasoning_pattern,
            "problem": problem_description,
            "complexity_level": complexity_level,
            "domain_context": domain_context,
            "structured_approach": framework,
            "meta_cognitive_guidance": meta_guidance,
            "evaluation_criteria": _generate_evaluation_criteria(reasoning_pattern),
            "quality_checkpoints": _generate_quality_checkpoints(reasoning_pattern),
            "usage_instructions": "Follow the structured approach step-by-step, using the meta-cognitive guidance to ensure quality reasoning at each stage."
        }

        logger.info(f"Successfully generated {reasoning_pattern} framework")
        return json.dumps(response, indent=2, ensure_ascii=False)

    except Exception as e:
        error_msg = f"Framework generation failed: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())

        error_response = {
            "reasoning_framework": reasoning_pattern,
            "problem": problem_description,
            "success": False,
            "error": error_msg,
            "fallback_advice": "Try using a simpler reasoning pattern or breaking the problem into smaller components."
        }

        return json.dumps(error_response, indent=2, ensure_ascii=False)


@mcp.tool()
async def structure_reasoning_step(
    ctx: Context,
    current_step: str,
    step_type: str,
    previous_thoughts: list[str] | None = None,
    target_insights: int = 3
) -> str:
    """
    Structure a specific reasoning step within a Graph of Thoughts process.

    This tool helps you properly structure individual reasoning steps according
    to Graph of Thoughts principles, ensuring each step contributes effectively
    to the overall reasoning process.

    Args:
        current_step: Description of the current reasoning step
        step_type: Type of reasoning step:
            - "generate": Create new thoughts/approaches
            - "evaluate": Assess quality of existing thoughts
            - "synthesize": Combine multiple thoughts into unified insights
            - "select": Choose the best thoughts to continue with
            - "refine": Improve existing thoughts
        previous_thoughts: Optional list of previous thoughts to build upon
        target_insights: Number of insights/approaches to generate

    Returns:
        Structured step guidance with specific instructions
    """
    try:
        logger.info(f"Structuring {step_type} step with target insights: {target_insights}")

        step_guidance = _generate_step_guidance(
            step_type,
            current_step,
            previous_thoughts or [],
            target_insights
        )

        response = {
            "step_type": step_type,
            "current_step": current_step,
            "target_insights": target_insights,
            "structured_guidance": step_guidance,
            "execution_template": _get_execution_template(step_type),
            "quality_indicators": _get_step_quality_indicators(step_type),
            "connection_prompts": _get_connection_prompts(step_type, previous_thoughts),
            "next_step_suggestions": _suggest_next_steps(step_type)
        }

        logger.info(f"Successfully structured {step_type} step")
        return json.dumps(response, indent=2, ensure_ascii=False)

    except Exception as e:
        error_msg = f"Step structuring failed: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())

        return json.dumps({
            "step_type": step_type,
            "current_step": current_step,
            "success": False,
            "error": error_msg
        }, indent=2)


@mcp.tool()
async def evaluate_reasoning_quality(
    ctx: Context,
    reasoning_content: str,
    evaluation_criteria: list[str] | None = None,
    reasoning_pattern: str = "general"
) -> str:
    """
    Evaluate the quality of reasoning according to Graph of Thoughts principles.

    This tool provides structured evaluation criteria and guidance for assessing
    the quality, completeness, and effectiveness of reasoning processes.

    Args:
        reasoning_content: The reasoning content to evaluate
        evaluation_criteria: Specific criteria to focus on (optional)
        reasoning_pattern: The reasoning pattern being evaluated

    Returns:
        Detailed evaluation framework with assessment criteria
    """
    try:
        logger.info(f"Generating evaluation framework for {reasoning_pattern} reasoning")

        if evaluation_criteria is None:
            evaluation_criteria = [
                "logical_consistency",
                "completeness_of_exploration",
                "insight_quality",
                "connection_strength",
                "practical_applicability",
                "creative_synthesis"
            ]

        evaluation_framework = {
            "reasoning_content": reasoning_content[:500] + "..." if len(reasoning_content) > 500 else reasoning_content,
            "reasoning_pattern": reasoning_pattern,
            "evaluation_criteria": evaluation_criteria,
            "assessment_framework": _generate_assessment_framework(evaluation_criteria),
            "quality_metrics": _generate_quality_metrics(reasoning_pattern),
            "improvement_suggestions": _generate_improvement_framework(reasoning_pattern),
            "scoring_rubric": _generate_scoring_rubric(evaluation_criteria),
            "meta_evaluation": "Assess whether the reasoning demonstrates the interconnected, graph-like thinking that characterizes high-quality GoT reasoning."
        }

        logger.info("Successfully generated evaluation framework")
        return json.dumps(evaluation_framework, indent=2, ensure_ascii=False)

    except Exception as e:
        error_msg = f"Evaluation framework generation failed: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())

        return json.dumps({
            "reasoning_pattern": reasoning_pattern,
            "success": False,
            "error": error_msg
        }, indent=2)


@mcp.tool()
async def compare_reasoning_approaches(
    ctx: Context,
    problem_description: str,
    approaches_to_compare: list[str] | None = None
) -> str:
    """
    Generate frameworks for comparing different reasoning approaches.

    This tool provides structured guidance for systematically comparing
    different Graph of Thoughts reasoning patterns on the same problem.

    Args:
        problem_description: The problem to analyze with multiple approaches
        approaches_to_compare: List of reasoning patterns to compare

    Returns:
        Comparative analysis framework with evaluation structure
    """
    try:
        logger.info("Generating comparative reasoning framework")

        if approaches_to_compare is None:
            approaches_to_compare = [
                "chain_of_thought",
                "tree_of_thoughts",
                "graph_of_thoughts",
                "iterative_refinement"
            ]

        comparative_framework = {
            "problem_description": problem_description,
            "approaches_to_compare": approaches_to_compare,
            "comparison_structure": _generate_comparison_structure(approaches_to_compare),
            "evaluation_dimensions": _generate_comparison_dimensions(),
            "execution_sequence": _generate_execution_sequence(approaches_to_compare, problem_description),
            "analysis_framework": _generate_analysis_framework(),
            "decision_matrix": _generate_decision_matrix(approaches_to_compare),
            "synthesis_guidance": "After executing each approach, synthesize insights about when each method is most effective and how they complement each other."
        }

        logger.info("Successfully generated comparative framework")
        return json.dumps(comparative_framework, indent=2, ensure_ascii=False)

    except Exception as e:
        error_msg = f"Comparative framework generation failed: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())

        return json.dumps({
            "problem": problem_description,
            "success": False,
            "error": error_msg
        }, indent=2)


@mcp.resource("got://methodology-guide")
async def reasoning_methodology_guide() -> str:
    """
    Comprehensive guide to Graph of Thoughts reasoning methodology.
    """
    return """
# Graph of Thoughts (GoT) Reasoning Methodology

## Overview
Graph of Thoughts is a structured reasoning framework that enables complex problem-solving
through interconnected thought processes that go beyond linear Chain-of-Thought or
tree-based Tree-of-Thoughts approaches.

## Core Principles

### 1. Interconnected Reasoning
- Thoughts are vertices in a reasoning graph
- Dependencies and relationships are edges
- Multiple reasoning paths can inform each other
- Synthesis creates emergent insights

### 2. Reasoning Operations
- **Generate**: Create new thoughts and approaches
- **Evaluate**: Assess quality and relevance of thoughts
- **Synthesize**: Combine thoughts into unified insights
- **Select**: Choose the most promising paths
- **Refine**: Improve and elaborate on existing thoughts

### 3. Reasoning Patterns

#### Chain of Thought (Linear)
- Best for: Straightforward problems with clear progression
- Structure: Step 1 → Step 2 → Step 3 → Solution
- Advantages: Clear, fast, easy to follow
- Limitations: Limited exploration of alternatives

#### Tree of Thoughts (Branching)
- Best for: Problems requiring exploration of multiple approaches
- Structure: Generate options → Evaluate → Select best → Continue
- Advantages: Explores alternatives while maintaining focus
- Limitations: Limited cross-branch interaction

#### Graph of Thoughts (Interconnected)
- Best for: Complex problems requiring synthesis of multiple perspectives
- Structure: Generate multiple thoughts → Connect and synthesize → Refine
- Advantages: Rich interaction, comprehensive exploration, emergent insights
- Limitations: More complex to execute, requires careful management

#### Iterative Refinement (Progressive)
- Best for: Problems requiring optimization and improvement
- Structure: Initial approach → Evaluate → Refine → Repeat
- Advantages: Continuous improvement, builds on insights
- Limitations: May converge on local optima

## Implementation Guidelines

### Problem Assessment
1. **Complexity Level**: Simple → Chain, Moderate → Tree, Complex → Graph
2. **Domain Requirements**: Technical, Creative, Strategic, Analytical
3. **Time Constraints**: More complex patterns require more time
4. **Quality vs Speed**: Balance thoroughness with efficiency

### Execution Best Practices
1. **Clear Problem Definition**: Start with precise problem statement
2. **Structured Generation**: Create diverse, high-quality initial thoughts
3. **Rigorous Evaluation**: Assess thoughts against relevant criteria
4. **Thoughtful Synthesis**: Look for non-obvious connections and insights
5. **Quality Checkpoints**: Regularly assess reasoning quality and direction

### Quality Indicators
- **Logical Consistency**: Thoughts support each other coherently
- **Comprehensive Coverage**: Important aspects are addressed
- **Creative Synthesis**: Novel insights emerge from combinations
- **Practical Utility**: Solutions are actionable and effective
- **Robust Reasoning**: Conclusions withstand scrutiny

## Advanced Techniques

### Meta-Cognitive Monitoring
- Regularly assess reasoning quality and progress
- Identify when to switch between reasoning patterns
- Monitor for cognitive biases and blind spots
- Adjust approach based on intermediate results

### Cross-Pattern Integration
- Start with Tree-of-Thoughts for broad exploration
- Use Graph-of-Thoughts for deep synthesis
- Apply Iterative Refinement for optimization
- Combine patterns for maximum effectiveness

### Domain-Specific Adaptations
- **Business Strategy**: Emphasize stakeholder perspectives and trade-offs
- **Scientific Research**: Focus on evidence integration and hypothesis testing
- **Creative Problem-Solving**: Prioritize novel connections and innovative insights
- **Technical Analysis**: Ensure systematic coverage and logical rigor

## Research Foundation

This methodology is based on "Graph of Thoughts: Solving Elaborate Problems with
Large Language Models" (Besta et al., AAAI 2024), which demonstrated significant
improvements over traditional reasoning approaches on complex tasks.

The key insight is that reasoning benefits from both structured exploration and
rich interconnections between ideas, enabling more sophisticated problem-solving
capabilities than linear or purely tree-based approaches.
"""


@mcp.resource("got://pattern-templates")
async def reasoning_pattern_templates() -> str:
    """
    Ready-to-use templates for different Graph of Thoughts reasoning patterns.
    """
    return """
# Graph of Thoughts Pattern Templates

## Chain of Thought Template

### Structure
1. **Problem Analysis**: Break down the problem into components
2. **Sequential Reasoning**: Work through each component logically
3. **Integration**: Combine insights into coherent solution
4. **Validation**: Check solution against original problem

### Execution Prompt
"Let me work through this step-by-step:
1. First, I'll analyze what exactly we're trying to solve...
2. Then, I'll work through the key components...
3. Next, I'll integrate these insights...
4. Finally, I'll validate this solution..."

## Tree of Thoughts Template

### Structure
1. **Initial Generation**: Create 3-5 different approaches
2. **Evaluation Phase**: Assess each approach for viability
3. **Selection**: Choose 2-3 most promising approaches
4. **Development**: Elaborate on selected approaches
5. **Final Selection**: Choose the best developed approach

### Execution Prompt
"I'll explore multiple approaches to this problem:

**Approach 1**: [First perspective/method]
**Approach 2**: [Second perspective/method]
**Approach 3**: [Third perspective/method]

Now evaluating each approach:
- Approach 1: [Strengths and weaknesses]
- Approach 2: [Strengths and weaknesses]
- Approach 3: [Strengths and weaknesses]

Selecting the most promising approaches for further development..."

## Graph of Thoughts Template

### Structure
1. **Multi-Perspective Generation**: Create 4-6 diverse thoughts
2. **Connection Mapping**: Identify relationships between thoughts
3. **Synthesis Clusters**: Group related thoughts into insight clusters
4. **Cross-Cluster Integration**: Connect insights across clusters
5. **Emergent Solution**: Derive solution from integrated insights

### Execution Prompt
"I'll approach this through interconnected reasoning:

**Thought Network Generation**:
- Perspective A: [First angle]
- Perspective B: [Second angle]
- Perspective C: [Third angle]
- Perspective D: [Fourth angle]

**Connection Analysis**:
- How A relates to B: [Connection]
- How B informs C: [Connection]
- How C and D together suggest: [Synthesis]

**Integrated Insights**:
From this network of thoughts, I can see that..."

## Iterative Refinement Template

### Structure
1. **Initial Solution**: Create working solution
2. **Quality Assessment**: Identify strengths and weaknesses
3. **Targeted Improvement**: Address specific weaknesses
4. **Re-evaluation**: Assess improved solution
5. **Further Refinement**: Continue until satisfied

### Execution Prompt
"I'll develop and refine a solution iteratively:

**Initial Solution**: [First attempt]

**Assessment**: What works well? What could be improved?
- Strengths: [List strengths]
- Areas for improvement: [List weaknesses]

**Refined Solution**: [Improved version addressing weaknesses]

**Re-assessment**: [Evaluate improvements and identify remaining issues]

**Final Refinement**: [Further improvements if needed]"

## Custom Pattern Creation

### Template for Custom Patterns
1. **Define Problem Type**: What kinds of problems is this for?
2. **Specify Operations**: What reasoning operations will you use?
3. **Set Structure**: How will the operations be sequenced?
4. **Create Execution Guide**: How should someone follow this pattern?
5. **Establish Quality Criteria**: How will you know if it's working?

### Example Custom Pattern: "Stakeholder Synthesis"
For problems involving multiple stakeholder perspectives:

1. **Stakeholder Mapping**: Identify all relevant stakeholders
2. **Perspective Generation**: Generate reasoning from each stakeholder viewpoint
3. **Conflict Identification**: Find areas of disagreement or tension
4. **Common Ground Discovery**: Identify shared interests and values
5. **Integrated Solution**: Create solution addressing multiple perspectives
6. **Stakeholder Validation**: Check solution against each stakeholder's needs
"""


# Helper functions for generating frameworks and guidance

def _generate_pattern_framework(pattern: str, problem: str, complexity: str, domain: str | None) -> dict[str, Any]:
    """Generate framework structure for specific reasoning pattern."""

    frameworks = {
        "chain_of_thought": {
            "approach": "Linear step-by-step reasoning",
            "steps": [
                "1. Problem decomposition: Break the problem into clear, manageable components",
                "2. Sequential analysis: Work through each component in logical order",
                "3. Integration: Combine insights from each component",
                "4. Solution formulation: Develop comprehensive solution",
                "5. Validation: Check solution against original problem requirements"
            ],
            "execution_guidance": "Follow each step completely before moving to the next. Ensure each step builds logically on the previous one.",
            "quality_focus": "Logical flow and completeness"
        },

        "tree_of_thoughts": {
            "approach": "Branching exploration with systematic selection",
            "steps": [
                "1. Divergent generation: Create 3-5 different approaches or perspectives",
                "2. Evaluation matrix: Assess each approach against key criteria",
                "3. Selective pruning: Choose 2-3 most promising approaches",
                "4. Parallel development: Elaborate on selected approaches simultaneously",
                "5. Convergent selection: Choose best developed approach or combine insights"
            ],
            "execution_guidance": "Generate diverse approaches before evaluating. Be systematic in evaluation and selection.",
            "quality_focus": "Breadth of exploration and quality of selection"
        },

        "graph_of_thoughts": {
            "approach": "Interconnected reasoning with synthesis",
            "steps": [
                "1. Multi-perspective generation: Create 4-6 diverse thoughts from different angles",
                "2. Connection mapping: Identify relationships and dependencies between thoughts",
                "3. Cluster formation: Group related thoughts into insight clusters",
                "4. Cross-cluster synthesis: Find connections between different clusters",
                "5. Emergent insight extraction: Derive novel insights from thought interactions",
                "6. Integrated solution: Formulate solution incorporating all insights"
            ],
            "execution_guidance": "Look for non-obvious connections between thoughts. Allow emergent insights to guide solution development.",
            "quality_focus": "Richness of connections and emergent insights"
        },

        "iterative_refinement": {
            "approach": "Progressive improvement through cycles",
            "steps": [
                "1. Initial solution: Create working solution or approach",
                "2. Critical evaluation: Identify strengths, weaknesses, and gaps",
                "3. Targeted improvement: Address specific weaknesses systematically",
                "4. Re-evaluation: Assess improvements and identify remaining issues",
                "5. Further refinement: Continue cycles until quality threshold met"
            ],
            "execution_guidance": "Be honest about weaknesses in each iteration. Focus improvements on most critical issues first.",
            "quality_focus": "Progressive improvement and optimization"
        }
    }

    base_framework = frameworks.get(pattern, frameworks["chain_of_thought"])

    # Add complexity and domain adaptations
    if complexity == "highly_complex":
        base_framework["additional_guidance"] = "For highly complex problems, consider breaking into sub-problems and applying the pattern to each."

    if domain:
        base_framework["domain_considerations"] = f"Adapt the approach for {domain} context by focusing on domain-specific criteria and constraints."

    return base_framework


def _generate_meta_cognitive_guidance(pattern: str, complexity: str) -> dict[str, list[str]]:
    """Generate meta-cognitive guidance for monitoring reasoning quality."""

    guidance = {
        "monitoring_questions": [
            "Am I maintaining logical consistency across my reasoning?",
            "Have I considered multiple perspectives on this problem?",
            "Are my conclusions supported by my reasoning?",
            "What assumptions am I making that should be validated?"
        ],
        "quality_checkpoints": [
            "After initial analysis: Have I understood the problem correctly?",
            "During reasoning: Am I exploring sufficiently or getting stuck in one approach?",
            "Before concluding: Does my solution address the original problem?",
            "Final check: What could go wrong with this solution?"
        ],
        "bias_awareness": [
            "Confirmation bias: Am I only looking for evidence that supports my initial thoughts?",
            "Anchoring bias: Am I overly influenced by the first approach I considered?",
            "Availability bias: Am I giving too much weight to easily recalled examples?",
            "Overconfidence bias: Am I being appropriately uncertain about my conclusions?"
        ]
    }

    if pattern == "graph_of_thoughts":
        guidance["additional_monitoring"] = [
            "Are my thoughts genuinely interconnected or just listed separately?",
            "Have I found non-obvious connections that create new insights?",
            "Am I synthesizing effectively or just aggregating?"
        ]

    return guidance


def _generate_evaluation_criteria(pattern: str) -> list[str]:
    """Generate evaluation criteria specific to reasoning pattern."""

    base_criteria = [
        "logical_consistency",
        "completeness",
        "clarity",
        "evidence_support",
        "practical_applicability"
    ]

    pattern_specific = {
        "chain_of_thought": ["sequential_logic", "step_completeness"],
        "tree_of_thoughts": ["exploration_breadth", "selection_quality"],
        "graph_of_thoughts": ["connection_richness", "synthesis_quality", "emergent_insights"],
        "iterative_refinement": ["improvement_progression", "optimization_effectiveness"]
    }

    return base_criteria + pattern_specific.get(pattern, [])


def _generate_quality_checkpoints(pattern: str) -> dict[str, str]:
    """Generate quality checkpoints for reasoning process."""

    checkpoints = {
        "initial": "Is the problem clearly understood and well-defined?",
        "development": "Is the reasoning progressing logically and comprehensively?",
        "integration": "Are insights being properly connected and synthesized?",
        "conclusion": "Does the solution adequately address the original problem?",
        "validation": "What evidence supports or challenges this solution?"
    }

    if pattern == "graph_of_thoughts":
        checkpoints["synthesis"] = "Are genuine emergent insights emerging from thought connections?"

    return checkpoints


def _generate_step_guidance(step_type: str, current_step: str, previous_thoughts: list[str], target_insights: int) -> dict[str, Any]:
    """Generate guidance for specific reasoning step."""

    guidance_map = {
        "generate": {
            "objective": f"Create {target_insights} diverse, high-quality thoughts or approaches",
            "instructions": [
                "Think from different perspectives or angles",
                "Consider various methodologies or frameworks",
                "Include both conventional and creative approaches",
                "Ensure each thought is distinct and valuable"
            ],
            "quality_focus": "Diversity and originality of thoughts"
        },

        "evaluate": {
            "objective": "Assess the quality, relevance, and potential of existing thoughts",
            "instructions": [
                "Consider logical soundness and evidence support",
                "Assess relevance to the original problem",
                "Evaluate potential for leading to good solutions",
                "Look for complementary strengths across thoughts"
            ],
            "quality_focus": "Accuracy and comprehensiveness of evaluation"
        },

        "synthesize": {
            "objective": "Combine thoughts into unified, coherent insights",
            "instructions": [
                "Look for connections and patterns across thoughts",
                "Identify complementary aspects that strengthen each other",
                "Create new insights that emerge from combinations",
                "Maintain the best aspects of individual thoughts"
            ],
            "quality_focus": "Coherence and emergent value of synthesis"
        },

        "select": {
            "objective": "Choose the most promising thoughts to continue with",
            "instructions": [
                "Apply clear selection criteria consistently",
                "Balance quality with diversity of approaches",
                "Consider potential for further development",
                "Document rationale for selections made"
            ],
            "quality_focus": "Justification and strategic value of selections"
        },

        "refine": {
            "objective": "Improve and elaborate on existing thoughts",
            "instructions": [
                "Identify specific weaknesses or gaps to address",
                "Add detail, evidence, or logical support",
                "Consider edge cases and potential objections",
                "Enhance clarity and actionability"
            ],
            "quality_focus": "Substantive improvement and elaboration"
        }
    }

    base_guidance = guidance_map.get(step_type, guidance_map["generate"])

    # Add context from previous thoughts
    if previous_thoughts:
        base_guidance["context_consideration"] = f"Build upon these previous thoughts: {', '.join(previous_thoughts[:3])}..."

    return base_guidance


def _get_execution_template(step_type: str) -> str:
    """Get execution template for step type."""

    templates = {
        "generate": """
For this {step_type} step:
1. Perspective 1: [First approach/angle]
2. Perspective 2: [Second approach/angle]
3. Perspective 3: [Third approach/angle]
[Continue for target number of insights]
""",

        "evaluate": """
Evaluating each thought against key criteria:

Thought 1: [Summary]
- Strengths: [List strengths]
- Weaknesses: [List weaknesses]
- Potential: [Assessment of potential]

[Continue for each thought]
""",

        "synthesize": """
Identifying connections and synthesizing insights:

Connection 1: How [Thought A] and [Thought B] combine to suggest [New insight]
Connection 2: The pattern across [Thoughts X, Y, Z] indicates [Pattern insight]

Synthesized insight: [Combined understanding that emerges]
""",

        "select": """
Selection based on criteria:

Selected thoughts:
1. [Thought] - Selected because: [Rationale]
2. [Thought] - Selected because: [Rationale]

Not selected:
- [Thought] - Reason: [Why not selected]
""",

        "refine": """
Refining selected thoughts:

Original: [Original thought]
Identified improvements needed: [Specific gaps or weaknesses]
Refined version: [Improved thought with enhancements]
Enhancement rationale: [Why these improvements help]
"""
    }

    return templates.get(step_type, templates["generate"])


def _get_step_quality_indicators(step_type: str) -> list[str]:
    """Get quality indicators for step type."""

    indicators = {
        "generate": [
            "Thoughts are diverse and cover different angles",
            "Each thought is substantive and well-developed",
            "Creative or non-obvious approaches are included",
            "All thoughts are relevant to the core problem"
        ],

        "evaluate": [
            "Assessment criteria are clearly stated",
            "Evaluation is consistent across all thoughts",
            "Both strengths and weaknesses are identified",
            "Potential for development is considered"
        ],

        "synthesize": [
            "Genuine connections are identified, not just similarities",
            "New insights emerge that weren't present in individual thoughts",
            "Synthesis maintains logical coherence",
            "Combined insight is more valuable than sum of parts"
        ],

        "select": [
            "Selection criteria are explicit and justified",
            "Rationale for each selection/rejection is clear",
            "Diversity of selected approaches is maintained",
            "Strategic value of selections is evident"
        ],

        "refine": [
            "Specific improvements are clearly identified",
            "Refinements address real weaknesses or gaps",
            "Enhanced version is substantively better",
            "Original strengths are preserved"
        ]
    }

    return indicators.get(step_type, indicators["generate"])


def _get_connection_prompts(step_type: str, previous_thoughts: list[str] | None) -> list[str]:
    """Generate prompts for connecting to previous thoughts."""

    if not previous_thoughts:
        return ["This is an initial step - focus on establishing strong foundations."]

    prompts = {
        "generate": [
            f"How can I build upon these previous insights: {previous_thoughts[:2]}?",
            "What new angles do the previous thoughts suggest?",
            "What gaps do I see that need to be addressed?"
        ],

        "evaluate": [
            "How do these thoughts relate to and strengthen previous insights?",
            "Which thoughts best build upon our established foundation?",
            "What evaluation criteria matter most given our previous work?"
        ],

        "synthesize": [
            "How do these thoughts connect with our previous insights?",
            "What patterns emerge when I combine new and previous thoughts?",
            "What unified understanding is emerging across all our work?"
        ],

        "select": [
            "Which thoughts best complement our previous insights?",
            "What combination of thoughts creates the strongest foundation?",
            "How do my selections advance our overall reasoning progress?"
        ],

        "refine": [
            "How can I strengthen the connections to our previous work?",
            "What refinements make these thoughts more coherent with previous insights?",
            "How can I enhance the overall logical flow of our reasoning?"
        ]
    }

    return prompts.get(step_type, prompts["generate"])


def _suggest_next_steps(step_type: str) -> list[str]:
    """Suggest logical next steps after current step type."""

    next_steps = {
        "generate": [
            "Evaluate the generated thoughts for quality and relevance",
            "Look for connections and patterns among the thoughts",
            "Select the most promising thoughts for further development"
        ],

        "evaluate": [
            "Select the highest-quality thoughts based on evaluation",
            "Synthesize insights from multiple evaluated thoughts",
            "Refine thoughts that show promise but have identifiable weaknesses"
        ],

        "synthesize": [
            "Evaluate the quality of synthesized insights",
            "Refine the synthesis for greater clarity or completeness",
            "Generate new thoughts inspired by synthesized insights"
        ],

        "select": [
            "Refine and elaborate on selected thoughts",
            "Generate additional thoughts building on selections",
            "Synthesize selected thoughts into comprehensive approach"
        ],

        "refine": [
            "Evaluate the improved thoughts against original criteria",
            "Synthesize refined thoughts with other insights",
            "Consider if further refinement cycles are needed"
        ]
    }

    return next_steps.get(step_type, next_steps["generate"])


def _generate_assessment_framework(criteria: list[str]) -> dict[str, dict[str, str]]:
    """Generate assessment framework for evaluation criteria."""

    frameworks = {}

    for criterion in criteria:
        if criterion == "logical_consistency":
            frameworks[criterion] = {
                "focus": "Internal logical coherence and absence of contradictions",
                "questions": [
                    "Do the conclusions follow logically from the premises?",
                    "Are there any internal contradictions?",
                    "Is the reasoning chain sound throughout?"
                ],
                "indicators": "Clear logical flow, consistent use of terms, valid inferences"
            }
        elif criterion == "completeness_of_exploration":
            frameworks[criterion] = {
                "focus": "Thoroughness of investigation and coverage of important aspects",
                "questions": [
                    "Have all important aspects been considered?",
                    "Are there obvious gaps in the analysis?",
                    "Has sufficient depth been achieved?"
                ],
                "indicators": "Comprehensive coverage, multiple perspectives, adequate depth"
            }
        elif criterion == "insight_quality":
            frameworks[criterion] = {
                "focus": "Originality, depth, and value of insights generated",
                "questions": [
                    "Are the insights novel and non-obvious?",
                    "Do insights provide genuine understanding?",
                    "How valuable are these insights for solving the problem?"
                ],
                "indicators": "Creative connections, deep understanding, actionable insights"
            }
        elif criterion == "connection_strength":
            frameworks[criterion] = {
                "focus": "Quality and meaningfulness of connections between ideas",
                "questions": [
                    "Are connections meaningful rather than superficial?",
                    "Do connections create new understanding?",
                    "How well do connected ideas support each other?"
                ],
                "indicators": "Meaningful relationships, mutual reinforcement, emergent understanding"
            }
        elif criterion == "practical_applicability":
            frameworks[criterion] = {
                "focus": "Real-world utility and implementability of solutions",
                "questions": [
                    "Can this solution be realistically implemented?",
                    "Are practical constraints adequately considered?",
                    "How useful is this for real-world application?"
                ],
                "indicators": "Feasible implementation, consideration of constraints, clear utility"
            }
        else:
            # Generic framework for other criteria
            frameworks[criterion] = {
                "focus": f"Assessment of {criterion.replace('_', ' ')}",
                "questions": [f"How well does the reasoning demonstrate {criterion.replace('_', ' ')}?"],
                "indicators": f"Clear evidence of {criterion.replace('_', ' ')}"
            }

    return frameworks


def _generate_quality_metrics(pattern: str) -> dict[str, str]:
    """Generate quality metrics specific to reasoning pattern."""

    base_metrics = {
        "coherence": "Overall logical consistency and flow",
        "completeness": "Coverage of important aspects and perspectives",
        "insight_value": "Quality and usefulness of insights generated",
        "solution_quality": "Effectiveness of final solution or conclusion"
    }

    pattern_metrics = {
        "chain_of_thought": {
            "sequential_logic": "Quality of step-by-step logical progression",
            "step_completeness": "Thoroughness of each reasoning step"
        },
        "tree_of_thoughts": {
            "exploration_breadth": "Diversity of approaches explored",
            "selection_wisdom": "Quality of choices made at decision points"
        },
        "graph_of_thoughts": {
            "connection_richness": "Meaningfulness of inter-thought connections",
            "synthesis_emergence": "Quality of emergent insights from combinations",
            "network_coherence": "Overall coherence of thought network"
        },
        "iterative_refinement": {
            "improvement_trajectory": "Quality of progressive improvements",
            "convergence_effectiveness": "How well refinements converge on solution"
        }
    }

    combined_metrics = {**base_metrics, **pattern_metrics.get(pattern, {})}
    return combined_metrics


def _generate_improvement_framework(pattern: str) -> dict[str, list[str]]:
    """Generate framework for improving reasoning quality."""

    general_improvements = {
        "logical_strengthening": [
            "Identify and address logical gaps or weak inferences",
            "Strengthen connections between premises and conclusions",
            "Add supporting evidence or reasoning where needed"
        ],
        "depth_enhancement": [
            "Explore implications more thoroughly",
            "Consider additional perspectives or angles",
            "Dig deeper into cause-and-effect relationships"
        ],
        "clarity_improvement": [
            "Make reasoning steps more explicit",
            "Clarify assumptions and premises",
            "Improve organization and flow"
        ]
    }

    pattern_specific = {
        "graph_of_thoughts": {
            "connection_enhancement": [
                "Look for additional meaningful connections between thoughts",
                "Strengthen weak connections with better reasoning",
                "Identify missing connections that could provide insights"
            ],
            "synthesis_improvement": [
                "Deepen synthesis by finding non-obvious combinations",
                "Create more coherent unified understanding",
                "Generate novel insights from thought interactions"
            ]
        }
    }

    improvements = general_improvements.copy()
    if pattern in pattern_specific:
        improvements.update(pattern_specific[pattern])

    return improvements


def _generate_scoring_rubric(criteria: list[str]) -> dict[str, dict[str, str]]:
    """Generate scoring rubric for evaluation criteria."""

    rubric = {}

    for criterion in criteria:
        rubric[criterion] = {
            "excellent": f"Demonstrates exceptional {criterion.replace('_', ' ')} with clear evidence",
            "good": f"Shows solid {criterion.replace('_', ' ')} with minor gaps",
            "adequate": f"Meets basic requirements for {criterion.replace('_', ' ')}",
            "needs_improvement": f"Significant gaps in {criterion.replace('_', ' ')} that should be addressed"
        }

    return rubric


def _generate_comparison_structure(approaches: list[str]) -> dict[str, Any]:
    """Generate structure for comparing reasoning approaches."""

    return {
        "execution_framework": {
            "step": "Apply each reasoning approach systematically to the same problem",
            "documentation": "Record the process, insights, and solutions from each approach",
            "standardization": "Use consistent evaluation criteria across all approaches"
        },
        "comparison_dimensions": [
            "Solution quality and completeness",
            "Reasoning depth and sophistication",
            "Creative insights generated",
            "Practical applicability",
            "Efficiency of process",
            "Ease of execution"
        ],
        "analysis_structure": {
            "individual_assessment": "Evaluate each approach on its own merits",
            "comparative_analysis": "Identify relative strengths and weaknesses",
            "situational_optimization": "Determine when each approach works best",
            "integration_opportunities": "Explore how approaches might be combined"
        }
    }


def _generate_comparison_dimensions() -> dict[str, str]:
    """Generate dimensions for comparing reasoning approaches."""

    return {
        "solution_quality": "How effective and complete is the final solution?",
        "process_efficiency": "How much time and effort does the approach require?",
        "insight_generation": "How many valuable insights does the approach produce?",
        "creative_potential": "How well does the approach enable creative problem-solving?",
        "systematic_rigor": "How systematic and rigorous is the reasoning process?",
        "practical_utility": "How useful is the approach for real-world problems?",
        "scalability": "How well does the approach work for complex problems?",
        "ease_of_execution": "How easy is it to follow and implement the approach?"
    }


def _generate_execution_sequence(approaches: list[str], problem: str) -> dict[str, Any]:
    """Generate execution sequence for comparative analysis."""

    return {
        "preparation": {
            "problem_standardization": f"Ensure all approaches address exactly this problem: {problem}",
            "criteria_establishment": "Define evaluation criteria before beginning",
            "documentation_setup": "Prepare to record process and results for each approach"
        },
        "execution_order": [
            {
                "approach": approach,
                "focus": f"Apply {approach} methodology systematically",
                "documentation": "Record reasoning process, insights, and solution",
                "evaluation": "Assess results against established criteria"
            } for approach in approaches
        ],
        "post_execution": {
            "comparative_analysis": "Compare results across all approaches",
            "pattern_identification": "Identify patterns in when each approach excels",
            "synthesis": "Develop insights about approach selection and combination"
        }
    }


def _generate_analysis_framework() -> dict[str, list[str]]:
    """Generate framework for analyzing comparative results."""

    return {
        "quantitative_analysis": [
            "Score each approach on defined criteria (1-10 scale)",
            "Calculate average scores and identify highest performers",
            "Analyze score patterns across different criteria"
        ],
        "qualitative_analysis": [
            "Identify unique strengths of each approach",
            "Analyze types of insights generated by each method",
            "Examine reasoning processes and their characteristics"
        ],
        "situational_analysis": [
            "Determine problem characteristics that favor each approach",
            "Identify when to use each method based on context",
            "Develop decision rules for approach selection"
        ],
        "integration_analysis": [
            "Explore how approaches could be combined effectively",
            "Identify complementary strengths across methods",
            "Design hybrid approaches for complex problems"
        ]
    }


def _generate_decision_matrix(approaches: list[str]) -> dict[str, dict[str, str]]:
    """Generate decision matrix for approach selection."""

    matrix = {}

    for approach in approaches:
        if approach == "chain_of_thought":
            matrix[approach] = {
                "best_for": "Straightforward problems with clear logical progression",
                "strengths": "Fast, clear, easy to follow and verify",
                "limitations": "Limited exploration of alternatives",
                "use_when": "Time is limited or problem is well-understood"
            }
        elif approach == "tree_of_thoughts":
            matrix[approach] = {
                "best_for": "Problems requiring exploration of multiple approaches",
                "strengths": "Good balance of exploration and focus",
                "limitations": "Limited interaction between branches",
                "use_when": "Multiple viable approaches exist"
            }
        elif approach == "graph_of_thoughts":
            matrix[approach] = {
                "best_for": "Complex problems requiring synthesis of multiple perspectives",
                "strengths": "Rich interactions, emergent insights, comprehensive",
                "limitations": "More complex to execute, requires more time",
                "use_when": "Problem is complex and quality matters more than speed"
            }
        elif approach == "iterative_refinement":
            matrix[approach] = {
                "best_for": "Problems requiring optimization and progressive improvement",
                "strengths": "Continuous improvement, builds on insights",
                "limitations": "May converge on local optima",
                "use_when": "Initial solution exists but needs improvement"
            }
        else:
            matrix[approach] = {
                "best_for": f"Problems suited to {approach} methodology",
                "strengths": "Approach-specific advantages",
                "limitations": "Approach-specific constraints",
                "use_when": f"Context favors {approach} approach"
            }

    return matrix


# Cleanup handlers
def cleanup():
    """Clean up resources on shutdown"""
    logger.info("Shutting down Graph of Thoughts MCP server")


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down gracefully")
    cleanup()
    sys.exit(0)


# Register cleanup handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)
atexit.register(cleanup)


def main():
    """Main entry point for the MCP server"""
    try:
        logger.info("Starting Graph of Thoughts MCP server")
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        logger.error(traceback.format_exc())
    finally:
        cleanup()


if __name__ == "__main__":
    main()

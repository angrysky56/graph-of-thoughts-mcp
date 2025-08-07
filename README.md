# Graph of Thoughts MCP Server 🧠

A sophisticated Model Context Protocol server that provides **Graph of Thoughts reasoning frameworks** for complex problem-solving through structured reasoning methodologies.

Based on the research from "Graph of Thoughts: Solving Elaborate Problems with Large Language Models" (Besta et al., AAAI 2024).

## 🌟 Features

- **Reasoning Framework Provider**: Provides structured guidance for Chain-of-Thought, Tree-of-Thoughts, Graph-of-Thoughts, and Iterative Refinement patterns
- **Meta-Cognitive Guidance**: Built-in quality assessment and improvement recommendations
- **Pattern Templates**: Ready-to-use templates for different reasoning approaches
- **Comparative Analysis**: Framework for comparing different reasoning methodologies
- **MCP Integration**: Seamless integration with Claude Desktop as a reasoning assistant
- **Zero Circular Dependencies**: Clean architecture that helps Claude reason better rather than attempting reasoning itself

## 🚀 Quick Start
Just add the adapted config to your client:

### Claude Desktop Configuration

Add the configuration from `example_mcp_config.json` to your Claude Desktop config:

```json
{
  "mcpServers": {
    "graph-of-thoughts": {
      "command": "uv",
      "args": [
        "--directory",
        "/your-path-to/graph-of-thoughts-mcp/graph_of_thoughts_mcp",
        "run",
        "server.py"
      ]
    }
  }
}
```

### Usage

Restart Claude Desktop and access the sophisticated reasoning tools:

## 🛠 Available MCP Tools

### `execute_reasoning_graph`
Execute sophisticated Graph of Thoughts reasoning on complex problems.

**Parameters:**
- `problem_description`: The complex problem to reason about
- `reasoning_pattern`: Pattern to use (`"chain_of_thought"`, `"tree_of_thoughts"`, `"graph_of_thoughts"`, `"iterative_refinement"`)
- `initial_data`: Optional initial context data
- `custom_operations`: For custom reasoning patterns

**Example Usage in Claude:**
```
Use the graph_of_thoughts tool to analyze this complex business scenario using the "graph_of_thoughts" pattern: "A tech startup needs to decide whether to pivot their product strategy given changing market conditions, limited runway, and competing priorities."
```

### `create_custom_reasoning_pattern`
Create and test custom reasoning patterns for specific problem types.

**Parameters:**
- `pattern_name`: Name for your custom pattern
- `problem_type`: Type of problem it's designed for
- `operations_sequence`: Sequence of operations (`["generate", "score", "aggregate", "select"]`)
- `description`: Optional description

### `analyze_reasoning_quality`
Analyze the quality and effectiveness of reasoning results.

**Parameters:**
- `reasoning_result`: JSON output from `execute_reasoning_graph`
- `evaluation_criteria`: Specific criteria to evaluate
- `benchmark_against`: Optional comparison baseline

### `compare_reasoning_approaches`
Compare different reasoning approaches on the same problem.

**Parameters:**
- `problem_description`: Problem to analyze with multiple approaches
- `approaches_to_compare`: List of reasoning patterns to compare
- `evaluation_metrics`: Metrics for comparison

## 📚 Reasoning Patterns

### Chain of Thought
- **Best for**: Straightforward problems with clear logical progression
- **Operations**: Generate → Score
- **Strengths**: Fast, clear reasoning path
- **Use Case**: Simple analytical problems

### Tree of Thoughts
- **Best for**: Problems requiring exploration of multiple approaches
- **Operations**: Generate(3) → Score → Select(2) → Generate(2) → Score → Select(1)
- **Strengths**: Explores alternatives while maintaining quality
- **Use Case**: Decision-making with multiple viable options

### Graph of Thoughts
- **Best for**: Complex problems requiring synthesis of multiple perspectives
- **Operations**: Generate(4) → Score → Select(3) → Aggregate → Generate(2) → Score → Select(1)
- **Strengths**: Rich interaction between ideas, comprehensive exploration
- **Use Case**: Complex strategic planning, research synthesis

### Iterative Refinement
- **Best for**: Problems requiring progressive optimization
- **Operations**: Generate → Score → Generate(improvements) → Score → Select → Refine
- **Strengths**: Continuous improvement, builds on insights
- **Use Case**: Creative writing, solution optimization

## 🎯 Advanced Usage Examples

### Complex Strategy Analysis
```
Execute graph reasoning on: "Develop a comprehensive AI governance framework for a multinational corporation considering regulatory compliance, ethical implications, competitive advantage, and stakeholder concerns" using the "graph_of_thoughts" pattern.
```

### Custom Pattern for Research
```
Create a custom reasoning pattern called "research_synthesis" for "academic_research" problems using operations: ["generate", "generate", "score", "select", "aggregate", "generate", "score"].
```

### Quality Assessment
```
Analyze the quality of the previous reasoning result focusing on logical consistency, evidence integration, and practical applicability.
```

## 🏗 Architecture

The system implements a sophisticated architecture inspired by the ETH Zurich Graph of Thoughts research:

- **Operations System**: Core reasoning building blocks (Generate, Score, Aggregate, Select)
- **Controller Pattern**: Manages execution flow and coordinates operations
- **Graph Builder**: Fluent interface for constructing reasoning workflows
- **MCP Integration**: Seamless Claude Desktop integration through sampling
- **Quality Analysis**: Comprehensive evaluation and improvement frameworks

## 🔬 Research Foundation

This implementation is based on "Graph of Thoughts: Solving Elaborate Problems with Large Language Models" (Besta et al., AAAI 2024), which demonstrated significant improvements over Chain-of-Thought and Tree-of-Thoughts approaches on complex reasoning tasks.

The key insight is that reasoning benefits from both structured exploration and rich interconnections between ideas, enabling more sophisticated problem-solving capabilities than traditional linear or tree-based approaches.

## 📖 Resources

- `got://methodology-guide`: Comprehensive methodology guide (accessible as MCP resource)
- **Original Paper**: [Graph of Thoughts: Solving Elaborate Problems with Large Language Models](https://arxiv.org/pdf/2308.09687.pdf)
- **ETH Zurich Repository**: [Official Implementation](https://github.com/spcl/graph-of-thoughts)

## 🛡 Error Handling

The system follows transparent error handling principles:
- All errors are properly displayed with clear, actionable messages
- No fallback systems that mask underlying issues
- Comprehensive logging to stderr for MCP compatibility
- Graceful degradation with informative error responses

## 🚨 Performance Notes

- Optimized for sophisticated reasoning over speed
- Graph-of-Thoughts patterns are computationally intensive but provide superior results for complex problems
- Use Chain-of-Thought for simple problems requiring fast responses
- Quality scoring uses LLM evaluation for accuracy but can be customized with scoring functions

### Manual Installation
```bash
# Navigate to the project directory
cd /graph-of-thoughts-mcp

# Create and activate virtual environment
uv venv --python 3.12 --seed
source .venv/bin/activate

# Install dependencies
uv sync
```

## 🤝 Contributing

This project follows strict architectural principles:
- Zero code duplication - single implementation per feature
- No fallback systems - transparent error handling
- Component architecture with reusable, modular design
- Comprehensive documentation and type hints

## 📄 License

MIT License - See LICENSE file for details.

---

**Ready to revolutionize your reasoning capabilities?** Add this MCP server to Claude Desktop and experience the power of Graph of Thoughts methodology for complex problem-solving! 🧠✨

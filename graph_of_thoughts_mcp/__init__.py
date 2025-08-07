"""
Graph of Thoughts MCP Server

A sophisticated Model Context Protocol server implementing Graph of Thoughts
reasoning methodology for complex problem-solving through interconnected
reasoning networks.

Based on the research from "Graph of Thoughts: Solving Elaborate Problems
with Large Language Models" (Besta et al., AAAI 2024).
"""

from .__about__ import __version__
from .server import main, mcp

__all__ = ["main", "mcp", "__version__"]

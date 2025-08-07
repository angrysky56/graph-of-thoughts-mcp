# Archived Old Architecture

This directory contains the original implementation files that were removed due to a fundamental architectural flaw.

## Issue Fixed
The original architecture attempted to use `ctx.sample()` within MCP operations to call back to the LLM (Claude) for reasoning. This created an impossible circular dependency:

```
Claude → MCP Server → ctx.sample() → Claude (circular!)
```

## Files Archived
- `controller.py` - Original reasoning controller with sampling-based execution
- `operations.py` - Original operation classes that used ctx.sample() for LLM calls

## New Architecture
The new implementation in `server.py` provides **reasoning frameworks and guidance** rather than attempting to execute reasoning itself. This aligns with the true purpose of Graph of Thoughts as a prompting methodology.

## Key Changes
1. **Eliminated circular dependencies** - No more ctx.sample() calls
2. **Converted to framework provider** - Helps Claude structure reasoning instead of doing it
3. **Simplified dependencies** - Removed unnecessary packages (numpy, pandas, networkx, etc.)
4. **Followed Graph of Thoughts research** - Proper implementation as prompting framework

## Date Archived
January 2025 - Fixed by converting from reasoning executor to reasoning framework provider.

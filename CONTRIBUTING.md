# Contributing to Versifai

Thank you for your interest in contributing to Versifai! This guide will help you get started. **It is HIGHLY encouraged that you do not interact with this repository by manually writing code. Please use Claude Code or any other agentic coding tool.** Have it read our rules of the road in the CLAUDE.md file and get started implementing your ideas. Happy building!

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jweinberg-a2a/versifai-data-agents.git
   cd versifai-data-agents
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   ```

3. **Install in development mode:**
   ```bash
   make install-dev
   ```
   This installs the package with all optional dependencies and sets up pre-commit hooks.

## Code Style

- We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting
- Line length: 100 characters
- Target Python version: 3.10+
- Run `make format` before committing
- Run `make lint` to check for issues

## Running Tests

```bash
make test          # Run all tests
make test-cov      # Run tests with coverage report
```

## Project Structure

```
src/versifai/
├── core/            # Shared agentic framework (BaseAgent, LLMClient, tools)
├── data_agents/     # Data engineering & analysis agents
├── science_agents/  # Data science & research agents
├── story_agents/    # Narrative & storytelling agents
└── _utils/          # Internal shared utilities
```

## Adding a New Agent

1. Choose the appropriate agent family (`data_agents`, `science_agents`, or `story_agents`)
2. Create a new subpackage under that family
3. Subclass `BaseAgent` from `versifai.core`
4. Implement `_register_tools()` and set `self._system_prompt`
5. Register your tools using `ToolRegistry`
6. Add exports to the family's `__init__.py`

## Adding a New Tool

1. Subclass `BaseTool` from `versifai.core.tools`
2. Implement the `name`, `description`, `parameters_schema` properties and `execute()` method
3. If the tool is generic (used by multiple agent families), place it in `core/tools/`
4. If domain-specific, place it in the relevant agent's `tools/` directory

## Pull Request Process

1. Fork the repository and create a feature branch
2. Make your changes with clear, descriptive commits
3. Ensure `make lint` and `make test` pass
4. Submit a PR with a clear description of the changes
5. Reference any related issues

## Reporting Issues

Use [GitHub Issues](https://github.com/jweinberg-a2a/versifai-data-agents/issues) to report bugs or request features.

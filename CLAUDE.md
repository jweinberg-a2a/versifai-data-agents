# CLAUDE.md — Versifai Contributor Guide

This file is the authoritative reference for anyone contributing to versifai using Claude Code or any AI-assisted development tool. Read it before writing a single line of code.

---

## What is Versifai?

Versifai is an autonomous, LLM-powered framework with three specialized AI agents that discover, engineer, validate, and analyze data on Databricks Unity Catalog. It is a **library of reusable data agents and associated tooling** — not a platform, not infrastructure, not plumbing.

The three agents collaborate across the data lifecycle:

| Agent | Role | Key Tools |
|-------|------|-----------|
| **Data Engineer** | Ingests raw files, profiles them, designs schemas, transforms and loads into Delta tables | `explore_volume`, `profile_data`, `design_schema`, `transform_and_load`, `write_to_catalog` |
| **Data Scientist** | Runs statistical analysis, fits models, validates results, saves findings | `statistical_analysis`, `fit_model`, `check_confounders`, `validate_statistics`, `save_finding` |
| **StoryTeller** | Reads findings, evaluates evidence strength, constructs narrative reports with citations | `read_findings`, `evaluate_evidence`, `write_narrative`, `cite_source` |

Full tool reference: [docs/tool-inventory.md](docs/tool-inventory.md)

---

## Philosophy & Design Principles

These are non-negotiable. Every contribution must respect them.

### 1. Reproducibility First

> **Everything is reproducible if we REMOVE the AI agent from the process.**

The agent is there to create artifacts, notes, progress tracking, and documentation **on top of** actually outputting results. It is never acceptable to simply provide an answer. The code, assumptions, queries, data transformations, and statistical reasoning must all be recorded as artifacts that a human can follow without the AI.

- Every SQL query the agent runs is logged to theme notes
- Every visualization includes the SQL/data and render code in metadata
- Every finding includes evidence strings, p-values, effect sizes
- Every schema design produces CREATE TABLE SQL that can be run manually

### 2. Artifacts Alongside Results

Agents produce structured, auditable output — not just answers:

- **Notes files** — per-theme markdown files tracking reasoning and decisions (`save_note`)
- **Findings JSON** — structured research findings with evidence and significance (`save_finding`)
- **Schema definitions** — CREATE TABLE SQL for every table design (`design_schema`)
- **Chart metadata** — SQL queries, input data, and render code logged per chart (`create_visualization`)
- **Citation registry** — formatted references for all sources used (`cite_source`)
- **Progress tracking** — each agent phase records what was done and why

### 3. Don't Reinvent the Wheel

Versifai is built **on top of Databricks** as the base data platform. We do not build:

- Storage engines — we use Delta Lake
- SQL engines — we use Spark SQL via Unity Catalog
- Authentication — we use Databricks SDK and service principals
- Model tracking — we use MLflow (when configured)
- Cluster management — we use Databricks clusters

We also leverage established open-source libraries rather than reimplementing:

- `litellm` for multi-provider LLM access (not raw API calls)
- `pandas`, `scipy`, `scikit-learn`, `statsmodels` for data science
- `matplotlib`, `seaborn`, `plotly` for visualization
- `beautifulsoup4` for HTML extraction

If a library or platform already does it well, use it. Our value is the **agent orchestration, tool design, and domain logic** — not infrastructure.

### 4. Tool-Based Agent Architecture

All agent work happens through **tools**. Tools are the unit of capability:

- Each tool has a clear name, description, parameter schema, and returns a `ToolResult`
- Tools are composable — agents chain them via the ReAct loop
- Tools are testable in isolation — no LLM required for unit tests
- Tools are auditable — every call is logged to conversation memory

Never put business logic directly in agent code or system prompts. If an agent needs a new capability, build a tool.

### 5. Evidence-Based Claims

Statistical claims must be backed by actual evidence. The evidence tier system enforces this:

| Tier | Criteria | Use For |
|------|----------|---------|
| DEFINITIVE | p < 0.001, large effect size | Primary conclusions |
| STRONG | p < 0.01, medium+ effect size | Leading findings |
| SUGGESTIVE | p < 0.05 | Supporting evidence |
| CONTEXTUAL | Descriptive, no hypothesis test | Background context |
| WEAK | p >= 0.05, negligible effect | Limitations, caveats |

Narrative text must match the statistical evidence. If p=0.73, the finding is WEAK — regardless of what the text says.

---

## Project Structure

```
src/versifai/
├── core/                              # Shared agentic framework
│   ├── agent.py                       # BaseAgent — ReAct loop, phase execution
│   ├── llm.py                         # LLMClient — multi-provider via LiteLLM
│   ├── memory.py                      # AgentMemory — conversation history + compression
│   ├── display.py                     # AgentDisplay — formatted output
│   └── tools/
│       ├── base.py                    # BaseTool ABC + ToolResult dataclass
│       ├── registry.py                # ToolRegistry — register, execute, to_claude_tools()
│       ├── catalog_writer.py          # CatalogWriterTool, ExecuteSQLTool, SilverOnlyExecuteSQLTool, ListCatalogTablesTool
│       ├── column_renamer.py          # RenameColumnsTool
│       ├── dynamic_tool_builder.py    # DynamicToolBuilderTool (runtime tool creation)
│       ├── save_note.py               # SaveNoteTool (per-theme notes)
│       ├── view_chart.py              # ViewChartTool
│       ├── visualization.py           # CreateVisualizationTool (charts, maps, tables)
│       ├── web_search.py              # WebSearchTool
│       └── web_scraper.py             # WebScraperTool
│
├── data_agents/
│   └── engineer/
│       ├── agent.py                   # DataEngineerAgent
│       ├── config.py                  # ProjectConfig dataclass
│       └── tools/                     # 8 tools: volume, file, profiler, schema, transformer, docs
│
├── science_agents/
│   └── scientist/
│       ├── agent.py                   # DataScientistAgent
│       ├── config.py                  # ResearchConfig dataclass
│       └── tools/                     # 8 tools: stats, models, confounders, validation, findings, literature, MLflow
│
└── story_agents/
    └── storyteller/
        ├── agent.py                   # StoryTellerAgent
        ├── config.py                  # StorytellerConfig dataclass
        └── tools/                     # 6 tools: findings, charts, tables, narrative, citations, evidence

tests/
├── conftest.py                        # Shared fixtures
├── test_core/                         # Core framework tests
├── test_data_agents/                  # Data agent tool tests
├── test_science_agents/               # Science agent tool tests (behavioral focus)
├── test_story_agents/                 # Story agent tool tests (behavioral focus)
└── test_integration/                  # Real LLM integration tests (@pytest.mark.integration)

docs/
├── index.md                           # Landing page (mkdocs)
├── getting-started.md                 # Installation, quick start, config
├── architecture.md                    # System design and data flow
├── tool-inventory.md                  # Complete tool reference (tables)
├── data-engineer.md                   # Data Engineer agent guide
├── data-scientist.md                  # Data Scientist agent guide
├── storyteller.md                     # StoryTeller agent guide
├── custom-agents.md                   # Building custom agents
├── contributing.md                    # Contributing guide (includes CONTRIBUTING.md)
├── changelog.md                       # Changelog (includes CHANGELOG.md)
├── assets/                            # Logo, favicon
└── api/                               # Auto-generated API reference (mkdocstrings)
    ├── core.md
    ├── data-agents.md
    ├── science-agents.md
    └── story-agents.md

examples/
└── silly_weather/                     # Reference implementation: "Do Ducks Predict Rain?"
    ├── engineer_config.py             # ProjectConfig for Data Engineer
    ├── storyteller_config.py          # StorytellerConfig for StoryTeller
    ├── notebooks/                     # Runnable Databricks notebooks
    │   ├── run_engineer.py            # Data ingestion pipeline
    │   ├── run_scientist.py           # Research analysis pipeline
    │   └── run_storyteller.py         # Narrative report pipeline
    └── research_configs/
        └── duck_rain_prediction.py    # ResearchConfig for duck-weather analysis
```

---

## Coding Standards

### Python Version & Style

- **Python 3.10+** — use `X | Y` union syntax, not `Union[X, Y]`
- **Line length: 100** characters
- **Linter/formatter: ruff** — configuration in `pyproject.toml`
- **Import sorting: isort** via ruff with `known-first-party = ["versifai"]`
- Run `make format` and `make lint` before every commit

### Naming Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Files | `snake_case.py` | `statistical_analysis.py` |
| Classes | `PascalCase` | `StatisticalAnalysisTool` |
| Tool names | `snake_case` string | `"statistical_analysis"` |
| Functions/methods | `snake_case` | `_execute()`, `_normalize_response()` |
| Constants | `UPPER_SNAKE_CASE` | `DIRECT_WRITE_MAX_ROWS` |
| Private members | Leading underscore | `_model`, `_cfg` |
| Config classes | `PascalCase` + `Config` suffix | `ProjectConfig`, `ResearchConfig` |
| Table names | `silver_` prefix for processed tables | `silver_county_demographics` |

### Import Conventions

```python
from __future__ import annotations          # Always first — enables PEP 604 syntax

import json                                  # stdlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd                          # third-party
from scipy import stats

from versifai.core.tools.base import BaseTool, ToolResult  # first-party

if TYPE_CHECKING:                            # type-only imports (avoid circular deps)
    from versifai.data_agents.engineer.config import ProjectConfig
```

### Data Science Naming Exception

Data science code conventionally uses uppercase variable names (`X`, `X_train`, `y_test`). These files have ruff N803/N806 exceptions in `pyproject.toml`:

```
"src/versifai/science_agents/scientist/tools/model_fitting.py" = ["N803", "N806"]
"src/versifai/science_agents/scientist/tools/statistical_analysis.py" = ["N806"]
```

---

## Tool Development Pattern

Every tool follows the same pattern. Here is the complete template:

```python
"""
Tool: tool_name

Brief description of what this tool does and when agents should use it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from versifai.core.tools.base import BaseTool, ToolResult

if TYPE_CHECKING:
    from versifai.data_agents.engineer.config import ProjectConfig


class MyNewTool(BaseTool):
    """One-line description of the tool."""

    def __init__(self, cfg: ProjectConfig) -> None:
        super().__init__()
        self._cfg = cfg

    @property
    def name(self) -> str:
        return "my_new_tool"

    @property
    def description(self) -> str:
        return (
            "Description visible to the LLM agent. Be specific about when "
            "to use this tool and what it returns."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "required_param": {
                    "type": "string",
                    "description": "What this parameter controls.",
                },
                "optional_param": {
                    "type": "integer",
                    "description": "Optional with default. Default 100.",
                    "default": 100,
                },
            },
            "required": ["required_param"],
        }

    def _execute(self, required_param: str, optional_param: int = 100, **kwargs) -> ToolResult:
        """Implement the tool logic. Return ToolResult always."""
        try:
            # Do the work
            result_data = {"key": "value", "count": optional_param}

            return ToolResult(
                success=True,
                data=result_data,
                summary=f"Processed {optional_param} items for {required_param}",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to process: {e}",
            )
```

### ToolResult Fields

| Field | Type | Purpose |
|-------|------|---------|
| `success` | `bool` | Whether the operation completed |
| `data` | `Any` | Result payload (dict, list, string) |
| `error` | `str` | Error message if `success=False` |
| `summary` | `str` | Human-readable summary for agent |
| `image_path` | `str` | Path to PNG for inline display |

### Registration

Tools are registered in the agent's `__init__` or setup method:

```python
from versifai.core.tools.registry import ToolRegistry

registry = ToolRegistry()
registry.register(MyNewTool(cfg=self._cfg))
```

---

## Agent Architecture

### ReAct Loop

The core execution pattern in `BaseAgent._run_phase()`:

```
1. Agent receives prompt + system prompt
2. LLM reasons about what to do → produces text + tool_use blocks
3. ToolRegistry executes each tool call → returns ToolResult
4. Results are stored in AgentMemory as tool_result messages
5. LLM sees results, reasons again → may call more tools or produce final answer
6. Loop continues until: stop_reason="end_turn" OR max_turns reached
```

### Key Classes

| Class | File | Role |
|-------|------|------|
| `BaseAgent` | `core/agent.py` | ReAct loop, phase execution, error tracking |
| `LLMClient` | `core/llm.py` | Multi-provider LLM via LiteLLM, prompt caching, retries |
| `AgentMemory` | `core/memory.py` | Message history, compression at 30 messages, tool result trimming |
| `AgentDisplay` | `core/display.py` | Formatted console output |
| `ToolRegistry` | `core/tools/registry.py` | Tool registration, execution, Claude API schema generation |
| `BaseTool` | `core/tools/base.py` | Abstract base class for all tools |
| `ToolResult` | `core/tools/base.py` | Dataclass for tool return values |

### Config Pattern

All configs are **dataclasses** composed from smaller building blocks:

```python
@dataclass
class ProjectConfig:
    name: str
    catalog: str
    schema: str
    volume_path: str
    documentation_urls: dict[str, list[str]] = field(default_factory=dict)
    # ... more fields
```

Configs flow from orchestrator → agent → tools. Each tool receives only the config it needs.

### SQL Write Protection

- `ExecuteSQLTool` — full read/write access (Data Engineer only)
- `SilverOnlyExecuteSQLTool` — DDL/DML restricted to `silver_*` tables (Data Scientist, StoryTeller)
- SELECT queries are unrestricted for all agents

---

## Contribution Requirements

### Every New Tool Must Include

1. **Unit tests** in the appropriate `tests/test_<agent_type>/` directory
   - Test instantiation and property values
   - Test `_execute()` with valid inputs
   - Test error handling (missing params, bad data, edge cases)
   - Test Databricks-specific behavior with mocks where needed

2. **Behavioral tests** (for tools that make statistical or factual claims)
   - Feed intentionally wrong data — verify the tool catches the error
   - Feed borderline data — verify correct classification
   - Test that narrative text doesn't override statistical evidence

3. **Integration tests** (if the tool interacts with the LLM agent)
   - Real LLM calls via `ReviewerAgent` or similar test harness
   - Marked with `@pytest.mark.integration`
   - Skipped automatically without API key

4. **Documentation update** to `docs/tool-inventory.md`
   - Add the tool to the appropriate agent section
   - Include parameter table, return table, and special notes
   - Update the cross-reference matrix if the tool is shared

5. **Example usage** in a notebook or the existing example project

### Every New Agent Must Include

1. Agent class extending `BaseAgent` with system prompt and tool registration
2. Config dataclass with all required fields
3. Full test suite (unit + behavioral + integration)
4. Documentation in `docs/architecture.md`
5. Example notebook demonstrating the agent in action

### PR Checklist

Before submitting a pull request, verify:

- [ ] `ruff format --check src/ tests/` passes (formatting)
- [ ] `ruff check src/ tests/` passes (linting)
- [ ] `mypy src/versifai --ignore-missing-imports` passes (type checking)
- [ ] `make test` passes (all unit + behavioral tests)
- [ ] `pytest tests/test_integration/ -v -m integration` passes (if you have API key)
- [ ] New tools have unit tests with edge cases
- [ ] Behavioral tests feed bad data and verify correct detection
- [ ] `docs/tool-inventory.md` updated for any new/changed tools
- [ ] No hardcoded API keys, tokens, or secrets
- [ ] `.env` file is NOT committed (check `.gitignore`)
- [ ] Imports use `TYPE_CHECKING` for circular dependency prevention
- [ ] Tool `_execute()` always returns `ToolResult`, never raises

---

## Testing Standards

### Test Structure

```
tests/
├── conftest.py                  # Shared fixtures (mock configs, sample data)
├── test_core/
│   ├── test_imports.py          # Smoke tests — verify all imports work
│   ├── test_base_tool.py        # BaseTool, ToolResult, ToolRegistry
│   └── ...
├── test_data_agents/            # Data engineer tool tests
├── test_science_agents/         # Science tool tests (behavioral focus)
├── test_story_agents/           # Story tool tests (behavioral focus)
└── test_integration/            # Real LLM integration tests
    ├── conftest.py              # CMS_CITATION, requires_llm, test fixtures
    └── test_agent_behavioral.py # Full ReAct loop with actual LLM calls
```

### Running Tests

```bash
# All unit + behavioral tests (no API key needed)
make test

# Integration tests only (requires ANTHROPIC_API_KEY or OPENAI_API_KEY)
pytest tests/test_integration/ -v -m integration

# Skip integration tests
pytest tests/ -v -m "not integration"

# Override integration test model
VERSIFAI_TEST_MODEL=claude-sonnet-4-6 pytest tests/test_integration/ -v

# With coverage
make test-cov
```

### Test Markers

| Marker | Purpose |
|--------|---------|
| `integration` | Tests that make real LLM API calls. Skipped without API key. |

### Behavioral Test Pattern

Behavioral tests verify that tools produce **correct conclusions**, not just that code runs:

```python
def test_no_false_significance(self):
    """Groups with tiny mean difference on noisy data should NOT be significant."""
    data = generate_noisy_data(mean_diff=0.001, noise=10.0)
    result = tool._execute(analysis_type="hypothesis_test", data=data, ...)
    assert result.data["p_value"] > 0.05, "False positive: noise detected as significant"

def test_simpsons_paradox_detected(self):
    """Aggregate positive trend with all-negative subgroups must be flagged."""
    data = generate_simpsons_paradox_data()
    result = tool._execute(data=data, ...)
    assert result.data["paradox_detected"] is True
```

---

## Databricks Environment Notes

### Unity Catalog

All tables live in a three-level namespace: `catalog.schema.table`

```sql
-- Example
SELECT * FROM my_catalog.my_schema.silver_county_demographics LIMIT 10;
```

Table naming convention:
- `bronze_*` — raw ingested data (engineer only)
- `silver_*` — cleaned, transformed, validated data
- `gold_*` — analysis-ready aggregations

### Volumes (FUSE Mount)

Databricks Volumes are accessed via FUSE mounts at `/Volumes/catalog/schema/volume/`.

**Critical quirk: no file append on FUSE.** The `SaveNoteTool` uses a read-then-write pattern:

```python
# CORRECT — read existing content, append, write back
existing = path.read_text() if path.exists() else ""
path.write_text(existing + new_content)

# WRONG — append mode fails on FUSE
with open(path, "a") as f:  # This will fail!
    f.write(new_content)
```

### Spark vs SDK Fallback

Tools that interact with Databricks follow a two-tier execution pattern:

1. **Try Spark first** — faster, native in Databricks notebooks
2. **Fall back to Databricks SDK** — works outside notebooks, async with polling

```python
try:
    result = spark.sql(query)  # Fast path
except Exception:
    result = databricks_sdk_execute(query)  # Fallback
```

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | LLM API access (Anthropic Claude) |
| `OPENAI_API_KEY` | LLM API access (OpenAI, alternative) |
| `DATABRICKS_HOST` | Databricks workspace URL |
| `DATABRICKS_TOKEN` | Databricks personal access token |
| `DATABRICKS_SP_CLIENT_ID` | Service principal client ID |
| `DATABRICKS_SP_CLIENT_SECRET_DEV` | Service principal secret (dev) |
| `DATABRICKS_CLUSTER_ID_DEV` | Compute cluster ID (dev) |

Store in `.env` (git-ignored). Never hardcode secrets.

---

## CI & Code Quality

### CI Pipeline (`.github/workflows/ci.yml`)

Every push and PR to `main` runs four checks — **all must pass**:

| Job | What it runs | Command |
|-----|-------------|---------|
| **Lint** | ruff check + format | `ruff check src/ tests/` and `ruff format --check src/ tests/` |
| **Test** | Unit + behavioral tests (3.10, 3.11, 3.12) | `pytest tests/ -v --tb=short -m "not integration"` |
| **Type Check** | mypy static analysis | `mypy src/versifai --ignore-missing-imports` |
| **Build** | Package build + import verification | `python -m build` + smoke-test imports |

Integration tests (`tests/test_integration/`) are **excluded from CI** — they require LLM API keys (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) which are not available in the GitHub environment. Run them locally when you have keys set.

### Ruff (Linting & Formatting)

Configuration lives in `pyproject.toml` under `[tool.ruff]`:

```bash
ruff format src/ tests/     # Auto-format all files
ruff check --fix src/ tests/ # Lint + auto-fix what's possible
ruff check src/ tests/       # Lint check only (CI runs this)
ruff format --check src/ tests/ # Format check only (CI runs this)
```

- **Line length:** 100 characters
- **Target:** Python 3.10
- **Import sorting:** isort via ruff with `known-first-party = ["versifai"]`
- **Always run `make format` before committing** to avoid CI lint failures

### Mypy (Type Checking)

Configuration lives in `pyproject.toml` under `[tool.mypy]`:

```bash
mypy src/versifai --ignore-missing-imports  # CI runs this exact command
mypy src/versifai/                          # Also works (trailing slash)
```

Key mypy settings:
- `python_version = "3.10"`
- `warn_return_any = true`, `warn_unused_configs = true`
- `disallow_untyped_defs = false` (not enforced yet)

**Suppressing mypy errors:**

When adding `# type: ignore` comments, **always include the specific error code(s)**:

```python
# CORRECT — specific error codes
result = some_call()  # type: ignore[no-any-return]
value = obj.attr      # type: ignore[attr-defined, union-attr]

# WRONG — bare ignore (hides real errors)
result = some_call()  # type: ignore
```

If a line has **multiple** mypy errors, list all codes in a single comment:
```python
# If mypy reports both [no-any-return] and [operator] on the same line:
x = foo() + bar()  # type: ignore[no-any-return, operator]
```

**Per-module ignores** for untyped third-party libraries are in `pyproject.toml`:
```toml
[[tool.mypy.overrides]]
module = ["litellm.*", "databricks.*", "requests.*", ...]
ignore_missing_imports = true
```

### Documentation Site

Docs are built with **MkDocs Material** and deployed to GitHub Pages via `.github/workflows/docs.yml`:

```bash
make docs-serve     # Local preview with live reload (localhost:8000)
make docs-build     # Build site in strict mode (catches broken links)
make docs-deploy    # Deploy to GitHub Pages (gh-pages branch)
```

- API reference is auto-generated from docstrings via **mkdocstrings**
- Docs deploy triggers on push to `main` when `docs/`, `mkdocs.yml`, or `src/` change
- Configuration: `mkdocs.yml`
- Dependencies: `pip install -e ".[docs]"` (separate from `dev` extras)

---

## Quick Reference

### Common Commands

```bash
make install-dev    # Install with all dev dependencies + pre-commit hooks
make lint           # ruff check + mypy
make format         # Auto-fix formatting (ruff format + ruff check --fix)
make test           # Run all unit tests
make test-cov       # Run tests with coverage report
make docs-serve     # Preview docs locally (localhost:8000)
make docs-build     # Build docs (strict mode)
```

### Pre-Commit Checklist

```bash
make format                    # Auto-fix formatting
ruff check src/ tests/         # Verify lint passes
mypy src/versifai --ignore-missing-imports  # Verify types pass
pytest tests/ -v -m "not integration"      # Verify tests pass
```

### Key Files to Read First

1. `src/versifai/core/tools/base.py` — BaseTool and ToolResult (the foundation)
2. `src/versifai/core/agent.py` — BaseAgent ReAct loop
3. `src/versifai/core/llm.py` — LLMClient multi-provider interface
4. `docs/tool-inventory.md` — Complete tool reference
5. `docs/architecture.md` — System design and data flow
6. `examples/silly_weather/` — Reference implementation

### Links

- [Documentation Site](https://jweinberg-a2a.github.io/versifai-data-agents/) — full docs on GitHub Pages
- [Tool Inventory](docs/tool-inventory.md) — every tool, every parameter, every return value
- [Architecture](docs/architecture.md) — system design and data flow
- [Contributing](CONTRIBUTING.md) — development setup and PR process
- [Changelog](CHANGELOG.md) — release history

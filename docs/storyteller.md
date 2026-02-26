# StoryTeller Agent

The `StoryTellerAgent` transforms research findings into evidence-grounded narrative reports. It reads findings, evaluates evidence quality, cites sources, and writes structured markdown sections that assemble into a complete report.

## How It Works

The storyteller operates in **section-based writing**. Each section of the narrative is defined in the config and written independently:

1. **Read findings** -Loads all findings from the scientist's output
2. **Evaluate evidence** -Classifies each finding by evidence tier
3. **Curate** -Selects the strongest, most relevant findings for each section
4. **Write** -Produces markdown with inline citations and visual references
5. **Assemble** -Combines all sections into a final narrative report with table of contents

## Available Tools

| Tool | Description |
|------|-------------|
| `read_findings` | Load and filter research findings (by theme, significance, keyword) |
| `read_table` | Read exported data tables from the scientist |
| `view_chart` | View charts produced by the scientist |
| `evaluate_evidence` | Classify findings by evidence tier |
| `cite_source` | Manage bibliography and inline citations |
| `write_narrative` | Write, read, update, and assemble narrative sections |
| `save_note` | Save editorial notes |
| `ask_human` | Pause and ask the operator a question |

See the [Tool Reference](tool-inventory.md) for detailed parameter and return value tables.

## Run Outputs

```
narrative/{config_name}/runs/{run_id}/
├── section_01_introduction.md
├── section_02_methodology.md
├── section_03_findings.md
├── ...
├── narrative_report.md    # Assembled full report
├── bibliography.json      # All cited sources
└── run_metadata.json      # Run info, timing, state
```

## Usage

```python
from versifai.story_agents import StoryTellerAgent, StorytellerConfig

cfg = StorytellerConfig(
    name="Churn Analysis Report",
    thesis="Customer churn is driven primarily by...",
    research_results_path="/tmp/results/churn",
    narrative_output_path="/tmp/narrative/churn",
    narrative_sections=[...],  # Define report sections
    # Domain-specific editorial guidance (optional)
    domain_writing_rules=(
        "Frame churn as a business risk, not a statistical curiosity. "
        "Always translate effect sizes into revenue impact."
    ),
    citation_source_guidance=(
        "SaaS industry reports (Bessemer, OpenView), academic churn literature, "
        "and company internal benchmarks."
    ),
)

agent = StoryTellerAgent(cfg=cfg, dbutils=dbutils)

# Full run
result = agent.run()

# Re-run specific sections
result = agent.run_sections(sections=[1, 2])
```

### Domain Guidance Fields

The storyteller prompts are **domain-agnostic**. Use these config fields to inject editorial rules:

| Field | Purpose | Default |
|-------|---------|---------|
| `domain_writing_rules` | Domain-specific editorial guidance (injected into system prompt) | `""` (generic rules) |
| `citation_source_guidance` | Preferred citation sources for the domain | `""` (generic academic sources) |

When these fields are empty, the agent uses generic editorial rules about evidence-based writing.

## Editorial Review

The storyteller has a dedicated editor mode for human-guided revisions:

```python
# Guided review with specific instructions
agent.run_editor(
    instructions="Simplify the methodology section for a policymaker audience."
)

# Open-ended review -the agent asks what to improve
agent.run_editor()
```

## Inter-Agent Dependencies

The storyteller reads outputs from a prior scientist run. Dependencies are declared in the config using `AgentDependency`:

```python
from versifai.core.run_manager import AgentDependency

cfg = StorytellerConfig(
    ...,
    dependencies=[
        AgentDependency(
            agent_type="scientist",
            config_name="geographic_disparity",
            base_path="/tmp/results/geographic_disparity",
        ),
    ],
)
```

The dependency resolver automatically finds the latest scientist run, or you can pin a specific `run_id`.

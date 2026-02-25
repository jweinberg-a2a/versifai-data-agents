---
hide:
  - navigation
---

<p align="center">
  <img src="assets/logo.png" alt="Versifai" width="400">
</p>

# Versifai

**Agentic AI framework for autonomous data engineering, science, and storytelling.**

Versifai provides specialized AI agents that automate the complete data lifecycle — from raw file discovery and schema design, through statistical analysis and modeling, to compelling narrative reports. Each agent operates autonomously using a **ReAct (Reason-Act-Observe) loop**, with human-in-the-loop oversight at every stage.

Built on [LiteLLM](https://docs.litellm.ai/) for multi-provider LLM support (Anthropic, OpenAI, Azure, and 100+ more).

## Agent Families

| Family | Agents | What It Does |
|--------|--------|--------------|
| **Data Agents** | `DataEngineerAgent`, `DataAnalystAgent` | Discover raw files, profile data, design schemas, transform and load into structured tables. |
| **Science Agents** | `DataScientistAgent` | Autonomous research — builds analytical datasets, runs hypothesis tests, fits models, produces charts and findings. |
| **Story Agents** | `StoryTellerAgent` | Transforms research findings into evidence-grounded narrative reports with citations and visual references. |

## Key Features

- **Autonomous agent loop** — ReAct-based agents that reason, act, and observe iteratively
- **Multi-provider LLM** — Swap between Claude, GPT-4, Azure, Gemini, or any LiteLLM-supported provider
- **Modular tool system** — Plug-and-play tools with a shared registry
- **Smart resume** — Agents persist state and resume from where they left off
- **Run isolation** — Each run gets its own directory with metadata and artifacts
- **Human-in-the-loop** — Built-in `ask_human` tool lets agents pause and request guidance
- **Databricks native** — First-class support for Unity Catalog, Delta tables, and Volumes

[Get Started](getting-started.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/jweinberg-a2a/versifai-data-agents){ .md-button }

# Building Custom Agents

Versifai's modular architecture makes it straightforward to add custom tools and agents.

## Create a Custom Tool

Subclass `BaseTool` and implement `name`, `description`, `parameters_schema`, and `execute()`:

```python
from versifai.core import BaseTool, ToolResult

class FetchWeatherTool(BaseTool):
    name = "fetch_weather"
    description = "Fetch current weather for a city"
    parameters_schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
        },
        "required": ["city"],
    }

    def execute(self, city: str) -> ToolResult:
        # Your implementation here
        data = call_weather_api(city)
        return ToolResult(success=True, data=data)
```

Every tool returns a `ToolResult`:

- `success=True` with `data` — the tool worked, data is returned to the agent
- `success=False` with `error` — the tool failed, error message is returned to the agent

## Create a Custom Agent

Subclass `BaseAgent` and wire up your tools:

```python
from versifai.core import (
    BaseAgent, LLMClient, AgentMemory, AgentDisplay, ToolRegistry,
)

class WeatherAgent(BaseAgent):
    def __init__(self):
        registry = ToolRegistry()
        registry.register(FetchWeatherTool())

        super().__init__(
            display=AgentDisplay(),
            memory=AgentMemory(),
            llm=LLMClient(model="gpt-4o"),
            registry=registry,
        )
        self._system_prompt = "You are a helpful weather assistant."

    def ask(self, question: str) -> str:
        return self._run_phase(prompt=question, max_turns=10)

# Use it
agent = WeatherAgent()
answer = agent.ask("What's the weather in San Francisco?")
```

## Where to Put Your Code

| What you're adding | Where it goes |
|---|---|
| A tool used by multiple agent families | `src/versifai/core/tools/` |
| A tool specific to one agent | `src/versifai/<family>/<agent>/tools/` |
| A new agent in an existing family | `src/versifai/<family>/<new_agent>/` |
| A new agent family | `src/versifai/<new_family>/` |
| Shared config or data models | `src/versifai/core/config.py` or `src/versifai/<family>/models/` |
| Internal helpers | `src/versifai/_utils/` |

## Key Design Patterns

- **BaseAgent** — All agents subclass `BaseAgent`, which provides the ReAct loop, error recovery, and tool dispatch
- **ToolRegistry** — Tools are registered at construction time; the agent's loop automatically matches LLM tool calls to registered tools
- **BaseTool** — Every tool implements `name`, `description`, `parameters_schema`, and `execute()`. Drop-in replaceable.
- **AgentMemory** — Manages conversation history with automatic summarization for long-running tasks

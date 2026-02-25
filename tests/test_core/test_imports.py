"""Smoke tests: verify all public APIs are importable."""


def test_import_versifai():
    import versifai
    assert hasattr(versifai, "__version__")


def test_import_core():
    from versifai.core import (
        BaseAgent,
        LLMClient,
    )
    assert BaseAgent is not None
    assert LLMClient is not None


def test_import_data_agents():
    from versifai.data_agents import (
        DataEngineerAgent,
    )
    assert DataEngineerAgent is not None


def test_import_science_agents():
    from versifai.science_agents import (
        DataScientistAgent,
    )
    assert DataScientistAgent is not None


def test_import_story_agents():
    from versifai.story_agents import (
        StoryTellerAgent,
    )
    assert StoryTellerAgent is not None


def test_catalog_config():
    from versifai.core import CatalogConfig
    cfg = CatalogConfig(catalog="test", schema="default")
    assert cfg.catalog == "test"
    assert cfg.schema == "default"


def test_agent_settings_defaults():
    from versifai.core import AgentSettings
    s = AgentSettings()
    assert s.max_agent_turns == 200
    assert s.max_turns_per_source == 120


def test_llm_response_dataclass():
    from versifai.core import LLMResponse
    r = LLMResponse(
        content=[{"type": "text", "text": "hello"}],
        stop_reason="end_turn",
        usage={"prompt_tokens": 10, "completion_tokens": 5},
    )
    assert r.input_tokens == 10
    assert r.output_tokens == 5
    assert r.stop_reason == "end_turn"

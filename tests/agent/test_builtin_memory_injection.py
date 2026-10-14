"""Host-owned built-in memory injection retains the original native prompt/tool surface."""
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from agent.agent_init import _init_memory
from agent.system_prompt import _memory_parts
from tools.memory_tool import memory_tool


def bound_store():
    return SimpleNamespace(memory_enabled=True, user_profile_enabled=False,
        load_from_disk=Mock(), format_for_system_prompt=Mock(return_value="frozen canonical prompt"),
        target_enabled=Mock(return_value=True), reset_consolidation_failures=Mock(),
        invoke_memory_tool=Mock(return_value=json.dumps({"success": True})))


def test_injected_store_never_constructs_local_files_and_keeps_prompt():
    agent = SimpleNamespace(enabled_toolsets=["memory"], disabled_toolsets=[])
    store = bound_store()
    with patch("tools.memory_tool.MemoryStore", side_effect=AssertionError("No local store")):
        _init_memory(agent, {}, True, "cli", builtin_memory_store=store)
    assert agent._memory_store is store
    assert agent._memory_enabled is True
    assert agent._user_profile_enabled is False
    store.load_from_disk.assert_called_once_with()
    assert _memory_parts(agent) == ["frozen canonical prompt"]


def test_injected_failure_never_silently_falls_back_to_local_store():
    store = bound_store()
    store.load_from_disk.side_effect = EOFError("unconfirmed canonical memory")
    with patch("tools.memory_tool.MemoryStore", side_effect=AssertionError("No fallback")):
        with pytest.raises(EOFError):
            _init_memory(SimpleNamespace(enabled_toolsets=[], disabled_toolsets=[]), {}, True,
                         "cli", builtin_memory_store=store)


def test_invalid_injected_store_is_rejected():
    with pytest.raises(TypeError):
        _init_memory(SimpleNamespace(enabled_toolsets=[], disabled_toolsets=[]), {}, True,
                     "cli", builtin_memory_store=SimpleNamespace())


def test_bound_tool_preserves_arguments_and_native_json_result():
    store = bound_store()
    operations = [{"action": "add", "content": "canonical fixture"}]
    result = memory_tool(action="batch", target="user", operations=operations, store=store)
    assert json.loads(result) == {"success": True}
    store.invoke_memory_tool.assert_called_once_with(action="batch", target="user", content=None,
        old_text=None, new_text=None, operations=operations)


def test_bound_tool_transport_failure_does_not_mutate_local_files():
    store = bound_store()
    store.invoke_memory_tool.side_effect = EOFError("missing receipt")
    with pytest.raises(EOFError):
        memory_tool(action="add", content="fixture", store=store)


def test_bound_tool_wrong_result_is_rejected():
    store = bound_store()
    store.invoke_memory_tool.return_value = {"success": True}
    with pytest.raises(TypeError):
        memory_tool(action="add", content="fixture", store=store)


def test_original_constructor_forwards_bound_store_without_local_memory(tmp_path):
    from providers import register_provider
    from providers.base import ProviderProfile
    from run_agent import AIAgent
    from hermes_state import SessionDB
    client = SimpleNamespace(close=Mock())
    class MemoryFixtureProvider(ProviderProfile):
        def create_client(self, **kwargs):
            return client
    register_provider(MemoryFixtureProvider(name="builtin-injection-fixture", supports_model_listing=False))
    store = bound_store()
    db = SessionDB(tmp_path / "canonical-fixture.db")
    db.create_session("builtin-injection-fixture", "test")
    try:
        with patch("tools.memory_tool.MemoryStore", side_effect=AssertionError("No local memory")):
            agent = AIAgent(provider="builtin-injection-fixture", model="synthetic-no-inference",
                api_key="not-a-credential", base_url="http://127.0.0.1:1/no-network",
                session_id="builtin-injection-fixture", session_db=db, builtin_memory_store=store,
                enabled_toolsets=[], skip_context_files=True, skip_memory=True,
                skip_background_review=True, quiet_mode=True)
        assert agent._memory_store is store
        assert _memory_parts(agent) == ["frozen canonical prompt"]
        agent.release_clients()
    finally:
        db.close()

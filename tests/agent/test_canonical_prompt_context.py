"""Original prompt caching/restore behavior with a trusted canonical renderer."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from agent.system_prompt import (
    build_system_prompt, build_system_prompt_parts, invalidate_system_prompt, reconstruct_static_prefix,
)


def fixture_agent(provider):
    return SimpleNamespace(prompt_context_provider=provider, _cached_system_prompt=None,
        _cached_system_prompt_static=None, _memory_store=SimpleNamespace(load_from_disk=Mock()),
        _emit_diagnostic_status=Mock(), _use_prompt_caching=True)


def fixture_provider():
    return SimpleNamespace(build_parts=Mock(return_value={
        'stable': 'canonical identity and auto-loaded skill',
        'context': 'canonical project guidance', 'volatile': 'canonical memory and plugin section'}),
        invalidate=Mock())


def test_native_tiers_and_static_cache_never_read_local_context():
    provider = fixture_provider()
    agent = fixture_agent(provider)
    with patch('agent.system_prompt._identity_parts', side_effect=AssertionError('No local identity')):
        result = build_system_prompt(agent, 'caller fixture')
    assert result == '\n\n'.join(provider.build_parts.return_value.values())
    assert agent._cached_system_prompt_static == 'canonical identity and auto-loaded skill'
    provider.build_parts.assert_called_once_with(system_message='caller fixture')


@pytest.mark.parametrize('parts', [None, {}, {'stable': 123, 'context': '', 'volatile': ''},
    {'stable': '', 'context': '', 'volatile': '', 'extra': ''}])
def test_invalid_canonical_receipt_has_no_local_fallback(parts):
    provider = fixture_provider()
    provider.build_parts.return_value = parts
    with patch('agent.system_prompt._identity_parts', side_effect=AssertionError('No fallback')):
        with pytest.raises(TypeError):
            build_system_prompt_parts(fixture_agent(provider))


def test_failure_keeps_native_invalidation_state_until_parent_confirms():
    provider = fixture_provider()
    agent = fixture_agent(provider)
    agent._cached_system_prompt = 'previous canonical prompt'
    provider.invalidate.side_effect = EOFError('unconfirmed invalidate')
    with pytest.raises(EOFError):
        invalidate_system_prompt(agent)
    assert agent._cached_system_prompt == 'previous canonical prompt'
    agent._memory_store.load_from_disk.assert_not_called()


def test_native_invalidation_refreshes_canonical_context_and_memory():
    provider = fixture_provider()
    agent = fixture_agent(provider)
    agent._cached_system_prompt = 'old'
    agent._plugin_system_prompt_sections_snapshot = ['original plugin']
    invalidate_system_prompt(agent)
    provider.invalidate.assert_called_once_with()
    agent._memory_store.load_from_disk.assert_called_once_with()
    assert agent._cached_system_prompt is None
    assert agent._plugin_system_prompt_sections_previous == ['original plugin']
    assert not hasattr(agent, '_plugin_system_prompt_sections_snapshot')


def test_original_resume_reconstructs_only_matching_canonical_prefix():
    provider = fixture_provider()
    agent = fixture_agent(provider)
    agent._cached_system_prompt = 'canonical identity and auto-loaded skill\n\nstored context'
    reconstruct_static_prefix(agent)
    assert agent._cached_system_prompt_static == 'canonical identity and auto-loaded skill'
    agent._cached_system_prompt_static = None
    agent._cached_system_prompt = 'different persisted prompt'
    reconstruct_static_prefix(agent)
    assert agent._cached_system_prompt_static is None


def test_real_original_constructor_passes_trusted_provider(tmp_path):
    from providers import register_provider
    from providers.base import ProviderProfile
    from run_agent import AIAgent
    from hermes_state import SessionDB
    provider = fixture_provider()
    client = SimpleNamespace(close=Mock())
    class FixtureProvider(ProviderProfile):
        def create_client(self, **kwargs):
            return client
    register_provider(FixtureProvider(name='canonical-prompt-fixture', supports_model_listing=False))
    db = SessionDB(tmp_path / 'canonical.db')
    db.create_session('canonical-prompt-fixture', 'test')
    try:
        agent = AIAgent(provider='canonical-prompt-fixture', model='synthetic-no-inference',
            api_key='not-a-credential', base_url='http://127.0.0.1:1/no-network',
            session_id='canonical-prompt-fixture', session_db=db, prompt_context_provider=provider,
            enabled_toolsets=[], skip_memory=True, skip_context_files=True,
            skip_background_review=True, quiet_mode=True)
        assert agent.prompt_context_provider is provider
        with patch('agent.system_prompt._identity_parts', side_effect=AssertionError('No local identity')):
            assert agent._build_system_prompt().startswith('canonical identity')
        agent.release_clients()
    finally:
        db.close()

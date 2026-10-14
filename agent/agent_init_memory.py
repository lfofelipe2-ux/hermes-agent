"""Built-in memory initialization, including trusted host-owned stores."""

from contextlib import suppress


def init_builtin_memory(agent, config, skip_memory, builtin_memory_store=None):
    mem_config = None
    # Persistent memory (MEMORY.md + USER.md) — loaded from disk
    agent._memory_store = None
    agent._memory_enabled = False
    agent._user_profile_enabled = False
    agent._memory_nudge_interval = 10
    agent._turns_since_memory = 0
    agent._iters_since_skill = 0
    # skip_memory skips the external *provider*; enabled_toolsets=["memory"] still gets the
    # built-in store so the memory tool never sees store=None.
    # Flush/background agents can still pass enabled_toolsets=["memory"] so the built-in file store exists
    # and the memory tool does not fail with store=None (#65429). A toolset on disabled_toolsets is not a
    # request: a caller that denylists memory while its default toolset still names it must not get
    # MEMORY.md loaded by an enabled-only check. (Cron agents now run with skip_memory=False and take the
    # normal path here.)
    _memory_toolset_requested = (
        "memory" in (agent.enabled_toolsets or [])
        and "memory" not in (agent.disabled_toolsets or [])
    )
    if builtin_memory_store is not None:
        # Trusted host-owned injection: no local MemoryStore construction or fallback.
        # This replaces only the built-in store; external providers keep their lifecycle.
        required = ("load_from_disk", "format_for_system_prompt", "target_enabled",
                    "reset_consolidation_failures")
        if (any(not callable(getattr(builtin_memory_store, name, None)) for name in required)
                or type(getattr(builtin_memory_store, "memory_enabled", None)) is not bool
                or type(getattr(builtin_memory_store, "user_profile_enabled", None)) is not bool):
            raise TypeError("Injected built-in memory store does not implement the native contract")
        from tools.memory_tool import get_builtin_memory_config
        mem_config = get_builtin_memory_config(config)
        agent._memory_enabled = builtin_memory_store.memory_enabled
        agent._user_profile_enabled = builtin_memory_store.user_profile_enabled
        agent._memory_nudge_interval = int(mem_config.get("nudge_interval", 10))
        agent._memory_store = builtin_memory_store
        agent._memory_store.load_from_disk()
    elif not skip_memory or _memory_toolset_requested:
        # Memory is optional — don't break agent init
        with suppress(Exception):
            from tools.memory_tool import (
                MemoryStore, get_builtin_memory_config, get_builtin_memory_store_flags,
            )
            mem_config = get_builtin_memory_config(config)
            agent._memory_enabled, agent._user_profile_enabled = get_builtin_memory_store_flags(
                config
            )
            agent._memory_nudge_interval = int(mem_config.get("nudge_interval", 10))
            if agent._memory_enabled or agent._user_profile_enabled:
                agent._memory_store = MemoryStore(
                    memory_char_limit=mem_config.get("memory_char_limit", 2200),
                    user_char_limit=mem_config.get("user_char_limit", 1375),
                    memory_enabled=agent._memory_enabled,
                    user_profile_enabled=agent._user_profile_enabled,
                )
                agent._memory_store.load_from_disk()

    return mem_config

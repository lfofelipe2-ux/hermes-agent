"""Trusted host-owned rendering seam; no local context fallback or transport."""


def validate_prompt_context_provider(provider):
    if provider is not None and not all(
        callable(getattr(provider, name, None)) for name in ("build_parts", "invalidate")
    ):
        raise TypeError("Canonical prompt provider requires build_parts and invalidate")


def build_canonical_prompt_parts(provider, system_message):
    validate_prompt_context_provider(provider)
    result = provider.build_parts(system_message=system_message)
    if (not isinstance(result, dict) or set(result) != {"stable", "context", "volatile"}
            or not all(isinstance(value, str) for value in result.values())):
        raise TypeError("Canonical prompt provider must return the three native text tiers")
    return dict(result)

"""Config-driven selection of equation-recognition providers (feature 008, FR-010/FR-011).

Resolves the configured backends to concrete :class:`~equation_extraction.providers.EquationProvider`
instances, keyed by provider role. The default roles are ``qwen_vl`` (Qwen2.5-VL via
OpenAI-compatible API) and ``generic`` (plain-text passthrough fallback). New providers register
here via :func:`register_provider` without touching stage orchestration or the Equation Extraction
Context contract (constitution X). An unknown role falls back to :class:`GenericProvider`.
"""

from __future__ import annotations

from typing import Callable

from equation_extraction.providers import (
    EquationProvider,
    GenericProvider,
    QwenVLProvider,
)

__all__ = [
    "register_provider",
    "resolve_provider",
    "resolve_providers",
    "provider_identities",
    "close_providers",
]

_REGISTRY: dict[str, Callable[[], EquationProvider]] = {
    "qwen_vl": QwenVLProvider,
    "generic": GenericProvider,
}


def register_provider(role: str, factory: Callable[[], EquationProvider]) -> None:
    """Register a provider factory under ``role`` (for tests or alternative backends)."""
    _REGISTRY[role] = factory


def resolve_provider(role: str) -> EquationProvider:
    """Return a provider instance for ``role``; unknown roles fall back to the generic provider."""
    factory = _REGISTRY.get(role, GenericProvider)
    return factory()


def resolve_providers() -> dict[str, EquationProvider]:
    """Instantiate one provider per role for the document run (FR-011)."""
    return {role: factory() for role, factory in _REGISTRY.items()}


def provider_identities(providers: dict[str, EquationProvider]) -> dict[str, str]:
    """Map provider role → backend identity for the context ``providers`` field and cache key."""
    return {role: getattr(prov, "backend", role) for role, prov in providers.items()}


def close_providers(providers: dict[str, EquationProvider]) -> None:
    """Release any resources (e.g. pooled HTTP clients) held by resolved providers.

    Providers are not required to expose ``close``; those that do (``QwenVLProvider``) get
    their pooled client closed at the end of a document run. Errors are swallowed — cleanup
    must never fail the stage.
    """
    for prov in providers.values():
        close = getattr(prov, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # pragma: no cover - cleanup is best-effort
                pass

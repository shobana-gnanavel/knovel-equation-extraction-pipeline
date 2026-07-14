"""Category → provider selection (feature 008, FR-010/FR-012).

Maps an equation's content category to a provider role using the default mapping
(:data:`~equation_extraction.classifier.DEFAULT_PROVIDER_BY_CATEGORY`) with configurable overrides
parsed from ``KNOVEL_EQUATION_PROVIDER_MAP`` (e.g. ``"chemical_equation=generic"``). Pure and
deterministic; selection never decides recognition, only routing.
"""

from __future__ import annotations

from typing import Any

from equation_extraction.classifier import DEFAULT_PROVIDER_BY_CATEGORY

__all__ = ["parse_provider_map", "select_provider"]

_VALID_ROLES = {"qwen_vl", "generic"}


def parse_provider_map(raw: str) -> dict[str, str]:
    """Parse ``"cat=role,cat2=role2"`` into a category→role override map (ignores malformed pairs)."""
    overrides: dict[str, str] = {}
    for pair in (raw or "").split(","):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        category, _, role = pair.partition("=")
        category, role = category.strip(), role.strip()
        if category in DEFAULT_PROVIDER_BY_CATEGORY and role in _VALID_ROLES:
            overrides[category] = role
    return overrides


def select_provider(category: str, *, config: Any) -> str:
    """Return the provider role for ``category`` (configured override beats the default map)."""
    overrides = parse_provider_map(getattr(config, "KNOVEL_EQUATION_PROVIDER_MAP", ""))
    if category in overrides:
        return overrides[category]
    return DEFAULT_PROVIDER_BY_CATEGORY.get(category, "generic")

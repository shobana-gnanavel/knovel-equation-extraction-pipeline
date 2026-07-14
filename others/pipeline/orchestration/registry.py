"""Provider/plugin registry for the orchestrator (feature 015, FR-009..012, 018..020).

Formalizes the existing per-stage "config string selects provider + import-guarded availability" pattern into one
registry with capability discovery, health checks, and version/interface-compatibility validation. The default
provider for each stage is declared in ``pipeline.config`` (``KNOVEL_*_PROVIDER`` / ``*_ENGINE`` / ``*_BACKEND``);
this registry surfaces descriptors for the Provider Summary and rejects incompatible providers. Imports only
``pipeline`` + stdlib.
"""

from __future__ import annotations

from pipeline import config as _config
from pipeline.models import ORCH_SCHEMA_VERSION, ProviderDescriptor

__all__ = [
    "STAGE_PROVIDER_KEYS",
    "register_provider",
    "discover_providers",
    "resolve_provider_id",
    "health_check",
    "validate_compatibility",
]

# Maps each stage to the pipeline.config attribute(s) that name its default provider/engine/backend. Used for
# capability discovery and the Provider Summary; the orchestrator never imports the provider implementations.
STAGE_PROVIDER_KEYS: dict[str, list[str]] = {
    "classification": ["CLASSIFIER_DOC_LANGUAGE_BACKEND"],
    "preprocessing": ["KNOVEL_PREPROCESS_BACKEND"],
    "layout": ["KNOVEL_LAYOUT_BACKEND"],
    "reading_order": ["KNOVEL_READING_ORDER_STRATEGY"],
    "text_extraction": ["KNOVEL_TEXT_NATIVE_ENGINE", "KNOVEL_TEXT_OCR_ENGINE"],
    "equation_extraction": ["KNOVEL_EQUATION_MATH_PROVIDER"],
    "table_extraction": ["KNOVEL_TABLE_DIGITAL_PROVIDER"],
    "visual_extraction": ["KNOVEL_VISUAL_GENERAL_PROVIDER"],
    "metadata_extraction": ["KNOVEL_METADATA_DOCUMENT_PROVIDER"],
    "relationship_builder": ["KNOVEL_RELATIONSHIP_PROVIDER"],
    "validation": ["KNOVEL_VALIDATION_PROVIDER"],
    "serialization": ["KNOVEL_EXPORT_PROVIDER"],
}

# Custom-registered providers (plugin path): (stage, provider_id) -> descriptor.
_REGISTRY: dict[tuple[str, str], ProviderDescriptor] = {}


def register_provider(descriptor: ProviderDescriptor) -> None:
    """Register a custom provider/plugin descriptor (FR-018)."""
    key = (descriptor.stage or "", descriptor.id)
    _REGISTRY[key] = descriptor


def resolve_provider_id(stage: str, overrides: dict[str, str] | None = None) -> str:
    """The configured provider id for a stage (config override wins over the KNOVEL_* default)."""
    if overrides and stage in overrides:
        return overrides[stage]
    for attr in STAGE_PROVIDER_KEYS.get(stage, []):
        value = getattr(_config, attr, None)
        if value:
            return str(value)
    return "default"


def discover_providers(stage: str | None = None) -> list[ProviderDescriptor]:
    """List provider descriptors for a stage (or all stages) for capability discovery (FR-011)."""
    descriptors: list[ProviderDescriptor] = []
    stages = [stage] if stage is not None else list(STAGE_PROVIDER_KEYS)
    for name in stages:
        provider_id = resolve_provider_id(name)
        descriptors.append(
            ProviderDescriptor(
                id=provider_id,
                stage=name,
                interface=f"I{name.title().replace('_', '')}",
                version=ORCH_SCHEMA_VERSION,
                available=True,
                compatible=True,
                health_detail="configured default",
            )
        )
    for (reg_stage, _), descriptor in _REGISTRY.items():
        if stage is None or reg_stage == stage:
            descriptors.append(descriptor)
    return descriptors


def validate_compatibility(descriptor: ProviderDescriptor) -> bool:
    """Whether a provider/plugin's declared version is compatible with the running pipeline (FR-019/20).

    Compatibility uses major-version matching against ``ORCH_SCHEMA_VERSION``; an empty/unparseable version is
    treated as incompatible.
    """
    if not descriptor.version:
        return False
    try:
        provider_major = int(descriptor.version.split(".", 1)[0])
        pipeline_major = int(ORCH_SCHEMA_VERSION.split(".", 1)[0])
    except ValueError:
        return False
    return provider_major == pipeline_major


def health_check(stage: str, provider_id: str | None = None) -> ProviderDescriptor:
    """Return a descriptor with availability + compatibility resolved for a stage's provider (FR-012)."""
    provider_id = provider_id or resolve_provider_id(stage)
    registered = _REGISTRY.get((stage, provider_id))
    if registered is not None:
        descriptor = registered
    else:
        descriptor = ProviderDescriptor(
            id=provider_id,
            stage=stage,
            interface=f"I{stage.title().replace('_', '')}",
            version=ORCH_SCHEMA_VERSION,
        )
    descriptor.compatible = validate_compatibility(descriptor)
    descriptor.health_detail = (
        "ok" if descriptor.available and descriptor.compatible else "unavailable"
    )
    return descriptor

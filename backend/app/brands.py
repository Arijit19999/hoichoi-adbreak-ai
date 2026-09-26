"""Brand catalogue loading.

Brands are data, never code: everything downstream works from the context strings in the
catalogue, so a new (unseen) brand only needs a new entry in brands.json.
"""

import json
from pathlib import Path

from .config import get_settings

REQUIRED_FIELDS = ("brand_id", "display_name", "category", "target_contexts", "negative_contexts", "creatives")


def _norm(context: str) -> str:
    return " ".join(context.lower().split())


def load_catalogue(path: Path | None = None) -> list[dict]:
    path = path or get_settings().resolve(get_settings().brands_path)
    brands = json.loads(path.read_text(encoding="utf-8"))
    for brand in brands:
        missing = [f for f in REQUIRED_FIELDS if f not in brand]
        if missing:
            raise ValueError(f"brand {brand.get('brand_id', '?')} is missing {missing}")
        brand["target_contexts"] = [_norm(c) for c in brand["target_contexts"]]
        brand["negative_contexts"] = [_norm(c) for c in brand["negative_contexts"]]
    return brands


def vocabulary(brands: list[dict]) -> tuple[list[str], list[str]]:
    """(negative contexts, target contexts) across the catalogue, deduplicated."""
    negative = sorted({c for b in brands for c in b["negative_contexts"]})
    target = sorted({c for b in brands for c in b["target_contexts"]} - set(negative))
    return negative, target

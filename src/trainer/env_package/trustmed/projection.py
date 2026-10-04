"""Action projection for the trustmed env package."""

from __future__ import annotations


def trustmed_projection(raw_texts: list[str]) -> tuple[list[str], list[int]]:
    return list(raw_texts), [1] * len(raw_texts)

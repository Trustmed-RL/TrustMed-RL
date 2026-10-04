"""Action projection for the sp_consult env package."""

from __future__ import annotations


def sp_consult_projection(raw_texts: list[str]) -> tuple[list[str], list[int]]:
    return list(raw_texts), [1] * len(raw_texts)

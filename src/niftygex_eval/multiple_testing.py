"""Holm step-down adjustment within pre-registered test families."""
from __future__ import annotations

from typing import Any


def holm(pvalues: list[float | None]) -> list[float | None]:
    """Holm-adjusted p-values (monotone, capped at 1); None entries are skipped and stay None."""
    present = [(p, i) for i, p in enumerate(pvalues) if p is not None]
    m = len(present)
    adjusted: list[float | None] = [None] * len(pvalues)
    running = 0.0
    for rank, (p, i) in enumerate(sorted(present)):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[i] = running
    return adjusted


def adjust_family(name: str, description: str, members: list[dict[str, Any]], alpha: float) -> dict[str, Any]:
    """``members``: dicts with test, p, primary. Adds p_holm and reject_holm."""
    adjusted = holm([m.get("p") for m in members])
    rows = [{**m, "p_holm": a, "reject_holm": (a is not None and a < alpha)} for m, a in zip(members, adjusted)]
    return {"family": name, "description": description, "alpha": alpha, "tests": len([m for m in members if m.get("p") is not None]), "members": rows}

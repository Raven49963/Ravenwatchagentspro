from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


def finite_float(value: Any) -> float | None:
    """Convert a value to a finite float, returning None when it is unusable."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def clamp(value: float, lower: float, upper: float) -> float:
    if lower > upper:
        raise ValueError("lower bound must not exceed upper bound")
    return float(min(upper, max(lower, value)))


def rank_correlation(left: pd.Series, right: pd.Series) -> float:
    """Return a finite Spearman-style rank correlation for two aligned series."""
    left_rank = left.rank(method="average")
    right_rank = right.rank(method="average")
    if left_rank.nunique(dropna=True) < 2 or right_rank.nunique(dropna=True) < 2:
        return 0.0
    value = float(left_rank.corr(right_rank))
    return value if math.isfinite(value) else 0.0

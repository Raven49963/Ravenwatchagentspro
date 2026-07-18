from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from .numeric import clamp


DIRECTION_MIN = -100.0
DIRECTION_MAX = 100.0
RATING_MIN = 0.0
RATING_MAX = 100.0
NEUTRAL_RATING = 50.0
MILD_DIRECTION_THRESHOLD = 8.0
STRONG_DIRECTION_THRESHOLD = 25.0


class DirectionBand(str, Enum):
    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"


@dataclass(frozen=True)
class CalibratedDirection:
    raw_directional_score: float
    directional_score: float
    signal_strength: float
    calibration_factor: float


def clip_direction(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return clamp(value, DIRECTION_MIN, DIRECTION_MAX)


def direction_to_rating(directional_score: float) -> float:
    """Map -100..100 direction onto a 0..100 rating with 50 as neutral."""
    rating = NEUTRAL_RATING + clip_direction(directional_score) / 2.0
    return clamp(rating, RATING_MIN, RATING_MAX)


def direction_band(directional_score: float) -> DirectionBand:
    score = clip_direction(directional_score)
    if score >= STRONG_DIRECTION_THRESHOLD:
        return DirectionBand.STRONG_BULLISH
    if score >= MILD_DIRECTION_THRESHOLD:
        return DirectionBand.BULLISH
    if score > -MILD_DIRECTION_THRESHOLD:
        return DirectionBand.NEUTRAL
    if score > -STRONG_DIRECTION_THRESHOLD:
        return DirectionBand.BEARISH
    return DirectionBand.STRONG_BEARISH


def calibrate_direction(
    raw_directional_score: float,
    reliability: float,
    *,
    minimum_calibration: float,
) -> CalibratedDirection:
    """Shrink a directional signal toward neutral as evidence weakens."""
    minimum = clamp(minimum_calibration, 0.0, 1.0)
    reliability = clamp(reliability, 0.0, 1.0)
    raw_score = clip_direction(raw_directional_score)
    calibration_factor = minimum + (1.0 - minimum) * reliability
    directional_score = clip_direction(raw_score * calibration_factor)
    return CalibratedDirection(
        raw_directional_score=raw_score,
        directional_score=directional_score,
        signal_strength=abs(directional_score),
        calibration_factor=calibration_factor,
    )

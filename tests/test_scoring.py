from pathlib import Path
import math
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from quant_starter.numeric import clamp, finite_float
from quant_starter.scoring import (
    DirectionBand,
    calibrate_direction,
    clip_direction,
    direction_band,
    direction_to_rating,
)


class NumericContractTests(unittest.TestCase):
    def test_finite_float_rejects_missing_and_non_finite_values(self) -> None:
        self.assertEqual(finite_float("12.5"), 12.5)
        self.assertIsNone(finite_float(None))
        self.assertIsNone(finite_float("not-a-number"))
        self.assertIsNone(finite_float(math.inf))
        self.assertIsNone(finite_float(math.nan))

    def test_clamp_validates_bounds(self) -> None:
        self.assertEqual(clamp(8.0, 0.0, 5.0), 5.0)
        self.assertEqual(clamp(-2.0, 0.0, 5.0), 0.0)
        with self.assertRaises(ValueError):
            clamp(1.0, 2.0, 1.0)


class ScoringContractTests(unittest.TestCase):
    def test_direction_and_rating_scales_share_one_mapping(self) -> None:
        cases = (
            (-120.0, -100.0, 0.0),
            (-50.0, -50.0, 25.0),
            (0.0, 0.0, 50.0),
            (50.0, 50.0, 75.0),
            (120.0, 100.0, 100.0),
        )
        for value, direction, rating in cases:
            with self.subTest(value=value):
                self.assertEqual(clip_direction(value), direction)
                self.assertEqual(direction_to_rating(value), rating)

    def test_direction_bands_preserve_decision_boundaries(self) -> None:
        cases = (
            (25.0, DirectionBand.STRONG_BULLISH),
            (8.0, DirectionBand.BULLISH),
            (7.999, DirectionBand.NEUTRAL),
            (-7.999, DirectionBand.NEUTRAL),
            (-8.0, DirectionBand.BEARISH),
            (-25.0, DirectionBand.STRONG_BEARISH),
        )
        for score, expected in cases:
            with self.subTest(score=score):
                self.assertIs(direction_band(score), expected)

    def test_calibration_shrinks_direction_toward_neutral(self) -> None:
        result = calibrate_direction(
            80.0,
            0.5,
            minimum_calibration=0.35,
        )

        self.assertAlmostEqual(result.calibration_factor, 0.675)
        self.assertAlmostEqual(result.raw_directional_score, 80.0)
        self.assertAlmostEqual(result.directional_score, 54.0)
        self.assertAlmostEqual(result.signal_strength, 54.0)


if __name__ == "__main__":
    unittest.main()

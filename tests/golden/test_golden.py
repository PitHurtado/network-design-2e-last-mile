"""Numerical-equivalence tests: every producer must reproduce its stored expectation.

    poetry run python -m unittest tests.golden.test_golden          # all
    GOLDEN_SKIP_SOLVE=1 poetry run python -m unittest tests.golden.test_golden   # no Gurobi

On a mismatch, `tests/golden/_dumps/` holds the full actual outputs for diffing.
"""

import json
import os
import unittest

from tests.golden.producers import PRODUCERS
from tests.golden.support import EXPECTED, Workspace, normalize_ids


def _normalized(value):
    """Canonical JSON with scenario ids normalized, applied to both sides of a comparison."""
    return json.loads(normalize_ids(json.dumps(value, sort_keys=True, default=str)))


class GoldenTests(unittest.TestCase):
    maxDiff = 4000

    @classmethod
    def setUpClass(cls):
        cls.ws = Workspace()

    @classmethod
    def tearDownClass(cls):
        cls.ws.close()

    def _check(self, name: str):
        expected = _normalized(json.loads((EXPECTED / f"{name}.json").read_text()))
        self.assertEqual(_normalized(PRODUCERS[name](self.ws)), expected, f"golden {name} changed")

    def test_g1_fit(self):
        self._check("g1_fit")

    def test_g2_recalibrate(self):
        self._check("g2_recalibrate")

    def test_g3_generate(self):
        self._check("g3_generate")

    def test_g4_compare(self):
        self._check("g4_compare")

    def test_g5_spatial(self):
        self._check("g5_spatial")

    def test_g6_reports(self):
        self._check("g6_reports")

    def test_g7_ca(self):
        self._check("g7_ca")

    def test_g9_capacity(self):
        self._check("g9_capacity")

    @unittest.skipIf(os.environ.get("GOLDEN_SKIP_SOLVE"), "GOLDEN_SKIP_SOLVE set")
    def test_g8_solve(self):
        self._check("g8_solve")


if __name__ == "__main__":
    unittest.main()

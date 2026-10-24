"""Reports must reject componentwise readback and geometry regressions."""

import importlib.util
from pathlib import Path
import sys
import unittest


SPEC = importlib.util.spec_from_file_location("packing_comparison_tested", Path(__file__).with_name("compare.py"))
compare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = compare
SPEC.loader.exec_module(compare)


class CostGuardTests(unittest.TestCase):
    def baseline(self):
        return {"pictures": 21, "prediction_slices": 100, "prediction_bytes": 2000,
                "output_readback_bytes": 76800, "coded_width": 160,
                "coded_height": 64, "input_pictures": 1}

    def test__given_fewer_pictures_but_larger_output_readback__when_comparing__then_regression_is_rejected(self):
        before = self.baseline()
        after = {**before, "pictures": 20, "output_readback_bytes": 122880}
        with self.assertRaisesRegex(AssertionError, "output_readback_bytes"):
            compare.assert_costs(before, after)

    def test__given_same_geometry__when_any_guarded_cost_increases__then_other_savings_cannot_hide_it(self):
        before = self.baseline()
        for key in compare.GUARDED:
            after = {**before, **{name: 0 for name in compare.GUARDED}, key: before[key]+1}
            with self.subTest(cost=key), self.assertRaisesRegex(AssertionError, key):
                compare.assert_costs(before, after)
        compare.assert_costs(before, {**before, "pictures": 20})
        with self.assertRaisesRegex(AssertionError, "coded_width"):
            compare.assert_costs(before, {**before, "coded_width": 176})


if __name__ == "__main__":
    unittest.main(verbosity=2)

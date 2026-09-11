from __future__ import annotations

import unittest

from experiments.run_beta_gen_factorial import (
    EXPECTED_TREATMENTS,
    REPOSITORY_ROOT,
    VARIANT_CONFIGS,
    load_yaml,
    preflight,
)


class Email3ExperimentTests(unittest.TestCase):
    def test_six_primary_conditions_are_balanced(self) -> None:
        variants = (
            "A_original",
            "B_beta_input",
            "D_input_target_pid02",
            "A_original_pid1",
            "B_beta_input_pid1",
            "D_input_target_pid1",
        )
        configs = {
            variant: load_yaml(REPOSITORY_ROOT / "configs" / VARIANT_CONFIGS[variant])
            for variant in variants
        }
        record = preflight(configs, (20260822,))

        self.assertEqual(set(record["conditions"]), set(variants))
        self.assertEqual(
            {EXPECTED_TREATMENTS[variant][2] for variant in variants},
            {0.2, 1.0},
        )
        self.assertIn(
            ["A_original_pid1", "B_beta_input_pid1"],
            record["exact_initial_function_pairs"],
        )


if __name__ == "__main__":
    unittest.main()

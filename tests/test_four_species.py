from __future__ import annotations

import unittest

from forwardfm_step1.data import configured_species, response_population_sql


class FourSpeciesConfigurationTests(unittest.TestCase):
    @staticmethod
    def config(species=None):
        data = {}
        if species is not None:
            data["generated_species"] = species
        return {"data": data}

    def test_legacy_default_is_unchanged(self) -> None:
        self.assertEqual(configured_species(self.config()), (-211, 211, 2212))

    def test_four_species_order_is_preserved(self) -> None:
        config = self.config([11, -211, 211, 2212])
        self.assertEqual(configured_species(config), (11, -211, 211, 2212))
        selection = response_population_sql(config)
        self.assertIn("gen_pid = 11", selection)
        self.assertIn("trigger_mcindex = mcindex", selection)
        self.assertIn("usable_for_hadron_response_training", selection)

    def test_duplicates_and_unsupported_species_fail(self) -> None:
        with self.assertRaises(ValueError):
            configured_species(self.config([11, 11]))
        with self.assertRaises(ValueError):
            configured_species(self.config([13]))


if __name__ == "__main__":
    unittest.main()

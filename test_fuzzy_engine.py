"""Checks that the model reproduces every number in the presentation.
Run:  python -m unittest -v
"""
import unittest

from fuzzy_engine import (DEMAND, PRICE_LABELS, STOCK, FuzzyPricingEngine, Guardrails, trapmf)

DECK_LIMITS = Guardrails(min_price=3200, max_price=4800, max_change_pct=15)


class TestMemberships(unittest.TestCase):
    def test_price_labels(self):            # slides 4-5
        self.assertAlmostEqual(PRICE_LABELS.fuzzify(450)["Fair"], 0.8)
        s = PRICE_LABELS.fuzzify(600)
        self.assertAlmostEqual(s["Fair"], 0.6)
        self.assertAlmostEqual(s["Expensive"], 0.2)
        self.assertAlmostEqual(PRICE_LABELS.fuzzify(700)["Expensive"], 0.6)
        self.assertAlmostEqual(PRICE_LABELS.fuzzify(350)["Cheap"], 0.4)

    def test_case_inputs(self):              # slide 9
        self.assertAlmostEqual(DEMAND.fuzzify(8)["High"], 0.7)
        self.assertAlmostEqual(DEMAND.fuzzify(8)["Medium"], 0.3)
        self.assertAlmostEqual(STOCK.fuzzify(30)["Low"], 0.6)

    def test_shoulders(self):
        self.assertEqual(float(trapmf(-100, -float("inf"), -float("inf"), 0, 1)), 1.0)
        self.assertEqual(float(trapmf(100, 0, 1, float("inf"), float("inf"))), 1.0)


class TestCaseStudy(unittest.TestCase):
    def setUp(self):
        self.eng = FuzzyPricingEngine(guardrails=DECK_LIMITS)
        self.x = {"demand": 8, "stock": 30}

    def test_rule_strengths(self):
        d = self.eng.evaluate(self.x, 4000)
        self.assertEqual([round(a.strength, 2) for a in d.activations], [0.6, 0.3])

    def test_sugeno_4387(self):
        d = self.eng.evaluate(self.x, 4000, "sugeno")
        self.assertAlmostEqual(d.change_pct, 9.6667, places=3)
        self.assertEqual(d.final_price, 4387)

    def test_mamdani_4374(self):
        d = self.eng.evaluate(self.x, 4000, "mamdani")
        self.assertAlmostEqual(d.change_pct, 9.3556, places=3)
        self.assertEqual(d.final_price, 4374)

    def test_sensitivity_4383(self):
        self.assertEqual(self.eng.evaluate({"demand": 7.9, "stock": 30}, 4000).final_price, 4383)

    def test_no_rule_fallback(self):
        d = self.eng.evaluate({"demand": 8, "stock": 80}, 4000)
        self.assertIsNotNone(d.fallback_reason)
        self.assertEqual(d.final_price, 4000)

    def test_invalid_input_fallback(self):
        d = self.eng.evaluate({"demand": 12, "stock": 30}, 4000)
        self.assertIn("invalid", d.fallback_reason)
        self.assertEqual(d.final_price, 4000)


class TestGuardrails(unittest.TestCase):
    def test_margin_floor_3733(self):        # slide on safety checks
        g = Guardrails(unit_cost=2800, min_margin=0.25)
        self.assertAlmostEqual(g.margin_floor(), 3733.33, places=2)

    def test_floor_applied(self):
        g = Guardrails(min_price=3200, max_price=4800, max_change_pct=15, unit_cost=2800, min_margin=0.25)
        eng = FuzzyPricingEngine.from_rule_set("extended", guardrails=g)
        d = eng.evaluate({"demand": 1, "stock": 95, "competitor_gap": 0}, 4000)
        self.assertLess(d.raw_price, 3733)
        self.assertEqual(d.final_price, 3733)

    def test_change_cap(self):
        price, applied, _ = Guardrails(max_change_pct=5).apply(4000, 4500)
        self.assertAlmostEqual(price, 4200)
        self.assertTrue(applied)

    def test_conflict_keeps_reference(self):
        price, _, warnings = Guardrails(max_price=3000, unit_cost=2800, min_margin=0.25).apply(4000, 4100)
        self.assertEqual(price, 4000)
        self.assertTrue(warnings)


if __name__ == "__main__":
    unittest.main()

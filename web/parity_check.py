"""
parity_check.py — checks that web/engine.js gives the same answers as fuzzy_engine.py.

Run from the project folder (needs Node.js):
    python web/parity_check.py
"""
import json
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from fuzzy_engine import FuzzyPricingEngine, Guardrails  # noqa: E402

JS_RUNNER = r"""
const E = require(process.argv[1]);
const cases = JSON.parse(require("fs").readFileSync(0, "utf8"));
const out = cases.map((c) => {
  const eng = E.FuzzyPricingEngine.fromRuleSet(c.rules, { guardrails: new E.Guardrails(c.guardrails) });
  const d = eng.evaluate(c.inputs, c.reference, c.method);
  return { final_price: d.final_price, change_pct: d.change_pct, fallback: d.fallback_reason,
           strengths: d.activations.map((a) => a.strength), applied: d.guardrails_applied,
           warnings: d.warnings };
});
process.stdout.write(JSON.stringify(out));
"""


def make_cases(n=3000, seed=7):
    rnd = random.Random(seed)
    cases = [  # the deck's numbers first
        dict(rules="case-study", method="sugeno", reference=4000, inputs={"demand": 8, "stock": 30},
             guardrails=dict(min_price=3200, max_price=4800, max_change_pct=15)),
        dict(rules="case-study", method="mamdani", reference=4000, inputs={"demand": 8, "stock": 30},
             guardrails=dict(min_price=3200, max_price=4800, max_change_pct=15)),
        dict(rules="extended", method="sugeno", reference=4000,
             inputs={"demand": 1, "stock": 95, "competitor_gap": 0},
             guardrails=dict(min_price=3200, max_price=4800, max_change_pct=15, unit_cost=2800, min_margin=0.25)),
        dict(rules="case-study", method="sugeno", reference=4000, inputs={"demand": 12, "stock": 30},
             guardrails={}),
    ]
    for _ in range(n):
        ref = rnd.choice([999, 2500, 4000, 7350, 10000])
        gr = {}
        if rnd.random() < 0.8:
            gr.update(min_price=ref * rnd.uniform(0.7, 0.95), max_price=ref * rnd.uniform(1.03, 1.3))
        if rnd.random() < 0.7:
            gr["max_change_pct"] = rnd.choice([3, 5, 10, 15])
        if rnd.random() < 0.4:
            gr.update(unit_cost=ref * rnd.uniform(0.5, 0.95), min_margin=rnd.choice([0.1, 0.25, 0.4]))
        cases.append(dict(
            rules=rnd.choice(["case-study", "extended"]),
            method=rnd.choice(["sugeno", "sugeno", "mamdani"]),
            reference=ref,
            inputs={"demand": round(rnd.uniform(-0.5, 10.5), 2), "stock": round(rnd.uniform(-2, 102), 1),
                    "competitor_gap": round(rnd.uniform(-31, 31), 1)},
            guardrails=gr,
        ))
    return cases


def main():
    cases = make_cases()
    js = subprocess.run(["node", "-e", JS_RUNNER, os.path.join(HERE, "engine.js")],
                        input=json.dumps(cases), capture_output=True, text=True, check=True)
    js_results = json.loads(js.stdout)

    bad = 0
    for c, j in zip(cases, js_results):
        eng = FuzzyPricingEngine.from_rule_set(c["rules"], guardrails=Guardrails(**c["guardrails"]))
        p = eng.evaluate(c["inputs"], c["reference"], c["method"])
        same = (p.final_price == j["final_price"]
                and (p.fallback_reason is None) == (j["fallback"] is None)
                and [a.strength for a in p.activations] == j["strengths"]
                and p.guardrails_applied == j["applied"]
                and len(p.warnings) == len(j["warnings"])
                and (p.change_pct is None or abs(p.change_pct - j["change_pct"]) < 1e-9))
        if not same:
            bad += 1
            if bad <= 5:
                print("MISMATCH", c, "\n  python:", p.final_price, p.change_pct, p.guardrails_applied,
                      "\n  js:    ", j)
    print(f"{len(cases)} cases, {len(cases) - bad} match, {bad} differ")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

"""
main.py — command-line entry point for the fuzzy pricing model.

Examples
--------
  python main.py demo                                  # walk through every example from the deck
  python main.py price --demand 8 --stock 30           # price one product, step by step
  python main.py price --demand 8 --stock 30 --method both
  python main.py price --demand 3 --stock 85 --gap 6 --rules extended --cost 2800 --min-margin 0.25
  python main.py plots                                 # save 6 charts to ./charts
  python main.py app                                   # interactive window with sliders
"""

from __future__ import annotations

import argparse
import sys

from fuzzy_engine import (STOCK, FuzzyPricingEngine, Guardrails, PRICE_LABELS, cost_plus_price,
                          hard_demand_rule_change, hard_stock_rule_change, log_decision)

# Make ₹, × and ÷ print correctly on Windows terminals too.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


def make_guardrails(args) -> Guardrails:
    ref = args.reference
    return Guardrails(
        min_price=args.min_price if args.min_price is not None else 0.8 * ref,
        max_price=args.max_price if args.max_price is not None else 1.2 * ref,
        max_change_pct=args.max_change,
        unit_cost=args.cost,
        min_margin=args.min_margin,
    )


def cmd_price(args):
    eng = FuzzyPricingEngine.from_rule_set(args.rules, guardrails=make_guardrails(args))
    inputs = {"demand": args.demand, "stock": args.stock, "competitor_gap": args.gap}
    methods = ["sugeno", "mamdani"] if args.method == "both" else [args.method]
    for i, m in enumerate(methods):
        dec = eng.evaluate(inputs, args.reference, m, args.product)
        if args.log:
            log_decision(dec, args.log)
        if args.json:
            print(dec.to_json())
        else:
            if i:
                print("\n" + "-" * 72 + "\n")
            print(dec.explain())
    if args.log:
        print(f"\n(decision record appended to {args.log})")


def banner(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def cmd_demo(_args):
    ref = 4000
    case = FuzzyPricingEngine(guardrails=Guardrails(min_price=3200, max_price=4800, max_change_pct=15))
    headphones = {"demand": 8, "stock": 30}

    banner("1. Key idea: a price can be partly 'fair' and partly 'expensive'")
    for p in (450, 600, 700):
        s = PRICE_LABELS.fuzzify(p)
        print(f"   ₹{p}:  " + "   ".join(f"{t} {m:.1f}" for t, m in s.items()))

    banner("2. Why hard rules fail (IF stock < 20% THEN +10%)")
    for s in (19.9, 20.1):
        fuzzy = 10 * float(STOCK.membership("Low", s))
        print(f"   stock {s:>4}%:  hard rule {hard_stock_rule_change(s):+5.1f}%    fuzzy rule {fuzzy:+5.2f}%")

    banner("3. Worked example: ₹4,000 headphones, demand 8/10, stock 30%")
    print(case.evaluate(headphones, ref, "sugeno", "headphones").explain())

    banner("4. Same example with the Mamdani (balance point) method")
    m = case.evaluate(headphones, ref, "mamdani", "headphones")
    print(f"   Centroid = {m.change_pct:+.2f}%  →  ₹{m.final_price:,.0f}   (Sugeno gave ₹4,387)")

    banner("5. Small input change → small price change")
    print("   demand   fuzzy price   hard rule (IF demand ≥ 8 THEN +12% ELSE +5%)")
    for d in (7.5, 7.9, 8.0, 8.1, 8.5):
        f = case.evaluate({"demand": d, "stock": 30}, ref).final_price
        h = ref * (1 + hard_demand_rule_change(d) / 100)
        print(f"   {d:>5}    ₹{f:,.0f}        ₹{h:,.0f}")

    banner("6. Safety check: the margin floor protects profit")
    g = Guardrails(min_price=3200, max_price=4800, max_change_pct=15, unit_cost=2800, min_margin=0.25)
    print(f"   Lowest safe price = ₹2,800 ÷ (1 − 0.25) = ₹{g.margin_floor():,.2f}")
    print(f"   (cost-plus at 25% markup would be ₹{cost_plus_price(2800, 0.25):,.0f} – only a 20% margin)")
    ext = FuzzyPricingEngine.from_rule_set("extended", guardrails=g)
    d = ext.evaluate({"demand": 1, "stock": 95, "competitor_gap": 0}, ref, "sugeno", "headphones")
    print(f"   Weak demand + lots of stock → fuzzy says {d.change_pct:+.2f}% = ₹{d.raw_price:,.0f}")
    for a in d.guardrails_applied:
        print(f"   Safety check: {a}")
    print(f"   Final price: ₹{d.final_price:,.0f}")

    banner("7. Fallback: when no rule applies, keep the current price")
    fb = case.evaluate({"demand": 8, "stock": 80}, ref, "sugeno", "headphones")
    print(f"   demand 8, stock 80% (not 'Low') → {fb.fallback_reason}")
    print(f"   Final price: ₹{fb.final_price:,.0f}")
    print("\nAll numbers are hypothetical teaching values, not results from a real company.")


def cmd_plots(args):
    import matplotlib
    matplotlib.use("Agg")
    from visualize import save_all_charts
    for p in save_all_charts(args.out):
        print("saved", p)


def cmd_app(_args):
    from visualize import run_app
    run_app()


def build_parser():
    p = argparse.ArgumentParser(description="Fuzzy price optimization model")
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("price", help="recommend a price for one product, step by step")
    pr.add_argument("--demand", type=float, required=True, help="demand score 0–10")
    pr.add_argument("--stock", type=float, required=True, help="stock level 0–100 (%% of normal stock)")
    pr.add_argument("--gap", type=float, default=0.0,
                    help="competitor gap %% (positive = we are pricier); used by --rules extended")
    pr.add_argument("--reference", type=float, default=4000, help="current/reference price in ₹")
    pr.add_argument("--method", choices=["sugeno", "mamdani", "both"], default="sugeno")
    pr.add_argument("--rules", choices=["case-study", "extended"], default="case-study")
    pr.add_argument("--product", default="headphones")
    pr.add_argument("--min-price", type=float, help="price floor (default 80%% of reference)")
    pr.add_argument("--max-price", type=float, help="price ceiling (default 120%% of reference)")
    pr.add_argument("--max-change", type=float, default=15, help="largest allowed change in %%")
    pr.add_argument("--cost", type=float, help="unit cost in ₹ (enables the margin floor)")
    pr.add_argument("--min-margin", type=float, help="minimum gross margin, e.g. 0.25")
    pr.add_argument("--json", action="store_true", help="print the full decision record as JSON")
    pr.add_argument("--log", help="append the decision record to this .jsonl file")
    pr.set_defaults(func=cmd_price)

    sub.add_parser("demo", help="run every example from the presentation").set_defaults(func=cmd_demo)

    pl = sub.add_parser("plots", help="save charts as PNG files")
    pl.add_argument("--out", default="charts")
    pl.set_defaults(func=cmd_plots)

    sub.add_parser("app", help="open the interactive slider app").set_defaults(func=cmd_app)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)

# Fuzzy Price Optimization – Python working model

A from-scratch fuzzy logic pricing engine that follows the same 6 steps as the presentation and reproduces its numbers exactly (₹4,000 headphones → **₹4,387** with Sugeno, **₹4,374** with Mamdani).

> All prices, rules and outputs are hypothetical teaching values, not results from a real company.

## Setup

Needs Python 3.9+ with NumPy and matplotlib:

```
pip install numpy matplotlib
```

## Run it

| Command | What it does |
|---|---|
| `python main.py demo` | Walks through every example from the deck: scores, price jump, worked example, Mamdani, sensitivity, margin floor, fallback |
| `python main.py price --demand 8 --stock 30` | Prices one product and prints each step |
| `python main.py price --demand 8 --stock 30 --method both` | Compares Sugeno and Mamdani |
| `python main.py price --demand 3 --stock 85 --gap 6 --rules extended --cost 2800 --min-margin 0.25` | Uses the fuller 11-rule system with competitor prices and a profit floor |
| `python main.py price ... --json --log decisions.jsonl` | Prints the full decision record and saves it (the audit trail) |
| `python main.py plots` | Saves 6 charts to `charts/` (useful for the report or slides) |
| `python main.py app` | Opens an interactive window: drag the sliders and watch the scores, rules and price change live |
| `python -m unittest -v` | Checks the model still matches the deck's numbers |

## Web app

`web/index.html` is a browser version of the slider app: inputs, rule set, method and safety limits on the left, the six steps worked out live on the right, plus a price map, a price-vs-demand chart and the JSON decision record. It runs `web/engine.js`, a JavaScript port of `fuzzy_engine.py` that gives the same prices (NumPy's summation order is copied so ₹0.50 ties round the same way).

| Command | What it does |
|---|---|
| `python web/parity_check.py` | Runs 3,004 cases through both engines and checks they match (needs Node.js) |
| `python -m http.server 8000 --directory web` | Serves the web app locally at http://localhost:8000 |

## Files

| File | Contents |
|---|---|
| `fuzzy_engine.py` | The model: membership functions, inputs, rules, Sugeno and Mamdani inference, safety checks, decision record |
| `visualize.py` | Charts and the interactive slider app |
| `main.py` | Command-line interface |
| `test_fuzzy_engine.py` | 13 tests against the deck's numbers |

## How the model works

1. **Collect data:** checks that demand (0–10), stock (0–100%) and competitor gap (−30 to +30%) are present and in range. If not, it keeps the reference price.
2. **Turn into scores (fuzzification):** trapezoid/triangle membership functions give each input a 0–1 score for *Low/Medium/High*, *Low/Normal/High* and *Cheaper/Similar/Pricier*.
3. **Apply the rules:** each IF–THEN rule gets a strength. AND takes the smaller score; OR takes the bigger one.
4. **Combine** and
5. **Get one number (defuzzification):**
   - *Sugeno:* weighted average of the rules' fixed outputs, (0.6×12 + 0.3×5) ÷ 0.9 = 9.67%.
   - *Mamdani:* cut each output triangle at its rule's strength, merge the shapes, and take the balance point (centroid), giving 9.36%.
6. **Safety checks:** price floor and ceiling (default 80–120% of the reference), a ±15% change cap, and an optional profit-margin floor of cost ÷ (1 − margin). If no rule applies, the reference price is kept.

### Rule sets

- **case-study** (default): the two rules from the deck.
  - A: IF demand is High AND stock is Low THEN +12%
  - B: IF demand is Medium AND stock is Low THEN +5%
- **extended:** all 9 demand × stock combinations plus 2 competitor rules. It covers every input, so the fallback never triggers.

To change the model, edit the `terms` of `DEMAND`, `STOCK` or `COMPETITOR_GAP`, or the rule lists in `fuzzy_engine.py`. Rule outputs are named shapes in `OUTPUT_TERMS`.

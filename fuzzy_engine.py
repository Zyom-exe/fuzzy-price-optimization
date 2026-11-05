"""
fuzzy_engine.py
================
A small, readable fuzzy price optimization engine written from scratch
(only NumPy is needed). It follows the same 6 steps as the presentation:

    1. Collect data        -> validate the inputs
    2. Turn into scores    -> fuzzification (membership functions)
    3. Apply the rules     -> IF-THEN rules, AND = min, OR = max
    4. Combine             -> weight each rule by its strength
    5. Get one number      -> defuzzification (Sugeno average or Mamdani centroid)
    6. Safety check        -> guardrails: price floor/ceiling, margin floor, change cap

All numbers in the default "case-study" set-up are the hypothetical teaching
values from the deck: ₹4,000 headphones, demand 8/10, stock 30%  ->  ₹4,387.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

MODEL_VERSION = "1.0.0"
INF = math.inf


# ---------------------------------------------------------------------------
# 1. Membership functions
# ---------------------------------------------------------------------------
def trapmf(x, a: float, b: float, c: float, d: float):
    """Trapezoid membership function.

    Rises from 0 at `a` to 1 at `b`, stays 1 until `c`, falls to 0 at `d`.
    Use -inf for a "left shoulder" (full membership on the far left) and
    +inf for a "right shoulder". A triangle is simply b == c.
    Works on single numbers and on NumPy arrays.
    """
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rise = np.where(x <= a, 0.0, np.where(x >= b, 1.0, (x - a) / (b - a)))
        fall = np.where(x >= d, 0.0, np.where(x <= c, 1.0, (d - x) / (d - c)))
    return np.clip(np.minimum(rise, fall), 0.0, 1.0)


def trimf(x, a: float, b: float, c: float):
    """Triangle membership function (a special trapezoid)."""
    return trapmf(x, a, b, b, c)


# ---------------------------------------------------------------------------
# 2. Fuzzy variables (inputs) and their word labels ("terms")
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FuzzyVariable:
    name: str                       # key used in the inputs dict
    label: str                      # human readable name
    unit: str
    low: float                      # valid range of the input
    high: float
    terms: Dict[str, Tuple[float, float, float, float]]  # term -> trapezoid (a, b, c, d)

    def membership(self, term: str, x):
        a, b, c, d = self.terms[term]
        return trapmf(x, a, b, c, d)

    def fuzzify(self, x: float) -> Dict[str, float]:
        """Return the score (0-1) of `x` for every word label."""
        return {t: float(self.membership(t, x)) for t in self.terms}

    def universe(self, n: int = 601) -> np.ndarray:
        return np.linspace(self.low, self.high, n)


# Demand score 0-10. High: 4.5 -> 9.5 ; Medium: triangle peaking at 4.5.
# At demand 8 these give High 0.7 and Medium 0.3 (the deck's values).
DEMAND = FuzzyVariable(
    "demand", "Demand score", "/10", 0, 10,
    {"Low": (-INF, -INF, 0, 4.5),
     "Medium": (0, 4.5, 4.5, 9.5),
     "High": (4.5, 9.5, INF, INF)},
)

# Stock as % of the normal stock level. Low: full up to 10%, gone at 60%.
# At stock 30% Low = 0.6 (the deck's value).
STOCK = FuzzyVariable(
    "stock", "Stock level", "%", 0, 100,
    {"Low": (-INF, -INF, 10, 60),
     "Normal": (30, 60, 60, 90),
     "High": (60, 90, INF, INF)},
)

# Competitor gap = (our price - competitor price) / competitor price x 100.
# Positive = we are more expensive than the competitor.
COMPETITOR_GAP = FuzzyVariable(
    "competitor_gap", "Competitor gap", "%", -30, 30,
    {"Cheaper": (-INF, -INF, -10, 0),
     "Similar": (-5, 0, 0, 5),
     "Pricier": (0, 10, INF, INF)},
)

# Teaching example from slides 4-5 (₹0-₹1,000): not used for pricing decisions,
# but handy for showing what "partly fair, partly expensive" means.
PRICE_LABELS = FuzzyVariable(
    "price", "Price", "₹", 0, 1000,
    {"Cheap": (-INF, -INF, 200, 450),
     "Fair": (250, 500, 500, 750),
     "Expensive": (550, 800, INF, INF)},
)

ALL_VARIABLES = {v.name: v for v in (DEMAND, STOCK, COMPETITOR_GAP)}

# Output: recommended price change in %. Each term is a triangle.
# Sugeno uses the peak value; Mamdani uses the whole shape.
OUTPUT_TERMS: Dict[str, Tuple[float, float, float, float]] = {
    "Large decrease": (-14, -12, -12, -10),
    "Small decrease": (-7, -5, -5, -3),
    "No change": (-2, 0, 0, 2),
    "Small increase": (3, 5, 5, 7),
    "Large increase": (10, 12, 12, 14),
}
OUTPUT_RANGE = (-15.0, 15.0)


def output_peak(term: str) -> float:
    a, b, c, d = OUTPUT_TERMS[term]
    return (b + c) / 2.0


# ---------------------------------------------------------------------------
# 3. Rules
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Rule:
    name: str
    conditions: Tuple[Tuple[str, str], ...]   # ((variable, term), ...)
    output_term: str
    connector: str = "AND"                     # "AND" (min) or "OR" (max)

    @property
    def output_value(self) -> float:
        return output_peak(self.output_term)

    def text(self) -> str:
        parts = [f"{ALL_VARIABLES[v].label.split()[0].lower() if v in ALL_VARIABLES else v} is {t}"
                 for v, t in self.conditions]
        joined = f" {self.connector} ".join(parts)
        return f"IF {joined} THEN {self.output_term} ({self.output_value:+g}%)"


def _r(name, output, *conds, connector="AND"):
    return Rule(name, tuple(conds), output, connector)


# The two rules used in the presentation (slides 7-9).
CASE_STUDY_RULES: List[Rule] = [
    _r("A", "Large increase", ("demand", "High"), ("stock", "Low")),
    _r("B", "Small increase", ("demand", "Medium"), ("stock", "Low")),
]

# A fuller rule base: every demand x stock combination (3 x 3 = 9 rules)
# plus two competitor rules. Use this to explore a more realistic system.
EXTENDED_RULES: List[Rule] = CASE_STUDY_RULES + [
    _r("C", "No change", ("demand", "Low"), ("stock", "Low")),
    _r("D", "Small increase", ("demand", "High"), ("stock", "Normal")),
    _r("E", "No change", ("demand", "Medium"), ("stock", "Normal")),
    _r("F", "Small decrease", ("demand", "Low"), ("stock", "Normal")),
    _r("G", "No change", ("demand", "High"), ("stock", "High")),
    _r("H", "Small decrease", ("demand", "Medium"), ("stock", "High")),
    _r("I", "Large decrease", ("demand", "Low"), ("stock", "High")),
    _r("J", "Small decrease", ("competitor_gap", "Pricier")),
    _r("K", "Small increase", ("competitor_gap", "Cheaper")),
]

RULE_SETS = {"case-study": CASE_STUDY_RULES, "extended": EXTENDED_RULES}


# ---------------------------------------------------------------------------
# 6. Guardrails (safety checks)
# ---------------------------------------------------------------------------
@dataclass
class Guardrails:
    min_price: Optional[float] = None       # absolute floor, e.g. ₹3,200
    max_price: Optional[float] = None       # absolute ceiling, e.g. ₹4,800
    max_change_pct: Optional[float] = None  # largest allowed move vs reference, e.g. 15
    unit_cost: Optional[float] = None       # used for the margin floor
    min_margin: Optional[float] = None      # e.g. 0.25 for a 25% gross margin

    def margin_floor(self) -> Optional[float]:
        """Lowest price that still earns the minimum gross margin: cost / (1 - margin)."""
        if self.unit_cost is None or self.min_margin is None:
            return None
        if not 0 <= self.min_margin < 1:
            raise ValueError("min_margin must be between 0 and 1 (e.g. 0.25)")
        return self.unit_cost / (1.0 - self.min_margin)

    def apply(self, reference: float, raw_price: float) -> Tuple[float, List[str], List[str]]:
        """Return (price, rules applied, warnings)."""
        applied, warnings = [], []
        price = raw_price

        if self.max_change_pct is not None:
            hi = reference * (1 + self.max_change_pct / 100)
            lo = reference * (1 - self.max_change_pct / 100)
            if price > hi:
                applied.append(f"change cap: limited to +{self.max_change_pct:g}%")
                price = hi
            elif price < lo:
                applied.append(f"change cap: limited to -{self.max_change_pct:g}%")
                price = lo

        floors = []
        if self.min_price is not None:
            floors.append(("minimum price", self.min_price))
        mf = self.margin_floor()
        if mf is not None:
            floors.append(("margin floor", mf))
        floor_name, floor = max(floors, key=lambda f: f[1]) if floors else (None, None)
        ceiling = self.max_price

        if floor is not None and ceiling is not None and floor > ceiling:
            warnings.append(f"{floor_name} ₹{floor:,.0f} is above the maximum price ₹{ceiling:,.0f}: "
                            "kept the reference price, needs manual review")
            return reference, applied + ["conflict: kept reference price"], warnings

        if floor is not None and price < floor:
            applied.append(f"{floor_name}: raised to ₹{floor:,.2f}")
            price = floor
        if ceiling is not None and price > ceiling:
            applied.append(f"maximum price: lowered to ₹{ceiling:,.2f}")
            price = ceiling
        return price, applied, warnings


# ---------------------------------------------------------------------------
# The decision record (what gets logged for every price)
# ---------------------------------------------------------------------------
@dataclass
class RuleActivation:
    rule: str
    text: str
    strength: float
    output_term: str
    output_pct: float


@dataclass
class PricingDecision:
    product: str
    method: str
    rule_set: str
    reference_price: float
    inputs: Dict[str, float]
    memberships: Dict[str, Dict[str, float]] = field(default_factory=dict)
    activations: List[RuleActivation] = field(default_factory=list)
    change_pct: Optional[float] = None          # unconstrained fuzzy output
    raw_price: Optional[float] = None
    final_price: float = 0.0
    guardrails_applied: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    fallback_reason: Optional[str] = None
    model_version: str = MODEL_VERSION
    timestamp: str = field(default_factory=lambda: _dt.datetime.now().isoformat(timespec="seconds"))

    @property
    def final_change_pct(self) -> float:
        return (self.final_price / self.reference_price - 1) * 100

    def to_dict(self) -> dict:
        d = asdict(self)
        d["final_change_pct"] = round(self.final_change_pct, 4)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    # Human-friendly explanation, following the 6 steps of the deck ---------
    def explain(self) -> str:
        method_name = {"sugeno": "Sugeno weighted average",
                       "mamdani": "Mamdani centroid (balance point)"}[self.method]
        L = []
        L.append(f"FUZZY PRICING DECISION  ·  {self.rule_set} rules  ·  {method_name}")
        L.append(f"Product: {self.product}    Reference price: ₹{self.reference_price:,.0f}")
        L.append("")
        L.append("Step 1  Collect data")
        L.append("        " + ",  ".join(
            f"{ALL_VARIABLES[k].label.lower()} = {v:g}{ALL_VARIABLES[k].unit}" for k, v in self.inputs.items()))
        if self.fallback_reason and not self.memberships:
            L.append(f"\nFALLBACK  {self.fallback_reason}")
            L.append(f"RESULT    ₹{self.final_price:,.0f}  (reference price kept)")
            return "\n".join(L)

        L.append("Step 2  Turn into scores (fuzzification)")
        for var, scores in self.memberships.items():
            s = "  |  ".join(f"{t} {m:.2f}" for t, m in scores.items())
            L.append(f"        {ALL_VARIABLES[var].label:<15} {s}")

        L.append("Step 3  Apply the rules   (AND = smaller score, OR = bigger score)")
        for a in self.activations:
            mark = "  <- fires" if a.strength > 0 else ""
            L.append(f"        Rule {a.rule}: {a.text}")
            L.append(f"                strength {a.strength:.2f}{mark}")

        if self.fallback_reason:
            L.append(f"\nFALLBACK  {self.fallback_reason}")
            L.append(f"RESULT    ₹{self.final_price:,.0f}  (reference price kept)")
            return "\n".join(L)

        fired = [a for a in self.activations if a.strength > 0]
        L.append("Step 4-5  Combine and get one number")
        if self.method == "sugeno":
            num = " + ".join(f"{a.strength:.2f}×{a.output_pct:g}" for a in fired)
            den = " + ".join(f"{a.strength:.2f}" for a in fired)
            L.append(f"        ({num}) ÷ ({den}) = {self.change_pct:+.2f}%")
        else:
            L.append("        Each output shape is cut at its rule's strength, the shapes are")
            L.append(f"        merged, and their balance point (centroid) is {self.change_pct:+.2f}%")
        L.append(f"        Raw price = ₹{self.reference_price:,.0f} × (1 {self.change_pct:+.4f}/100) "
                 f"= ₹{self.raw_price:,.2f}")

        L.append("Step 6  Safety checks")
        if self.guardrails_applied:
            for g in self.guardrails_applied:
                L.append(f"        - {g}")
        else:
            L.append("        All limits passed, no changes needed")
        for w in self.warnings:
            L.append(f"        WARNING: {w}")

        L.append("")
        L.append(f"RESULT  ₹{self.final_price:,.0f}   ({self.final_change_pct:+.2f}% vs reference)")
        return "\n".join(L)


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------
class FuzzyPricingEngine:
    def __init__(self,
                 rules: Sequence[Rule] = CASE_STUDY_RULES,
                 rule_set_name: str = "case-study",
                 guardrails: Optional[Guardrails] = None,
                 variables: Optional[Dict[str, FuzzyVariable]] = None,
                 min_firing: float = 1e-6,
                 resolution: float = 0.001):
        self.rules = list(rules)
        self.rule_set_name = rule_set_name
        self.guardrails = guardrails or Guardrails()
        self.variables = variables or ALL_VARIABLES
        self.min_firing = min_firing
        self.resolution = resolution
        self.required_inputs = sorted({v for r in self.rules for v, _ in r.conditions},
                                      key=list(self.variables).index)

    @classmethod
    def from_rule_set(cls, name: str, **kwargs) -> "FuzzyPricingEngine":
        return cls(RULE_SETS[name], rule_set_name=name, **kwargs)

    # Step 1 -----------------------------------------------------------------
    def validate(self, inputs: Dict[str, float]) -> Optional[str]:
        for name in self.required_inputs:
            v = inputs.get(name)
            var = self.variables[name]
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return f"missing input '{name}'"
            if not var.low <= v <= var.high:
                return f"input '{name}' = {v} is outside its valid range {var.low}–{var.high}"
        return None

    # Step 2 -----------------------------------------------------------------
    def fuzzify(self, inputs: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        return {name: self.variables[name].fuzzify(inputs[name]) for name in self.required_inputs}

    # Step 3 -----------------------------------------------------------------
    def fire_rules(self, memberships) -> List[RuleActivation]:
        acts = []
        for rule in self.rules:
            degrees = [memberships[v][t] for v, t in rule.conditions]
            strength = min(degrees) if rule.connector == "AND" else max(degrees)
            acts.append(RuleActivation(rule.name, rule.text(), round(strength, 10),
                                       rule.output_term, rule.output_value))
        return acts

    # Steps 4-5 --------------------------------------------------------------
    @staticmethod
    def sugeno(acts: Sequence[RuleActivation]) -> float:
        """Zero-order Sugeno: weighted average of the rules' constant outputs."""
        w = np.array([a.strength for a in acts])
        z = np.array([a.output_pct for a in acts])
        return float((w * z).sum() / w.sum())

    def aggregate(self, acts: Sequence[RuleActivation]):
        """Mamdani: cut (clip) every output shape at its rule strength, merge with max."""
        x = np.arange(OUTPUT_RANGE[0], OUTPUT_RANGE[1] + self.resolution, self.resolution)
        mu = np.zeros_like(x)
        for a in acts:
            if a.strength > 0:
                a_, b_, c_, d_ = OUTPUT_TERMS[a.output_term]
                mu = np.maximum(mu, np.minimum(a.strength, trapmf(x, a_, b_, c_, d_)))
        return x, mu

    def mamdani(self, acts: Sequence[RuleActivation]) -> float:
        x, mu = self.aggregate(acts)
        return float((x * mu).sum() / mu.sum())

    # Everything together ----------------------------------------------------
    def evaluate(self, inputs: Dict[str, float], reference_price: float,
                 method: str = "sugeno", product: str = "product") -> PricingDecision:
        method = method.lower()
        if method not in ("sugeno", "mamdani"):
            raise ValueError("method must be 'sugeno' or 'mamdani'")
        used = {k: float(inputs[k]) for k in self.required_inputs if inputs.get(k) is not None}
        dec = PricingDecision(product=product, method=method, rule_set=self.rule_set_name,
                              reference_price=float(reference_price), inputs=used,
                              final_price=float(round_rupee(reference_price)))

        problem = self.validate(inputs)                                   # step 1
        if problem:
            dec.fallback_reason = f"invalid data ({problem}): kept the reference price"
            return dec

        dec.memberships = self.fuzzify(inputs)                            # step 2
        dec.activations = self.fire_rules(dec.memberships)                # step 3

        if sum(a.strength for a in dec.activations) < self.min_firing:
            dec.fallback_reason = "no rule applies to these inputs: kept the reference price"
            return dec

        fired = [a for a in dec.activations if a.strength > 0]           # steps 4-5
        dec.change_pct = self.sugeno(fired) if method == "sugeno" else self.mamdani(fired)
        dec.raw_price = reference_price * (1 + dec.change_pct / 100)

        price, applied, warnings = self.guardrails.apply(reference_price, dec.raw_price)  # step 6
        dec.guardrails_applied, dec.warnings = applied, warnings
        dec.final_price = float(round_rupee(price))
        return dec


def round_rupee(x: float) -> int:
    """Round to the nearest rupee (0.5 always rounds up)."""
    return int(math.floor(x + 0.5))


# ---------------------------------------------------------------------------
# Simple non-fuzzy baselines, for comparison (slides 3 and 10)
# ---------------------------------------------------------------------------
def hard_stock_rule_change(stock: float) -> float:
    """IF stock < 20% THEN +10% ELSE 0%."""
    return 10.0 if stock < 20 else 0.0


def hard_demand_rule_change(demand: float) -> float:
    """IF demand >= 8 THEN +12% ELSE +5%."""
    return 12.0 if demand >= 8 else 5.0


def cost_plus_price(unit_cost: float, markup: float) -> float:
    """Price = cost × (1 + markup), e.g. markup 0.25 for 25%."""
    return unit_cost * (1 + markup)


def log_decision(decision: PricingDecision, path: str) -> None:
    """Append the full decision record as one JSON line (the audit trail)."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(decision.to_json() + "\n")

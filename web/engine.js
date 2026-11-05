/*
 * engine.js — JavaScript port of fuzzy_engine.py for the web app.
 * Same membership functions, rules, Sugeno/Mamdani inference and guardrails,
 * so it gives the same prices as the Python model (checked by parity_check.py).
 * Works in the browser (window.FuzzyEngine) and in Node (module.exports).
 */
(function (root) {
  "use strict";

  const MODEL_VERSION = "1.0.0";
  const INF = Infinity;

  // 1. Membership functions ---------------------------------------------------
  function trapmf(x, a, b, c, d) {
    const rise = x <= a ? 0 : x >= b ? 1 : (x - a) / (b - a);
    const fall = x >= d ? 0 : x <= c ? 1 : (d - x) / (d - c);
    return Math.min(1, Math.max(0, Math.min(rise, fall)));
  }

  // 2. Fuzzy variables ----------------------------------------------------------
  function variable(name, label, unit, low, high, terms) {
    return {
      name, label, unit, low, high, terms,
      membership(term, x) { const [a, b, c, d] = terms[term]; return trapmf(x, a, b, c, d); },
      fuzzify(x) {
        const out = {};
        for (const t of Object.keys(terms)) out[t] = this.membership(t, x);
        return out;
      },
    };
  }

  const DEMAND = variable("demand", "Demand score", "/10", 0, 10, {
    Low: [-INF, -INF, 0, 4.5], Medium: [0, 4.5, 4.5, 9.5], High: [4.5, 9.5, INF, INF],
  });
  const STOCK = variable("stock", "Stock level", "%", 0, 100, {
    Low: [-INF, -INF, 10, 60], Normal: [30, 60, 60, 90], High: [60, 90, INF, INF],
  });
  const COMPETITOR_GAP = variable("competitor_gap", "Competitor gap", "%", -30, 30, {
    Cheaper: [-INF, -INF, -10, 0], Similar: [-5, 0, 0, 5], Pricier: [0, 10, INF, INF],
  });
  const PRICE_LABELS = variable("price", "Price", "₹", 0, 1000, {
    Cheap: [-INF, -INF, 200, 450], Fair: [250, 500, 500, 750], Expensive: [550, 800, INF, INF],
  });
  const ALL_VARIABLES = { demand: DEMAND, stock: STOCK, competitor_gap: COMPETITOR_GAP };
  const VARIABLE_ORDER = Object.keys(ALL_VARIABLES);

  const OUTPUT_TERMS = {
    "Large decrease": [-14, -12, -12, -10],
    "Small decrease": [-7, -5, -5, -3],
    "No change": [-2, 0, 0, 2],
    "Small increase": [3, 5, 5, 7],
    "Large increase": [10, 12, 12, 14],
  };
  const OUTPUT_RANGE = [-15, 15];

  function outputPeak(term) { const [, b, c] = OUTPUT_TERMS[term]; return (b + c) / 2; }

  // 3. Rules ---------------------------------------------------------------------
  function rule(name, output, conditions, connector = "AND") {
    return {
      name, output_term: output, conditions, connector,
      get output_value() { return outputPeak(output); },
      text() {
        const parts = conditions.map(([v, t]) =>
          `${ALL_VARIABLES[v] ? ALL_VARIABLES[v].label.split(" ")[0].toLowerCase() : v} is ${t}`);
        const pct = outputPeak(output);
        return `IF ${parts.join(` ${connector} `)} THEN ${output} (${pct >= 0 ? "+" : ""}${pct}%)`;
      },
    };
  }

  const CASE_STUDY_RULES = [
    rule("A", "Large increase", [["demand", "High"], ["stock", "Low"]]),
    rule("B", "Small increase", [["demand", "Medium"], ["stock", "Low"]]),
  ];
  const EXTENDED_RULES = CASE_STUDY_RULES.concat([
    rule("C", "No change", [["demand", "Low"], ["stock", "Low"]]),
    rule("D", "Small increase", [["demand", "High"], ["stock", "Normal"]]),
    rule("E", "No change", [["demand", "Medium"], ["stock", "Normal"]]),
    rule("F", "Small decrease", [["demand", "Low"], ["stock", "Normal"]]),
    rule("G", "No change", [["demand", "High"], ["stock", "High"]]),
    rule("H", "Small decrease", [["demand", "Medium"], ["stock", "High"]]),
    rule("I", "Large decrease", [["demand", "Low"], ["stock", "High"]]),
    rule("J", "Small decrease", [["competitor_gap", "Pricier"]]),
    rule("K", "Small increase", [["competitor_gap", "Cheaper"]]),
  ]);
  const RULE_SETS = { "case-study": CASE_STUDY_RULES, extended: EXTENDED_RULES };

  // 6. Guardrails ----------------------------------------------------------------
  const fmt0 = (v) => v.toLocaleString("en-IN", { maximumFractionDigits: 0 });
  const fmt2 = (v) => v.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const g = (v) => String(+v.toPrecision(6));

  class Guardrails {
    constructor({ min_price = null, max_price = null, max_change_pct = null,
                  unit_cost = null, min_margin = null } = {}) {
      Object.assign(this, { min_price, max_price, max_change_pct, unit_cost, min_margin });
    }

    marginFloor() {
      if (this.unit_cost == null || this.min_margin == null) return null;
      if (!(this.min_margin >= 0 && this.min_margin < 1)) {
        throw new Error("min_margin must be between 0 and 1 (e.g. 0.25)");
      }
      return this.unit_cost / (1 - this.min_margin);
    }

    apply(reference, rawPrice) {
      const applied = [], warnings = [];
      let price = rawPrice;

      if (this.max_change_pct != null) {
        const hi = reference * (1 + this.max_change_pct / 100);
        const lo = reference * (1 - this.max_change_pct / 100);
        if (price > hi) { applied.push(`change cap: limited to +${g(this.max_change_pct)}%`); price = hi; }
        else if (price < lo) { applied.push(`change cap: limited to -${g(this.max_change_pct)}%`); price = lo; }
      }

      const floors = [];
      if (this.min_price != null) floors.push(["minimum price", this.min_price]);
      const mf = this.marginFloor();
      if (mf != null) floors.push(["margin floor", mf]);
      let floorName = null, floor = null;
      for (const [n, v] of floors) if (floor == null || v > floor) { floorName = n; floor = v; }
      const ceiling = this.max_price;

      if (floor != null && ceiling != null && floor > ceiling) {
        warnings.push(`${floorName} ₹${fmt0(floor)} is above the maximum price ₹${fmt0(ceiling)}: ` +
                      "kept the reference price, needs manual review");
        return [reference, applied.concat(["conflict: kept reference price"]), warnings];
      }
      if (floor != null && price < floor) { applied.push(`${floorName}: raised to ₹${fmt2(floor)}`); price = floor; }
      if (ceiling != null && price > ceiling) { applied.push(`maximum price: lowered to ₹${fmt2(ceiling)}`); price = ceiling; }
      return [price, applied, warnings];
    }
  }

  function roundRupee(x) { return Math.floor(x + 0.5); }

  // NumPy's ndarray.sum(): first element + pairwise sum of the rest. Matching its
  // rounding keeps prices identical to Python when a result sits on a ₹0.50 tie.
  function pairwise(a, start, n) {
    if (n < 8) {
      let res = 0;
      for (let i = start; i < start + n; i++) res += a[i];
      return res;
    }
    if (n <= 128) {
      const r = [a[start], a[start + 1], a[start + 2], a[start + 3],
                 a[start + 4], a[start + 5], a[start + 6], a[start + 7]];
      let i;
      for (i = 8; i < n - (n % 8); i += 8) for (let j = 0; j < 8; j++) r[j] += a[start + i + j];
      let res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
      for (; i < n; i++) res += a[start + i];
      return res;
    }
    let n2 = Math.floor(n / 2);
    n2 -= n2 % 8;
    return pairwise(a, start, n2) + pairwise(a, start + n2, n - n2);
  }

  function npSum(a) { return a.length ? a[0] + pairwise(a, 1, a.length - 1) : 0; }

  // The engine -------------------------------------------------------------------
  class FuzzyPricingEngine {
    constructor({ rules = CASE_STUDY_RULES, rule_set_name = "case-study", guardrails = null,
                  min_firing = 1e-6, resolution = 0.001 } = {}) {
      this.rules = rules.slice();
      this.rule_set_name = rule_set_name;
      this.guardrails = guardrails || new Guardrails();
      this.min_firing = min_firing;
      this.resolution = resolution;
      const used = new Set();
      for (const r of this.rules) for (const [v] of r.conditions) used.add(v);
      this.required_inputs = VARIABLE_ORDER.filter((v) => used.has(v));
    }

    static fromRuleSet(name, opts = {}) {
      return new FuzzyPricingEngine({ ...opts, rules: RULE_SETS[name], rule_set_name: name });
    }

    // Step 1
    validate(inputs) {
      for (const name of this.required_inputs) {
        const v = inputs[name], vr = ALL_VARIABLES[name];
        if (v == null || Number.isNaN(v)) return `missing input '${name}'`;
        if (!(vr.low <= v && v <= vr.high)) {
          return `input '${name}' = ${v} is outside its valid range ${vr.low}–${vr.high}`;
        }
      }
      return null;
    }

    // Step 2
    fuzzify(inputs) {
      const out = {};
      for (const name of this.required_inputs) out[name] = ALL_VARIABLES[name].fuzzify(inputs[name]);
      return out;
    }

    // Step 3
    fireRules(memberships) {
      return this.rules.map((r) => {
        const degrees = r.conditions.map(([v, t]) => memberships[v][t]);
        const s = r.connector === "AND" ? Math.min(...degrees) : Math.max(...degrees);
        return { rule: r.name, text: r.text(), strength: Math.round(s * 1e10) / 1e10,
                 output_term: r.output_term, output_pct: r.output_value };
      });
    }

    // Steps 4-5
    static sugeno(acts) {
      return npSum(acts.map((a) => a.strength * a.output_pct)) / npSum(acts.map((a) => a.strength));
    }

    /** Mamdani: clip each output shape at its rule strength and merge with max.
     *  Same grid as np.arange(-15, 15 + res, res). */
    aggregate(acts) {
      const [lo, hi] = OUTPUT_RANGE, res = this.resolution;
      const step = (lo + res) - lo;
      const n = Math.ceil(((hi + res) - lo) / res);
      const x = new Float64Array(n), mu = new Float64Array(n);
      for (let i = 0; i < n; i++) x[i] = lo + i * step;
      for (const a of acts) {
        if (!(a.strength > 0)) continue;
        const [p, q, r, s] = OUTPUT_TERMS[a.output_term];
        for (let i = 0; i < n; i++) {
          const m = Math.min(a.strength, trapmf(x[i], p, q, r, s));
          if (m > mu[i]) mu[i] = m;
        }
      }
      return [x, mu];
    }

    mamdani(acts) {
      const [x, mu] = this.aggregate(acts);
      const xmu = new Float64Array(x.length);
      for (let i = 0; i < x.length; i++) xmu[i] = x[i] * mu[i];
      return npSum(xmu) / npSum(mu);
    }

    evaluate(inputs, referencePrice, method = "sugeno", product = "product") {
      method = method.toLowerCase();
      if (method !== "sugeno" && method !== "mamdani") throw new Error("method must be 'sugeno' or 'mamdani'");
      const used = {};
      for (const k of this.required_inputs) if (inputs[k] != null) used[k] = +inputs[k];
      const dec = {
        product, method, rule_set: this.rule_set_name, reference_price: +referencePrice, inputs: used,
        memberships: {}, activations: [], change_pct: null, raw_price: null,
        final_price: roundRupee(referencePrice), guardrails_applied: [], warnings: [],
        fallback_reason: null, model_version: MODEL_VERSION,
        timestamp: new Date().toISOString().slice(0, 19),
      };

      const problem = this.validate(inputs);
      if (problem) { dec.fallback_reason = `invalid data (${problem}): kept the reference price`; return finish(dec); }

      dec.memberships = this.fuzzify(inputs);
      dec.activations = this.fireRules(dec.memberships);
      if (dec.activations.reduce((s, a) => s + a.strength, 0) < this.min_firing) {
        dec.fallback_reason = "no rule applies to these inputs: kept the reference price";
        return finish(dec);
      }

      const fired = dec.activations.filter((a) => a.strength > 0);
      dec.change_pct = method === "sugeno" ? FuzzyPricingEngine.sugeno(fired) : this.mamdani(fired);
      dec.raw_price = referencePrice * (1 + dec.change_pct / 100);

      const [price, applied, warnings] = this.guardrails.apply(referencePrice, dec.raw_price);
      dec.guardrails_applied = applied;
      dec.warnings = warnings;
      dec.final_price = roundRupee(price);
      return finish(dec);
    }
  }

  function finish(dec) {
    dec.final_change_pct = Math.round((dec.final_price / dec.reference_price - 1) * 100 * 1e4) / 1e4;
    return dec;
  }

  // Non-fuzzy baselines ------------------------------------------------------------
  const hardStockRuleChange = (stock) => (stock < 20 ? 10 : 0);
  const hardDemandRuleChange = (demand) => (demand >= 8 ? 12 : 5);
  const costPlusPrice = (cost, markup) => cost * (1 + markup);

  const api = {
    MODEL_VERSION, trapmf, DEMAND, STOCK, COMPETITOR_GAP, PRICE_LABELS, ALL_VARIABLES,
    OUTPUT_TERMS, OUTPUT_RANGE, outputPeak, CASE_STUDY_RULES, EXTENDED_RULES, RULE_SETS,
    Guardrails, FuzzyPricingEngine, roundRupee, hardStockRuleChange, hardDemandRuleChange, costPlusPrice,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.FuzzyEngine = api;
})(typeof globalThis !== "undefined" ? globalThis : this);

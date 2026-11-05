"""
visualize.py
============
Charts for the presentation and an interactive slider app, built with matplotlib.

    save_all_charts("charts")   -> writes 6 PNG charts
    run_app()                   -> opens a window with sliders for demand, stock,
                                   competitor gap and reference price
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider

from fuzzy_engine import (COMPETITOR_GAP, DEMAND, OUTPUT_RANGE, OUTPUT_TERMS, PRICE_LABELS, STOCK,
                          FuzzyPricingEngine, Guardrails, hard_demand_rule_change,
                          hard_stock_rule_change, output_peak, trapmf)

BLUE, GREEN, ORANGE = "#3B7DD8", "#1E9E63", "#E8781F"
GREY, NAVY, PURPLE, ROSE = "#7A8699", "#0E1A2B", "#7C6FD8", "#B03A68"
TERM_STYLES = [(BLUE, "-"), (GREEN, "--"), (ORANGE, ":")]   # colour + line style (not colour alone)


def default_guardrails(reference: float) -> Guardrails:
    """Same proportions as the deck: ₹3,200–₹4,800 around ₹4,000, max ±15% per change."""
    return Guardrails(min_price=0.8 * reference, max_price=1.2 * reference, max_change_pct=15)


def _style(ax, title=None):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
    if title:
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=NAVY)


def _plot_variable(ax, var, value=None, faded=False):
    x = var.universe()
    for (term, _), (col, ls) in zip(var.terms.items(), TERM_STYLES):
        y = var.membership(term, x)
        ax.plot(x, y, color=col, ls=ls, lw=2.5, label=term, alpha=0.35 if faded else 1)
        ax.fill_between(x, y, color=col, alpha=0.06 if faded else 0.10)
    ax.set_ylim(-0.03, 1.12)
    ax.set_xlim(var.low, var.high)
    ax.set_ylabel("score")
    ax.legend(loc="upper center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.02))
    if value is not None:
        ax.axvline(value, color=NAVY, ls="--", lw=1.5)


# ---------------------------------------------------------------------------
# Static charts
# ---------------------------------------------------------------------------
def chart_price_labels(ax, marker=600):
    _plot_variable(ax, PRICE_LABELS, marker)
    s = PRICE_LABELS.fuzzify(marker)
    for (term, m), (col, _) in zip(s.items(), TERM_STYLES):
        if m > 0:
            ax.plot(marker, m, "o", ms=10, color=col, mec=NAVY)
            ax.annotate(f"{term} {m:.1f}", (marker, m), xytext=(10, 6), textcoords="offset points",
                        fontsize=10, fontweight="bold", color=col)
    ax.set_xlabel("price (₹)")
    _style(ax, f"One price, several labels: ₹{marker} is partly Fair and partly Expensive")


def chart_input_memberships(axes, inputs: Dict[str, float]):
    for ax, var in zip(axes, (DEMAND, STOCK, COMPETITOR_GAP)):
        v = inputs.get(var.name)
        _plot_variable(ax, var, v)
        scores = "  ".join(f"{t} {m:.2f}" for t, m in var.fuzzify(v).items()) if v is not None else ""
        ax.set_xlabel(f"{var.label} ({var.unit})")
        _style(ax, f"{var.label} = {v:g}  →  {scores}" if v is not None else var.label)


def chart_cliff(ax):
    s = np.linspace(0, 60, 1201)
    hard = np.array([hard_stock_rule_change(v) for v in s])
    fuzzy = 10 * STOCK.membership("Low", s)
    ax.plot(s, hard, color=GREY, lw=3, label="Hard rule: IF stock < 20% THEN +10%", drawstyle="steps-post")
    ax.plot(s, fuzzy, color=ORANGE, lw=3, label="Fuzzy rule: +10% × (how 'Low' stock is)")
    ax.axvspan(19, 21, color=ORANGE, alpha=0.12)
    ax.annotate("19.9% → +10%\n20.1% → +0%", (20, 5), xytext=(26, 6.5), fontsize=10, color=GREY,
                fontweight="bold", arrowprops=dict(arrowstyle="->", color=GREY))
    ax.set_xlabel("stock level (%)")
    ax.set_ylabel("price increase (%)")
    ax.legend(frameon=False, fontsize=9)
    _style(ax, "Hard rules jump; fuzzy rules slope")


def chart_price_vs_demand(ax, stock=30, reference=4000):
    eng = FuzzyPricingEngine(guardrails=default_guardrails(reference))
    d = np.linspace(5, 9.5, 181)
    sug = [eng.evaluate({"demand": v, "stock": stock}, reference).raw_price for v in d]
    mam = [eng.evaluate({"demand": v, "stock": stock}, reference, "mamdani").raw_price for v in d]
    hard = [reference * (1 + hard_demand_rule_change(v) / 100) for v in d]
    ax.plot(d, hard, color=GREY, lw=3, drawstyle="steps-post", label="Hard rule (IF demand ≥ 8 THEN +12% ELSE +5%)")
    ax.plot(d, sug, color=ORANGE, lw=3, label="Fuzzy – Sugeno")
    ax.plot(d, mam, color=ROSE, lw=2, ls="--", label="Fuzzy – Mamdani")
    for v in (7.9, 8.0):
        p = eng.evaluate({"demand": v, "stock": stock}, reference).final_price
        ax.plot(v, p, "o", color=ORANGE, mec=NAVY, ms=8)
    ax.annotate("7.9 → ₹4,383\n8.0 → ₹4,387", (7.95, 4385), xytext=(6.1, 4430), fontsize=10,
                fontweight="bold", color=NAVY, arrowprops=dict(arrowstyle="->", color=NAVY))
    ax.set_xlabel("demand score (/10)")
    ax.set_ylabel("price (₹)")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _style(ax, f"Price vs demand (stock fixed at {stock}%)")


def chart_defuzzification(ax, inputs=None, reference=4000):
    inputs = inputs or {"demand": 8, "stock": 30}
    eng = FuzzyPricingEngine(guardrails=default_guardrails(reference))
    dec_s = eng.evaluate(inputs, reference, "sugeno")
    dec_m = eng.evaluate(inputs, reference, "mamdani")
    x, mu = eng.aggregate(dec_m.activations)
    for term, (a, b, c, d) in OUTPUT_TERMS.items():
        ax.plot(x, trapmf(x, a, b, c, d), color=GREY, ls=":", lw=1)
    ax.fill_between(x, mu, color=ROSE, alpha=0.55, label="shapes cut at rule strength")
    ax.axvline(dec_m.change_pct, color=NAVY, lw=2.5, label=f"Mamdani balance point {dec_m.change_pct:+.2f}%")
    ax.axvline(dec_s.change_pct, color=BLUE, lw=2, ls="--", label=f"Sugeno weighted average {dec_s.change_pct:+.2f}%")
    ax.plot([dec_m.change_pct], [-0.05], marker="^", ms=18, color=NAVY, clip_on=False)
    ax.set_xlim(0, 16)
    ax.set_ylim(-0.08, 1.1)
    ax.set_xlabel("price change (%)")
    ax.set_ylabel("score")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _style(ax, "Defuzzification: blending the rules into one number")


def chart_heatmap(ax, fig, reference=4000, gap=0.0):
    eng = FuzzyPricingEngine.from_rule_set("extended", guardrails=default_guardrails(reference))
    d = np.linspace(0, 10, 81)
    s = np.linspace(0, 100, 81)
    Z = np.array([[eng.evaluate({"demand": dv, "stock": sv, "competitor_gap": gap}, reference).final_price
                   for dv in d] for sv in s])
    im = ax.imshow(Z, origin="lower", aspect="auto", extent=[0, 10, 0, 100], cmap="RdYlBu_r")
    cs = ax.contour(d, s, Z, levels=8, colors="k", linewidths=0.5, alpha=0.5)
    ax.clabel(cs, fmt=lambda v: f"₹{v:,.0f}", fontsize=8)
    fig.colorbar(im, ax=ax, label="recommended price (₹)")
    ax.set_xlabel("demand score (/10)")
    ax.set_ylabel("stock level (%)")
    _style(ax, "Extended rules: price for every demand × stock combination")
    ax.grid(False)


def save_all_charts(out_dir="charts", dpi=150) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []

    def save(fig, name):
        p = os.path.join(out_dir, name)
        fig.tight_layout()
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        paths.append(p)

    fig, ax = plt.subplots(figsize=(9, 4.5)); chart_price_labels(ax); save(fig, "1_price_labels.png")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    chart_input_memberships(axes, {"demand": 8, "stock": 30, "competitor_gap": 0})
    save(fig, "2_input_memberships.png")
    fig, ax = plt.subplots(figsize=(9, 4.5)); chart_cliff(ax); save(fig, "3_hard_vs_fuzzy.png")
    fig, ax = plt.subplots(figsize=(9, 4.8)); chart_defuzzification(ax); save(fig, "4_defuzzification.png")
    fig, ax = plt.subplots(figsize=(9, 4.8)); chart_price_vs_demand(ax); save(fig, "5_price_vs_demand.png")
    fig, ax = plt.subplots(figsize=(9, 5.5)); chart_heatmap(ax, fig); save(fig, "6_price_heatmap.png")
    return paths


# ---------------------------------------------------------------------------
# Interactive app
# ---------------------------------------------------------------------------
def build_app():
    fig = plt.figure(figsize=(15, 8.6))
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title("Fuzzy Price Optimization – live model")
    fig.suptitle("Fuzzy Price Optimization — move the sliders", x=0.05, ha="left",
                 fontsize=15, fontweight="bold", color=NAVY)

    ax_in = [fig.add_axes([0.05, 0.71, 0.38, 0.17]),
             fig.add_axes([0.05, 0.46, 0.38, 0.17]),
             fig.add_axes([0.05, 0.21, 0.38, 0.17])]
    ax_out = fig.add_axes([0.52, 0.50, 0.45, 0.38])
    ax_txt = fig.add_axes([0.52, 0.17, 0.45, 0.27]); ax_txt.axis("off")

    s_dem = Slider(fig.add_axes([0.14, 0.105, 0.28, 0.025]), "Demand (/10)", 0, 10, valinit=8, valstep=0.1)
    s_stk = Slider(fig.add_axes([0.14, 0.065, 0.28, 0.025]), "Stock (%)", 0, 100, valinit=30, valstep=1)
    s_gap = Slider(fig.add_axes([0.14, 0.025, 0.28, 0.025]), "Competitor gap (%)", -30, 30, valinit=0, valstep=1)
    s_ref = Slider(fig.add_axes([0.58, 0.065, 0.20, 0.025]), "Reference ₹", 1000, 10000, valinit=4000, valstep=100)
    r_met = RadioButtons(fig.add_axes([0.82, 0.02, 0.07, 0.11]), ("Sugeno", "Mamdani"))
    r_set = RadioButtons(fig.add_axes([0.90, 0.02, 0.09, 0.11]), ("case-study", "extended"))

    def update(_=None):
        rules = r_set.value_selected
        method = r_met.value_selected.lower()
        ref = s_ref.val
        inputs = {"demand": s_dem.val, "stock": s_stk.val, "competitor_gap": s_gap.val}
        eng = FuzzyPricingEngine.from_rule_set(rules, guardrails=default_guardrails(ref))
        dec = eng.evaluate(inputs, ref, method, "headphones")

        for ax, var in zip(ax_in, (DEMAND, STOCK, COMPETITOR_GAP)):
            ax.cla()
            used = var.name in eng.required_inputs
            _plot_variable(ax, var, inputs[var.name], faded=not used)
            sc = "  ".join(f"{t} {m:.2f}" for t, m in var.fuzzify(inputs[var.name]).items())
            _style(ax, f"{var.label}: {sc}" if used else f"{var.label}: not used by case-study rules")

        ax_out.cla()
        x = np.linspace(*OUTPUT_RANGE, 1201)
        for term, (a, b, c, d) in OUTPUT_TERMS.items():
            ax_out.plot(x, trapmf(x, a, b, c, d), color=GREY, ls=":", lw=1)
            ax_out.text(output_peak(term), 1.03, term, ha="center", fontsize=8, color=GREY)
        if dec.fallback_reason:
            ax_out.text(0, 0.5, "No rule applies →\nreference price kept", ha="center", va="center",
                        fontsize=14, color=ROSE, fontweight="bold")
            title = f"₹{dec.final_price:,.0f}  (no change)"
        else:
            fired = [a for a in dec.activations if a.strength > 0]
            if method == "mamdani":
                xx, mu = eng.aggregate(fired)
                ax_out.fill_between(xx, mu, color=ROSE, alpha=0.55)
            else:
                for a in fired:
                    ax_out.vlines(a.output_pct, 0, a.strength, color=PURPLE, lw=6)
                    ax_out.text(a.output_pct, a.strength + 0.03, f"{a.strength:.2f}", ha="center",
                                fontsize=9, color=PURPLE, fontweight="bold")
            ax_out.axvline(dec.change_pct, color=NAVY, lw=2.5)
            title = f"₹{dec.final_price:,.0f}   ({dec.final_change_pct:+.2f}% vs ₹{ref:,.0f})"
        ax_out.set_xlim(*OUTPUT_RANGE); ax_out.set_ylim(0, 1.12)
        ax_out.set_xlabel("recommended price change (%)")
        _style(ax_out, f"Recommended price: {title}")

        ax_txt.cla(); ax_txt.axis("off")
        lines = [f"{a.rule}: {a.text}  →  {a.strength:.2f}" for a in dec.activations if a.strength > 0]
        lines = lines[:7] or ["(no rule fires)"]
        if dec.change_pct is not None:
            lines.append(f"Fuzzy output {dec.change_pct:+.2f}%  →  raw ₹{dec.raw_price:,.2f}")
        lines += [f"Safety: {g}" for g in dec.guardrails_applied] or ["Safety: all limits passed"]
        if dec.fallback_reason:
            lines.append(dec.fallback_reason)
        ax_txt.text(0, 1, "Rules that fire (strength):\n" + "\n".join(lines), va="top", fontsize=9.5,
                    family="monospace", color=NAVY)
        fig.canvas.draw_idle()

    for w in (s_dem, s_stk, s_gap, s_ref):
        w.on_changed(update)
    r_met.on_clicked(update)
    r_set.on_clicked(update)
    update()
    fig._widgets = (s_dem, s_stk, s_gap, s_ref, r_met, r_set)   # keep references alive
    return fig


def run_app():
    build_app()
    plt.show()

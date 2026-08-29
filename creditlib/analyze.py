"""One-line analysis API.

The rest of the library is deliberately explicit: you build a discount curve,
build a schedule, call the pricer. That is right for a pricing engine and
wrong for doing analysis quickly. This module is the thin layer on top.

    from creditlib.analyze import analyze, compare

    r = analyze(85, name="Citigroup")                     # single 5Y quote
    r = analyze({1: 55, 3: 82, 5: 105, 7: 118, 10: 130})  # full term structure
    print(r)                                               # formatted summary
    print(r.report())                                      # markdown, paste into a memo

    c = compare({"JPM": 58, "BAC": 63, "USB": 67, "WFC": 72, "C": 85})
    print(c)

Spreads are given in BASIS POINTS here, not decimals. That is the one place
this module deviates from the rest of the library, and it is deliberate: you
think in bp, and `analyze(85)` should not silently mean 850,000bp.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Mapping

import numpy as np
import pandas as pd

from .bootstrap import BootstrapError, bootstrap_survival_curve, spread_bounds
from .cds import cs01, ir_dv01, jump_to_default, par_spread, price_cds, rec01
from .curves import CreditCurve
from .recovery import RecoveryModel, IssuerProfile, recovery_map
from .discount_curve import DiscountCurve
from .schedule import flat_schedule, make_cds_schedule

# Illustrative USD OIS-style curve. NOT a real SOFR curve -- see `default_curve`.
_DEFAULT_ZEROS = {
    0.25: 0.0430, 0.5: 0.0421, 1.0: 0.0402, 2.0: 0.0381,
    3.0: 0.0374, 5.0: 0.0372, 7.0: 0.0378, 10.0: 0.0390,
}


def default_curve() -> DiscountCurve:
    """A plausible USD discount curve, for when you don't have a real one.

    Using this is fine for CDS work: a 600bp parallel shift moves implied 5Y
    default probability by about 5bp of probability, because the two legs
    largely offset. It is NOT fine to describe the output as calibrated to
    market rates. Pass your own curve via `discount=` when it matters.
    """
    return DiscountCurve.from_zero_rates(list(_DEFAULT_ZEROS), list(_DEFAULT_ZEROS.values()))


def _normalise(spreads) -> dict[float, float]:
    """Accept a scalar 5Y quote or a {tenor: bp} mapping. Returns decimals."""
    if isinstance(spreads, (int, float, np.floating)):
        return {5.0: float(spreads) * 1e-4}
    if isinstance(spreads, Mapping):
        return {float(k): float(v) * 1e-4 for k, v in sorted(spreads.items())}
    raise TypeError("spreads must be a number (5Y bp) or a {tenor: bp} mapping")


@dataclass
class Analysis:
    """Everything you'd want about one name, computed once."""

    name: str
    quotes_bp: dict
    recovery: float
    curve: CreditCurve
    discount: DiscountCurve
    table: pd.DataFrame
    risk: dict
    trade_date: date
    notional: float
    coupon: float
    single_quote: bool
    max_rel_error: float
    bounds: pd.DataFrame = field(default=None)

    # ------------------------------------------------------------- accessors

    def pd_at(self, years: float) -> float:
        """Cumulative default probability by `years`."""
        return 1.0 - float(self.curve.survival(years))

    def survival_at(self, years: float) -> float:
        return float(self.curve.survival(years))

    def par_spread_at(self, years: float) -> float:
        """Par spread in bp for any maturity, including ones never quoted."""
        return par_spread(self.discount, self.curve, self.recovery,
                          flat_schedule(years, frequency=4)) * 1e4

    # ---------------------------------------------------------------- output

    def __str__(self) -> str:
        head = f"{self.name or 'Unnamed'}   R = {self.recovery:.0%}"
        body = self.table.to_string(index=False, float_format=lambda x: f"{x:,.4f}")
        risk = "\n".join(f"  {k:<22} {v:>16,.2f}" for k, v in self.risk.items())
        warn = ""
        if self.single_quote:
            warn = ("\n  NOTE: one quote -> one hazard -> the curve is FLAT by construction.\n"
                    "        Values away from 5Y are the model assumption, not market data.")
        return (f"{head}\n{'-' * len(head)}\n{body}\n\n"
                f"Risk on {self.notional:,.0f} at {self.coupon*1e4:.0f}bp coupon:\n{risk}\n"
                f"\n  max repricing error: {self.max_rel_error:.2e} relative{warn}")

    def report(self) -> str:
        """Markdown summary, ready to paste into a memo."""
        rows = "\n".join(
            f"| {r.tenor:.0f}Y | {r.quoted_bp:.0f} | {r.hazard_bp:.1f} | "
            f"{r.fwd_spread_bp:.1f} | {r.cum_PD:.2%} |"
            for r in self.table.itertuples()
        )
        caveat = (
            "\n> **Single quote.** One 5Y spread gives one hazard rate, so the curve is flat "
            "by construction. Points away from 5Y are the interpolation assumption, not "
            "market information.\n"
            if self.single_quote else ""
        )
        return f"""### {self.name or 'Credit'} — CDS-implied default probabilities

Recovery assumption: **{self.recovery:.0%}** (ISDA convention for senior unsecured; an
assumption, not a quote — spreads identify expected loss h(1−R), not h alone).

| Tenor | Quoted (bp) | Fwd hazard (bp) | Fwd spread (bp) | Cumulative PD |
|---|---|---|---|---|
{rows}
{caveat}
**Risk** on {self.notional:,.0f} notional at a {self.coupon*1e4:.0f}bp coupon:
CS01 {self.risk['CS01 (+1bp spread)']:,.0f} · IR DV01 {self.risk['IR DV01 (+1bp rates)']:,.0f} ·
JTD {self.risk['JTD (default now)']:,.0f} ({self.risk['JTD (default now)']/self.risk['CS01 (+1bp spread)']:,.0f}× CS01)

*Risk-neutral, not actuarial: these run roughly 8–12× historical default rates for
comparable ratings. Quotes repriced to {self.max_rel_error:.1e} relative.*
"""

    def plot(self, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        a1, a2 = ax
        t = np.linspace(0, max(self.curve.times[-1], 10), 400)
        a1.plot(t, (1 - np.asarray(self.curve.survival(t))) * 100, lw=2, color="#1F4E79")
        a1.plot(self.curve.times, (1 - np.asarray(self.curve.survival(self.curve.times))) * 100,
                "o", color="#1F4E79", ms=7, mec="white", mew=1.3, zorder=5, label="quoted")
        a1.set_xlabel("Years"); a1.set_ylabel("Cumulative default probability (%)")
        a1.set_title(f"{self.name} — implied PD", loc="left")
        a1.grid(alpha=.25); a1.legend(frameon=False)

        edges = np.concatenate(([0], self.curve.times))
        vals = np.concatenate((self.curve.rates[:1], self.curve.rates)) * 1e4
        a2.step(edges, vals, where="pre", lw=2, color="#9A4B00")
        a2.set_xlabel("Years"); a2.set_ylabel("Forward hazard (bp)")
        a2.set_title("Piecewise-flat hazards — FORWARD, not average", loc="left")
        a2.grid(alpha=.25)
        return ax


def analyze(
    spreads,
    name: str = "",
    recovery: float = 0.40,
    discount: DiscountCurve | None = None,
    trade_date: date | None = None,
    notional: float = 10_000_000,
    coupon: float = 0.0100,
) -> Analysis:
    """Bootstrap a name and compute everything, in one call.

    `spreads` is a 5Y quote in bp, or a {tenor_years: bp} mapping.
    """
    quotes = _normalise(spreads)
    single = len(quotes) == 1
    discount = discount or default_curve()
    trade_date = trade_date or date.today()

    boot = bootstrap_survival_curve(quotes, discount, recovery, trade_date=trade_date)

    table = pd.DataFrame({
        "tenor": boot.tenors,
        "quoted_bp": boot.quoted_spreads * 1e4,
        "repriced_bp": boot.repriced_spreads * 1e4,
        "hazard_bp": boot.hazards * 1e4,
        "fwd_spread_bp": boot.hazards * (1 - recovery) * 1e4,
        "survival": boot.survival,
        "cum_PD": 1 - boot.survival,
    })

    longest = float(boot.tenors[-1])
    sched = make_cds_schedule(trade_date, tenor_years=longest)
    risk = {
        "CS01 (+1bp spread)": cs01(discount, boot.curve, coupon, recovery, sched, notional),
        "IR DV01 (+1bp rates)": ir_dv01(discount, boot.curve, coupon, recovery, sched, notional),
        "Rec01 (+1pt recovery)": rec01(discount, boot.curve, coupon, recovery, sched, notional),
        "JTD (default now)": jump_to_default(discount, boot.curve, coupon, recovery,
                                             sched, notional),
    }

    bounds = None
    if not single:
        rows = []
        for k, t in enumerate(boot.tenors):
            lo, hi = spread_bounds(quotes, discount, recovery, tenor_index=k,
                                   trade_date=trade_date)
            rows.append({"tenor": t, "floor_bp": lo * 1e4, "quoted_bp": quotes[t] * 1e4,
                         "ceiling_bp": hi * 1e4})
        bounds = pd.DataFrame(rows)

    return Analysis(
        name=name, quotes_bp={k: v * 1e4 for k, v in quotes.items()}, recovery=recovery,
        curve=boot.curve, discount=discount, table=table, risk=risk, trade_date=trade_date,
        notional=notional, coupon=coupon, single_quote=single,
        max_rel_error=boot.max_rel_error, bounds=bounds,
    )


@dataclass
class Comparison:
    """Cross-section of names on a common discount curve."""

    table: pd.DataFrame
    analyses: dict
    recovery: object
    per_name_recovery: bool = False

    def __str__(self) -> str:
        body = self.table.to_string(index=False, float_format=lambda x: f"{x:,.4f}")
        if self.per_name_recovery:
            note = ("\n\nName-specific recovery. The ranking is NOT invariant here — it is "
                    "driven jointly by\nspread and assumed loss given default, so the ordering "
                    "carries a view, not just a quote.")
        else:
            note = (f"\n\nR = {self.recovery:.0%} uniform. Ranking is INVARIANT to this "
                    "assumption\n(h = s/(1-R) is a monotone transform of the spread); only the "
                    "levels move.")
        return body + note

    def report(self) -> str:
        rows = "\n".join(
            f"| {r.name} | {r.spread_5y_bp:.0f} | {r.hazard_bp:.1f} | {r.PD_5y:.2%} |"
            for r in self.table.itertuples()
        )
        lo, hi = self.table.PD_5y.min(), self.table.PD_5y.max()
        return f"""### CDS-implied credit ranking

| Name | 5Y spread (bp) | Implied hazard (bp) | 5Y cumulative PD |
|---|---|---|---|
{rows}

Recovery **{self.recovery:.0%}** applied uniformly. Because h = s/(1−R) is a monotone
transform of the spread, **the ranking is invariant to that assumption** — only the levels
move. Reordering would require name-specific recovery differentials.

Dispersion: {self.table.spread_5y_bp.max() - self.table.spread_5y_bp.min():.0f}bp of spread,
{(hi - lo) * 100:.2f}pp of 5Y default probability. Risk-neutral, not actuarial.
"""

    def plot(self, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4.6))
        t = np.linspace(0, 10, 400)
        colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(self.analyses)))
        for (nm, a), c in zip(self.analyses.items(), colors):
            pdv = (1 - np.asarray(a.curve.survival(t))) * 100
            style = dict(color=c, lw=2)
            if a.single_quote:
                ax.plot(t, pdv, ls="--", alpha=.85, **style)
                ax.plot(t[t <= 5], pdv[t <= 5], label=f"{nm} ({a.quotes_bp[5.0]:.0f}bp)", **style)
                ax.plot(5, (1 - a.survival_at(5)) * 100, "o", color=c, ms=7,
                        mec="white", mew=1.3, zorder=5)
            else:
                ax.plot(t, pdv, label=nm, **style)
        ax.set_xlabel("Years"); ax.set_ylabel("Cumulative default probability (%)")
        ax.set_title(f"Implied PD, R = {self.recovery:.0%}", loc="left")
        ax.grid(alpha=.25); ax.legend(frameon=False, fontsize=9)
        return ax


def compare(
    names: Mapping[str, object],
    recovery: float | Mapping[str, float] = 0.40,
    discount: DiscountCurve | None = None,
    trade_date: date | None = None,
) -> Comparison:
    """Bootstrap several names on a common curve and rank them.

        compare({"JPM": 58, "BAC": 63, "C": 85})
        compare({"JPM": {1: 40, 5: 58, 10: 75}, "C": {1: 62, 5: 85, 10: 104}})

    `recovery` may be a single float applied to every name, or a
    {name: recovery} mapping for name-specific assumptions.

    The distinction is not cosmetic. With a uniform recovery, h = s/(1-R) is a
    monotone transform of the spread, so the implied default-probability
    ranking is INVARIANT to the assumption. With name-specific recovery that
    invariance breaks, and two issuers quoted at the same spread can imply
    materially different default probabilities. See creditlib.recovery.
    """
    discount = discount or default_curve()
    trade_date = trade_date or date.today()
    per_name = isinstance(recovery, Mapping)
    if per_name:
        missing = set(names) - set(recovery)
        if missing:
            raise ValueError(f"no recovery supplied for: {sorted(missing)}")

    analyses, rows = {}, []
    for nm, sp in names.items():
        R = float(recovery[nm]) if per_name else float(recovery)
        a = analyze(sp, name=nm, recovery=R, discount=discount, trade_date=trade_date)
        analyses[nm] = a
        five = a.quotes_bp.get(5.0, np.nan)
        rows.append({
            "name": nm,
            "spread_5y_bp": five,
            "recovery": R,
            "hazard_bp": float(a.curve.hazard(5.0)) * 1e4,
            "PD_1y": a.pd_at(1.0),
            "PD_5y": a.pd_at(5.0),
            "PD_10y": a.pd_at(10.0),
        })

    table = pd.DataFrame(rows).sort_values("PD_5y").reset_index(drop=True)
    if not per_name:
        table = table.drop(columns=["recovery"])
    return Comparison(table=table, analyses=analyses, recovery=recovery,
                      per_name_recovery=per_name)


def recovery_sensitivity(
    names: Mapping[str, object],
    recoveries=(0.40, 0.25, 0.10),
    discount: DiscountCurve | None = None,
) -> pd.DataFrame:
    """Implied 5Y PD across recovery assumptions. Levels move; ranking does not."""
    discount = discount or default_curve()
    return pd.DataFrame(
        {f"R={r:.0%}": {nm: analyze(sp, recovery=r, discount=discount).pd_at(5.0)
                        for nm, sp in names.items()} for r in recoveries}
    )

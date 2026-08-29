"""creditlib quickstart -- run me first.

    python quickstart.py

Walks the four things the engine does, in the order you'd actually use them:
  1. Build a discount curve
  2. Price a CDS off a known credit curve
  3. Bootstrap a credit curve from market quotes
  4. Compute risk

Every number printed here is reproduced by the test suite.
"""

from datetime import date

import numpy as np

from creditlib import (
    CreditCurve,
    DiscountCurve,
    bootstrap_survival_curve,
    cs01,
    flat_schedule,
    ir_dv01,
    jump_to_default,
    make_cds_schedule,
    par_spread,
    price_cds,
    rec01,
    spread_bounds,
)

BAR = "=" * 74


# ---------------------------------------------------------------- 1. curves
print(BAR)
print("1. DISCOUNT CURVE  --  log-linear on discount factors")
print(BAR)

discount = DiscountCurve.from_zero_rates(
    times=[0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0],
    zeros=[0.0430, 0.0421, 0.0402, 0.0381, 0.0374, 0.0372, 0.0378, 0.0390],
    compounding="continuous",
)
print(discount)
print(f"\n  D(5)                    = {discount.discount_factor(5.0):.8f}")
print(f"  zero rate 5y (cc)       = {discount.zero_rate(5.0):.4%}")
print(f"  zero rate 5y (annual)   = {discount.zero_rate(5.0, 'annual'):.4%}")
print(f"  forward 3y-5y (cc)      = {discount.forward_rate(3.0, 5.0):.4%}")
print(f"  forward 3y-5y (simple)  = {discount.forward_rate(3.0, 5.0, 'simple'):.4%}")
print("\n  Same number, four conventions. Always state compounding AND day count.")


# ------------------------------------------------------------ 2. price a CDS
print("\n" + BAR)
print("2. PRICE A CDS  --  given a known credit curve")
print(BAR)

credit = CreditCurve(
    times=[0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
    rates=[0.0080, 0.0095, 0.0140, 0.0185, 0.0210, 0.0230],   # hazard rates
)
schedule = make_cds_schedule(trade_date=date(2026, 8, 17), tenor_years=5.0)
print(f"  schedule: {schedule.n} periods, {schedule.accrual_start[0]} -> {schedule.maturity}")

result = price_cds(
    discount_curve=discount,
    survival_curve=credit,
    spread=0.0100,          # 100bp standard IG coupon
    recovery=0.40,
    schedule=schedule,
    notional=10_000_000,
    protection_buyer=True,
)
print()
print(result)


# -------------------------------------------------------------- 3. bootstrap
print("\n" + BAR)
print("3. BOOTSTRAP  --  market quotes -> hazard curve")
print(BAR)

quotes = {1.0: 0.0055, 3.0: 0.0082, 5.0: 0.0105, 7.0: 0.0118, 10.0: 0.0130}
boot = bootstrap_survival_curve(
    spreads=quotes,
    discount_curve=discount,
    recovery=0.40,
    trade_date=date(2026, 8, 17),
)
print(boot)

print("\n  Hazards are FORWARD quantities. The 5y quote is 105bp, but the")
print("  marginal intensity over 3y-5y implies a "
      f"{boot.implied_forward_spread(2)*1e4:.1f}bp forward spread.")

print("\n  Feasible quote window at each tenor, given the shorter quotes:")
for k, t in enumerate(quotes):
    lo, hi = spread_bounds(quotes, discount, 0.40, tenor_index=k,
                           trade_date=date(2026, 8, 17))
    print(f"    {t:4.0f}Y  [{lo*1e4:8.2f}, {hi*1e4:10.2f}] bp   quoted {quotes[t]*1e4:7.2f}")

# Price an off-tenor CDS the market never quoted -- the point of a curve.
s4 = par_spread(discount, boot.curve, 0.40, flat_schedule(4.0, frequency=4))
print(f"\n  Interpolated 4y par spread (never quoted): {s4*1e4:.2f}bp")


# ------------------------------------------------------------------ 4. risk
print("\n" + BAR)
print("4. RISK  --  four different things, do not confuse them")
print(BAR)

N, COUPON, REC = 10_000_000, 0.0100, 0.40
sched5 = make_cds_schedule(date(2026, 8, 17), tenor_years=5.0)

c = cs01(discount, boot.curve, COUPON, REC, sched5, N, method="reprice")
i = ir_dv01(discount, boot.curve, COUPON, REC, sched5, N)
j = jump_to_default(discount, boot.curve, COUPON, REC, sched5, N)
r = rec01(discount, boot.curve, COUPON, REC, sched5, N)

print(f"  CS01   (+1bp par spread)  {c:>15,.2f}")
print(f"  IR DV01(+1bp rates)       {i:>15,.2f}   {abs(i/c):>8.2%} of CS01")
print(f"  Rec01  (+1pt recovery)    {r:>15,.2f}")
print(f"  JTD    (default now)      {j:>15,.2f}   {j/c:>8,.0f}x CS01")
print("\n  A book can be flat CS01 and still be destroyed by one default.")
print("  That ratio is why desks limit both.")


# ------------------------------------------------------------ sanity anchors
print("\n" + BAR)
print("SANITY ANCHORS  --  things you should be able to verify on paper")
print(BAR)

# Credit triangle is EXACT with continuous premium, flat r and h.
h, R = 0.02, 0.40
flat_dc, flat_cc = DiscountCurve.flat(0.05), CreditCurve.flat_hazard(h)
print(f"  credit triangle h(1-R)                  = {h*(1-R)*1e4:.4f} bp")
for freq in (1, 4, 365):
    s = par_spread(flat_dc, flat_cc, R,
                   flat_schedule(5.0, frequency=freq, accrual_basis=365.0))
    print(f"  par spread, premium paid {freq:>3}x/year     = {s*1e4:.4f} bp")
print("  -> converges to the triangle exactly as frequency rises.")

print(f"\n  bootstrap max repricing error: {boot.max_rel_error:.2e} relative")
print("  -> n quotes, n knots: a square system, so the fit is exact.")

print("\n" + BAR)
print("Next: notebooks/bank_credit_analysis.ipynb")
print("Docs: docs/creditlib_documentation.pdf")
print(BAR)

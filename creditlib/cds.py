"""Single-name CDS valuation.

Both legs are expectations of discounted contingent payoffs under the
assumption that default and interest rates are INDEPENDENT:

    PV_prot = (1-R) * integral_0^T D(t) h(t) Q(t) dt
    PV_prem = s * [ sum_i delta_i D(t_i) Q(t_i)
                    + sum_i (delta_i / dt_i) integral (t - t_{i-1}) D h Q dt ]

That independence assumption is standard and is wrong in exactly the states
that matter -- in a crisis, defaults cluster while rates rally, so D(tau) is
high precisely when tau is small. Second-order for IG, material for distressed.

Exact integration
-----------------
On any interval where f and h are BOTH constant -- guaranteed on the merged
grid of discount-curve knots, credit-curve knots and coupon boundaries,
because Steps 1 and 2 both used piecewise-flat rates -- with c = f + h:

    integral_a^b D h Q dt = (h/c) [ D(a)Q(a) - D(b)Q(b) ]

    integral_a^b (t-p) D h Q dt
        = h D(a)Q(a) { [1 - e^{-cL}(1+cL)]/c^2 + (a-p)(1 - e^{-cL})/c },  L = b-a

So the only approximation in the leg PVs is the curve interpolation already
chosen upstream. No quadrature error is layered on top. `method="midpoint"`
is provided for comparison against QuantLib's MidPointCdsEngine and to show
what the usual grid approach costs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .curves import CreditCurve
from .discount_curve import DiscountCurve
from .schedule import CdsSchedule

# Requested API name. CreditCurve stays canonical.
SurvivalCurve = CreditCurve

_TINY = 1e-12
_BP = 1e-4


# --------------------------------------------------------------------- utils

def _merged_grid(a: float, b: float, dc: DiscountCurve, cc: CreditCurve) -> np.ndarray:
    """Knots of both curves inside [a, b], plus the endpoints.

    On each resulting sub-interval f and h are constant, which is what makes
    the closed-form integrals exact.
    """
    knots = np.concatenate([np.asarray(dc.times, float), np.asarray(cc.times, float)])
    knots = knots[(knots > a + _TINY) & (knots < b - _TINY)]
    return np.unique(np.concatenate([[a], knots, [b]]))


def _fh(dc: DiscountCurve, cc: CreditCurve, a: float, b: float) -> tuple[float, float]:
    """Constant f and h on (a, b], sampled strictly inside."""
    mid = 0.5 * (a + b)
    return float(dc.instantaneous_forward(mid)), float(cc.hazard(mid))


def _dq(dc: DiscountCurve, cc: CreditCurve, t: float) -> float:
    return float(dc.discount_factor(t)) * float(cc.survival(t))


# ----------------------------------------------------------------- integrals

def _integral_DhQ(dc: DiscountCurve, cc: CreditCurve, a: float, b: float) -> float:
    """integral_a^b D(t) h(t) Q(t) dt, exact under piecewise-flat f and h."""
    if b <= a:
        return 0.0
    total = 0.0
    grid = _merged_grid(a, b, dc, cc)
    for lo, hi in zip(grid[:-1], grid[1:]):
        f, h = _fh(dc, cc, lo, hi)
        if h <= 0.0:
            continue
        c = f + h
        dq_lo, dq_hi = _dq(dc, cc, lo), _dq(dc, cc, hi)
        if abs(c) < _TINY:
            total += h * (hi - lo) * dq_lo          # limit c -> 0
        else:
            total += (h / c) * (dq_lo - dq_hi)
    return total


def _integral_offset_DhQ(dc: DiscountCurve, cc: CreditCurve,
                         a: float, b: float, p: float) -> float:
    """integral_a^b (t - p) D(t) h(t) Q(t) dt, exact under piecewise-flat f, h."""
    if b <= a:
        return 0.0
    total = 0.0
    grid = _merged_grid(a, b, dc, cc)
    for lo, hi in zip(grid[:-1], grid[1:]):
        f, h = _fh(dc, cc, lo, hi)
        if h <= 0.0:
            continue
        c = f + h
        L = hi - lo
        dq_lo = _dq(dc, cc, lo)
        if abs(c) < _TINY:
            # integral_0^L (u + lo - p) du
            total += h * dq_lo * (0.5 * L * L + (lo - p) * L)
        else:
            e = np.exp(-c * L)
            first = (1.0 - e * (1.0 + c * L)) / (c * c)
            second = (lo - p) * (1.0 - e) / c
            total += h * dq_lo * (first + second)
    return total


# ---------------------------------------------------------------------- legs

def protection_leg_pv(
    dc: DiscountCurve,
    cc: CreditCurve,
    schedule: CdsSchedule,
    recovery: float,
    notional: float = 1.0,
    method: str = "exact",
) -> float:
    """PV of (1 - R) paid at default, if default occurs before maturity."""
    if not 0.0 <= recovery < 1.0:
        raise ValueError("recovery must be in [0, 1)")
    t0 = max(0.0, float(schedule.t_start[0]))
    T = float(schedule.maturity_time)
    if T <= t0:
        return 0.0

    if method == "exact":
        integral = _integral_DhQ(dc, cc, t0, T)
    elif method == "midpoint":
        # Default assumed to land at the midpoint of each coupon period. This
        # is what QuantLib's MidPointCdsEngine does.
        integral = 0.0
        for a, b in zip(schedule.t_start, schedule.t_end):
            a = max(a, t0)
            if b <= a:
                continue
            mid = 0.5 * (a + b)
            integral += float(dc.discount_factor(mid)) * (
                float(cc.survival(a)) - float(cc.survival(b))
            )
    else:
        raise ValueError("method must be 'exact' or 'midpoint'")

    return notional * (1.0 - recovery) * integral


def premium_leg_pv(
    dc: DiscountCurve,
    cc: CreditCurve,
    schedule: CdsSchedule,
    spread: float,
    notional: float = 1.0,
    include_accrual: bool = True,
    method: str = "exact",
) -> tuple[float, float]:
    """Return (coupon PV, accrual-on-default PV) for a unit spread times `spread`."""
    coupon_pv, accrual_pv = 0.0, 0.0
    t0 = max(0.0, float(schedule.t_start[0]))

    for a, b, dlt in zip(schedule.t_start, schedule.t_end, schedule.delta):
        if b <= t0:
            continue
        coupon_pv += dlt * _dq(dc, cc, float(b))

        if not include_accrual:
            continue
        lo = max(float(a), t0)
        span = float(b) - float(a)
        if span <= 0.0:
            continue
        # delta / span converts a curve-time offset into an accrual fraction.
        # Exact when the day count is proportional within the period, which
        # holds for ACT/360 and ACT/365F up to day-rounding.
        scale = dlt / span
        if method == "exact":
            accrual_pv += scale * _integral_offset_DhQ(dc, cc, lo, float(b), float(a))
        elif method == "midpoint":
            mid = 0.5 * (lo + float(b))
            accrual_pv += scale * (mid - float(a)) * float(dc.discount_factor(mid)) * (
                float(cc.survival(lo)) - float(cc.survival(b))
            )
        else:
            raise ValueError("method must be 'exact' or 'midpoint'")

    return notional * spread * coupon_pv, notional * spread * accrual_pv


def risky_annuity(
    dc: DiscountCurve,
    cc: CreditCurve,
    schedule: CdsSchedule,
    notional: float = 1.0,
    include_accrual: bool = True,
    method: str = "exact",
) -> float:
    """RPV01: PV of 1 unit of running spread. Independent of the spread itself."""
    c, a = premium_leg_pv(dc, cc, schedule, 1.0, notional, include_accrual, method)
    return c + a


# -------------------------------------------------------------------- result

@dataclass
class CdsPricing:
    notional: float
    spread: float
    recovery: float
    protection_buyer: bool
    protection_leg_pv: float
    coupon_pv: float
    accrual_pv: float
    premium_leg_pv: float
    risky_annuity: float
    risky_annuity_clean: float
    stream_pv: float
    mtm: float
    par_spread: float
    accrued: float
    upfront_points: float

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    def __str__(self) -> str:
        side = "buyer" if self.protection_buyer else "seller"
        return (
            f"CDS protection {side}  N={self.notional:,.0f}  "
            f"coupon={self.spread*1e4:.1f}bp  R={self.recovery:.0%}\n"
            f"  protection leg PV : {self.protection_leg_pv:>15,.2f}\n"
            f"  premium leg PV    : {self.premium_leg_pv:>15,.2f}"
            f"   (coupons {self.coupon_pv:,.2f} + accrual {self.accrual_pv:,.2f})\n"
            f"  RPV01 dirty/clean : {self.risky_annuity:>15,.6f}"
            f" / {self.risky_annuity_clean:.6f}\n"
            f"  cashflow stream PV: {self.stream_pv:>15,.2f}\n"
            f"  MTM               : {self.mtm:>15,.2f}\n"
            f"  par spread        : {self.par_spread*1e4:>15.4f} bp\n"
            f"  accrued (rebate)  : {self.accrued:>15,.2f}\n"
            f"  upfront points    : {self.upfront_points:>15.4%}"
        )


def price_cds(
    discount_curve: DiscountCurve,
    survival_curve: CreditCurve,
    spread: float,
    recovery: float,
    schedule: CdsSchedule,
    notional: float = 1.0,
    protection_buyer: bool = True,
    include_accrual: bool = True,
    method: str = "exact",
) -> CdsPricing:
    """Value a single-name CDS.

    Sign convention: MTM is from the perspective of the protection BUYER by
    default, i.e. long protection / short credit. The buyer receives (1-R) on
    default and pays the running coupon, so

        MTM_buyer = PV_protection - spread * RPV01

    Set protection_buyer=False to flip the sign.
    """
    prot = protection_leg_pv(discount_curve, survival_curve, schedule,
                             recovery, notional, method)
    coupon, accr = premium_leg_pv(discount_curve, survival_curve, schedule,
                                  spread, notional, include_accrual, method)
    prem = coupon + accr
    # RPV01 is quoted PER UNIT NOTIONAL, market convention: a ~4.7 "year"
    # figure, not a currency amount. Cash annuity = notional * this.
    annuity = risky_annuity(discount_curve, survival_curve, schedule,
                            1.0, include_accrual, method)
    clean_annuity = annuity - schedule.accrued_fraction()
    if clean_annuity <= 0.0:
        raise ValueError("clean annuity is non-positive; schedule is degenerate")

    accrued = notional * spread * schedule.accrued_fraction()

    # `stream_pv` is the PV of the modelled cashflows alone: protection received
    # less every future coupon paid, INCLUDING the pre-trade portion of the
    # seasoned first coupon. It is the right quantity for jump-to-default.
    stream_pv = prot - prem

    # `mtm` is the CLEAN market value -- the quoted upfront. It credits the buyer
    # with the accrued rebate they receive at settlement for that pre-trade stub:
    #
    #     mtm = stream_pv + accrued = prot - spread * clean_annuity * notional
    #
    # Using the dirty annuity here (an earlier bug) leaves mtm inconsistent with
    # a par spread computed on the clean annuity: the identity
    # mtm = (s_par - s) * A_clean * N then fails, and pricing AT the par spread
    # returns -accrued rather than zero. Invisible on synthetic schedules, where
    # accrued_fraction() == 0, which is why the regression tests below use a
    # seasoned IMM schedule.
    mtm = prot - spread * clean_annuity * notional
    if not protection_buyer:
        mtm = -mtm
        stream_pv = -stream_pv

    par = prot / (clean_annuity * notional)

    return CdsPricing(
        notional=notional,
        spread=spread,
        recovery=recovery,
        protection_buyer=protection_buyer,
        protection_leg_pv=prot,
        coupon_pv=coupon,
        accrual_pv=accr,
        premium_leg_pv=prem,
        risky_annuity=annuity,
        risky_annuity_clean=clean_annuity,
        stream_pv=stream_pv,
        mtm=mtm,
        par_spread=par,
        accrued=accrued,
        upfront_points=mtm / notional,
    )


def par_spread(
    discount_curve: DiscountCurve,
    survival_curve: CreditCurve,
    recovery: float,
    schedule: CdsSchedule,
    include_accrual: bool = True,
    method: str = "exact",
    clean: bool = True,
) -> float:
    """s_par = PV_protection / RPV01.

    A quotient, not a root-find: the protection leg does not involve s, and
    the risky annuity does not either. The numerical solve in this library
    happens in the bootstrap, where the unknown is the hazard rate.

    `clean=True` (default, and the market convention) nets the accrued rebate
    out of the annuity. On a seasoned schedule the annuity otherwise carries a
    stub of coupon for which no protection is provided -- accrual began at the
    previous IMM date but protection runs from today. On a 1Y contract that
    stub is 16% of the annuity, and ignoring it overstates the bootstrapped
    hazard by ~18bp. Verified against QuantLib: clean matches to 0.3bp at 1Y
    and under 0.07bp beyond, dirty is out by 18bp.

    Synthetic schedules from `flat_schedule` have zero accrued, so clean and
    dirty coincide there and the Step 4 hand-checks are unaffected.
    """
    prot = protection_leg_pv(discount_curve, survival_curve, schedule, recovery, 1.0, method)
    ann = risky_annuity(discount_curve, survival_curve, schedule, 1.0, include_accrual, method)
    if clean:
        ann -= schedule.accrued_fraction()
        if ann <= 0.0:
            raise ValueError("clean annuity is non-positive; schedule is degenerate")
    return prot / ann


# ------------------------------------------------------------- risk measures

def _shift_hazards(cc: CreditCurve, dh: float) -> CreditCurve:
    return CreditCurve(cc.times, np.maximum(cc.rates + dh, 0.0))


def cs01(
    discount_curve: DiscountCurve,
    survival_curve: CreditCurve,
    spread: float,
    recovery: float,
    schedule: CdsSchedule,
    notional: float = 1.0,
    protection_buyer: bool = True,
    bump_bp: float = 1.0,
    method: str = "reprice",
) -> float:
    """Credit spread 01: MTM change for a +`bump_bp` move in the par spread.

    method="analytic" : RPV01 * N * bump. First order only. Exact if the
        annuity were unchanged by the spread move, which it is not -- wider
        spreads lower Q, which lowers the annuity.

    method="reprice"  : find the PARALLEL HAZARD SHIFT that raises the par
        spread by `bump_bp`, then reprice. Honest for a single maturity.
        The genuine desk calculation bumps every quoted par spread and
        re-bootstraps the whole term structure; that arrives in Step 5, and
        for a single-tenor curve the two coincide.
    """
    if method == "analytic":
        ann = risky_annuity(discount_curve, survival_curve, schedule, 1.0)
        return ann * notional * bump_bp * _BP

    if method != "reprice":
        raise ValueError("method must be 'analytic' or 'reprice'")

    base_par = par_spread(discount_curve, survival_curve, recovery, schedule)
    target = base_par + bump_bp * _BP

    def gap(dh: float) -> float:
        return par_spread(discount_curve, _shift_hazards(survival_curve, dh),
                          recovery, schedule) - target

    hi = max(bump_bp * _BP / (1.0 - recovery) * 5.0, 1e-4)
    while gap(hi) < 0.0 and hi < 5.0:
        hi *= 2.0
    dh = brentq(gap, 0.0, hi, xtol=1e-16, rtol=1e-15)

    bumped = _shift_hazards(survival_curve, dh)
    base = price_cds(discount_curve, survival_curve, spread, recovery, schedule,
                     notional, protection_buyer)
    up = price_cds(discount_curve, bumped, spread, recovery, schedule,
                   notional, protection_buyer)
    return up.mtm - base.mtm


def ir_dv01(
    discount_curve: DiscountCurve,
    survival_curve: CreditCurve,
    spread: float,
    recovery: float,
    schedule: CdsSchedule,
    notional: float = 1.0,
    protection_buyer: bool = True,
    bump_bp: float = 1.0,
) -> float:
    """MTM change for a parallel +`bump_bp` shift in continuously compounded zeros.

    Small for a CDS: discounting both legs harder largely cancels. Reporting
    it alongside CS01 makes the relative magnitude visible, which is the point.
    """
    t = discount_curve.times
    z = discount_curve.zero_rate(np.asarray(t, float))
    bumped = DiscountCurve.from_zero_rates(t, z + bump_bp * _BP,
                                           require_monotone_df=False)
    base = price_cds(discount_curve, survival_curve, spread, recovery, schedule,
                     notional, protection_buyer)
    up = price_cds(bumped, survival_curve, spread, recovery, schedule,
                   notional, protection_buyer)
    return up.mtm - base.mtm


def rec01(
    discount_curve: DiscountCurve,
    survival_curve: CreditCurve,
    spread: float,
    recovery: float,
    schedule: CdsSchedule,
    notional: float = 1.0,
    protection_buyer: bool = True,
    bump: float = 0.01,
) -> float:
    """MTM change for a +`bump` (absolute) move in the recovery assumption."""
    base = price_cds(discount_curve, survival_curve, spread, recovery, schedule,
                     notional, protection_buyer)
    up = price_cds(discount_curve, survival_curve, spread, min(recovery + bump, 0.999),
                   schedule, notional, protection_buyer)
    return up.mtm - base.mtm


def jump_to_default(
    discount_curve: DiscountCurve,
    survival_curve: CreditCurve,
    spread: float,
    recovery: float,
    schedule: CdsSchedule,
    notional: float = 1.0,
    protection_buyer: bool = True,
) -> float:
    """Instantaneous P&L if the reference entity defaults right now.

        JTD_buyer = (1 - R) * N  -  accrued  -  MTM_buyer

    You receive the default payment, you owe the premium accrued since the
    last coupon date, and you give up the mark you were already carrying.
    Not a derivative -- a discontinuity. A position can be short CS01 and
    still lose badly on JTD, which is why desks limit both.
    """
    p = price_cds(discount_curve, survival_curve, spread, recovery, schedule,
                  notional, protection_buyer)
    payoff = notional * (1.0 - recovery) - p.accrued
    if not protection_buyer:
        payoff = -payoff
    # Against `stream_pv`, not `mtm`: the clean mark already credits the accrued
    # rebate, and on default that stub is owed rather than rebated. Using mtm
    # here would double-count it.
    return payoff - p.stream_pv

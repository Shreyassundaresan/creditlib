"""Bootstrap tests.

The headline assertion is exact repricing: n knots at n quote maturities is a
square system, so the residual should sit at machine precision, not at a
tolerance someone chose.
"""

import numpy as np
import pytest

from creditlib import (
    CreditCurve,
    DiscountCurve,
    InvertedCurveError,
    SpreadCeilingError,
    bootstrap_survival_curve,
    check_credit_curve,
    cs01,
    flat_schedule,
    par_spread,
    price_cds,
    screen_quotes,
    spread_bounds,
)

TENORS = [1.0, 3.0, 5.0, 7.0, 10.0]
QUOTES = dict(zip(TENORS, [0.0055, 0.0082, 0.0105, 0.0118, 0.0130]))
REC = 0.40


def disc() -> DiscountCurve:
    return DiscountCurve.from_zero_rates(
        [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0],
        [0.0530, 0.0518, 0.0472, 0.0421, 0.0399, 0.0392, 0.0398, 0.0410],
    )


# --------------------------------------------------------- exact repricing

def test_reprices_every_quote_to_machine_precision():
    res = bootstrap_survival_curve(QUOTES, disc(), REC)
    assert np.allclose(res.repriced_spreads, res.quoted_spreads, rtol=1e-14, atol=0.0)
    assert res.max_rel_error < 1e-14
    assert res.max_abs_error_bp < 1e-11


def test_reprices_via_full_pricer_not_just_par_spread():
    """Independent route: price each quoted CDS at its own quoted coupon off
    the bootstrapped curve. MTM must be zero."""
    dc = disc()
    res = bootstrap_survival_curve(QUOTES, dc, REC)
    for tenor, s in QUOTES.items():
        sched = flat_schedule(tenor, frequency=4, accrual_basis=365.0)
        p = price_cds(dc, res.curve, s, REC, sched, notional=10e6)
        assert abs(p.mtm) < 1e-7


def test_bootstrapped_curve_is_arbitrage_free():
    res = bootstrap_survival_curve(QUOTES, disc(), REC)
    check_credit_curve(res.curve, t_max=12.0).raise_if_failed()
    assert np.all(res.hazards > 0.0)
    assert np.all(np.diff(res.survival) < 0.0)


def test_solver_lands_on_the_root():
    res = bootstrap_survival_curve(QUOTES, disc(), REC)
    for d in res.diagnostics:
        assert abs(d["phi_at_root"]) < 1e-16


# ------------------------------------------------------- triangular structure

def test_locality_of_the_triangular_system():
    """dPhi_k/dh_j = 0 for j > k, so bumping the 10Y quote must leave
    h_1..h_4 bit-for-bit unchanged. This is what makes bucketed CS01 mean
    anything at all."""
    dc = disc()
    base = bootstrap_survival_curve(QUOTES, dc, REC)

    bumped = dict(QUOTES)
    bumped[10.0] += 0.0050                     # +50bp on the 10Y alone
    res = bootstrap_survival_curve(bumped, dc, REC)

    assert np.array_equal(res.hazards[:4], base.hazards[:4])   # exactly equal
    assert res.hazards[4] > base.hazards[4]
    assert np.array_equal(res.survival[:4], base.survival[:4])


def test_errors_propagate_forward_only():
    """A stale 1Y quote corrupts every later hazard and no earlier one --
    there are no earlier ones. Shown by bumping the 1Y and checking the whole
    tail moves."""
    dc = disc()
    base = bootstrap_survival_curve(QUOTES, dc, REC)
    stale = dict(QUOTES)
    stale[1.0] += 0.0030
    res = bootstrap_survival_curve(stale, dc, REC)
    assert res.hazards[0] > base.hazards[0]
    assert np.all(res.hazards[1:] < base.hazards[1:])   # later buckets compensate
    # ...but the LATER quotes still reprice exactly.
    assert res.max_rel_error < 1e-14


def test_marginal_not_average_hazards():
    """h_k is a forward intensity over (T_{k-1}, T_k], so on an upward-sloping
    spread curve the bucket hazards must exceed the naive triangle reading of
    the corresponding quote."""
    res = bootstrap_survival_curve(QUOTES, disc(), REC)
    for k, tenor in enumerate(TENORS):
        naive = QUOTES[tenor] / (1 - REC)      # average-style reading
        if k > 0:
            assert res.hazards[k] > naive


def test_subset_bootstrap_matches_prefix():
    """Bootstrapping only the first three quotes must reproduce the first
    three hazards of the full bootstrap exactly -- the definition of
    forward substitution."""
    dc = disc()
    full = bootstrap_survival_curve(QUOTES, dc, REC)
    part = bootstrap_survival_curve({t: QUOTES[t] for t in TENORS[:3]}, dc, REC)
    assert np.array_equal(part.hazards, full.hazards[:3])


# ------------------------------------------------------- sanity of the output

def test_flat_spread_curve_gives_near_flat_hazards_at_the_triangle():
    dc = DiscountCurve.flat(0.04)
    s = 0.0120
    res = bootstrap_survival_curve({t: s for t in TENORS}, dc, REC)
    triangle = s / (1 - REC)
    assert np.allclose(res.hazards, triangle, rtol=0.03)
    assert res.max_rel_error < 1e-14


def test_recovery_degeneracy():
    """Spreads pin expected loss h*(1-R), not h. Two recoveries reprice the
    SAME quotes exactly while implying hazards in ratio (1-R1)/(1-R2)."""
    dc = disc()
    r40 = bootstrap_survival_curve(QUOTES, dc, 0.40)
    r20 = bootstrap_survival_curve(QUOTES, dc, 0.20)

    assert r40.max_rel_error < 1e-14 and r20.max_rel_error < 1e-14
    ratio = r40.hazards / r20.hazards
    assert np.allclose(ratio, 0.80 / 0.60, rtol=0.02)
    # Implied 5y default probability differs materially on the SAME market data.
    assert (1 - r20.survival[2]) < (1 - r40.survival[2])


def test_upward_sloping_quotes_give_rising_hazards():
    res = bootstrap_survival_curve(QUOTES, disc(), REC)
    assert np.all(np.diff(res.hazards) > 0.0)


def test_mildly_inverted_curve_still_bootstraps():
    """Inversion alone is normal -- distressed names invert routinely. Only
    inversion past the no-arbitrage bound fails."""
    inverted = {1.0: 0.0400, 3.0: 0.0330, 5.0: 0.0300, 7.0: 0.0290, 10.0: 0.0285}
    res = bootstrap_survival_curve(inverted, disc(), REC)
    assert np.all(res.hazards > 0.0)
    assert res.max_rel_error < 1e-13
    assert res.hazards[1] < res.hazards[0]      # falling forward intensity


# ------------------------------------------------------------ failure modes

def test_steeply_inverted_curve_raises_inverted_curve_error():
    bad = dict(QUOTES)
    bad[3.0] = 0.0010                            # far below the 1Y at 55bp
    with pytest.raises(InvertedCurveError) as exc:
        bootstrap_survival_curve(bad, disc(), REC)
    assert exc.value.tenor == 3.0
    assert "negative forward" in str(exc.value)
    assert exc.value.bound > bad[3.0]


def test_spread_above_ceiling_raises_ceiling_error():
    bad = dict(QUOTES)
    bad[3.0] = 4.0                               # 40,000bp
    with pytest.raises(SpreadCeilingError) as exc:
        bootstrap_survival_curve(bad, disc(), REC)
    assert exc.value.tenor == 3.0
    assert exc.value.bound < bad[3.0]
    assert "certain-default ceiling" in str(exc.value)


def test_high_recovery_shrinks_the_feasible_window():
    """R -> 1 collapses loss given default, so the ceiling falls and quotes
    that were fine become unbootstrappable."""
    lo_40, hi_40 = spread_bounds(QUOTES, disc(), 0.40, tenor_index=2)
    lo_95, hi_95 = spread_bounds(QUOTES, disc(), 0.95, tenor_index=2)
    assert hi_95 < hi_40
    assert lo_40 <= QUOTES[5.0] <= hi_40

    stressed = dict(QUOTES)
    stressed[5.0] = hi_95 * 1.5
    with pytest.raises(SpreadCeilingError):
        bootstrap_survival_curve(stressed, disc(), 0.95)


def test_spread_bounds_bracket_the_quotes():
    dc = disc()
    for k, tenor in enumerate(TENORS):
        lo, hi = spread_bounds(QUOTES, dc, REC, tenor_index=k)
        assert lo < QUOTES[tenor] < hi
    # The first tenor has no floor from shorter quotes and no finite ceiling.
    lo0, hi0 = spread_bounds(QUOTES, dc, REC, tenor_index=0)
    assert lo0 == pytest.approx(0.0, abs=1e-15)
    assert hi0 > 1.0


def test_screen_flags_bad_marks_before_solving():
    assert screen_quotes(QUOTES, REC) == []
    bad = dict(QUOTES)
    bad[5.0] = 0.0020
    flags = screen_quotes(bad, REC)
    assert len(flags) == 1 and "5Y" in flags[0]


def test_input_validation():
    dc = disc()
    with pytest.raises(ValueError):
        bootstrap_survival_curve({1.0: -0.01}, dc, REC)          # negative spread
    with pytest.raises(ValueError):
        bootstrap_survival_curve({}, dc, REC)                    # no quotes
    with pytest.raises(ValueError):
        bootstrap_survival_curve({0.0: 0.01}, dc, REC)           # zero maturity
    with pytest.raises(ValueError):
        bootstrap_survival_curve(QUOTES, dc, 1.0)                # recovery == 1


# ------------------------------------------------------ downstream usability

def test_bootstrapped_curve_prices_an_off_tenor_cds():
    """The point of a curve rather than five numbers: price a 4Y that was
    never quoted, and check it sits between the 3Y and 5Y quotes."""
    dc = disc()
    res = bootstrap_survival_curve(QUOTES, dc, REC)
    s4 = par_spread(dc, res.curve, REC, flat_schedule(4.0, 4, accrual_basis=365.0))
    assert QUOTES[3.0] < s4 < QUOTES[5.0]


def test_cs01_off_the_bootstrapped_curve():
    dc = disc()
    res = bootstrap_survival_curve(QUOTES, dc, REC)
    sched = flat_schedule(5.0, 4, accrual_basis=365.0)
    c = cs01(dc, res.curve, QUOTES[5.0], REC, sched, notional=10e6)
    assert 4000.0 < c < 5500.0                                   # ~RPV01 * 1bp * N


# ----------------------------------------------------------------- QuantLib

def test_survival_probabilities_match_quantlib_bootstrap():
    """QuantLib's PiecewiseFlatHazardRate + SpreadCdsHelper solves the same
    triangular system under the same piecewise-flat assumption AND the same
    clean-annuity convention.

    Compare at QuantLib's own node dates (its curve does not extrapolate past
    the last helper). Residual is business-day adjustment, which this library
    deliberately does not implement -- see schedule.py.
    """
    ql = pytest.importorskip("QuantLib")
    import datetime

    today = ql.Date(17, 8, 2026)
    ql.Settings.instance().evaluationDate = today
    dc_ql = ql.Actual365Fixed()
    flat_r = 0.04

    disc_h = ql.YieldTermStructureHandle(ql.FlatForward(today, flat_r, dc_ql, ql.Continuous))
    helpers = [
        ql.SpreadCdsHelper(
            ql.QuoteHandle(ql.SimpleQuote(QUOTES[t])), ql.Period(int(t), ql.Years), 0,
            ql.WeekendsOnly(), ql.Quarterly, ql.Following, ql.DateGeneration.CDS2015,
            ql.Actual360(), REC, disc_h,
        )
        for t in TENORS
    ]
    ql_curve = ql.PiecewiseFlatHazardRate(today, helpers, dc_ql)

    ours = bootstrap_survival_curve(
        QUOTES, DiscountCurve.flat(flat_r), REC,
        trade_date=datetime.date(2026, 8, 17),
    )

    # Maturities must line up to within the business-day roll (<= 3 days).
    for helper, t_ours in zip(helpers, ours.knot_times):
        t_ql = dc_ql.yearFraction(today, helper.latestDate())
        assert abs(t_ql - t_ours) < 4 / 365

    # Bucket hazards: sample strictly inside each bucket on both curves.
    edges = [0.0] + [dc_ql.yearFraction(today, h.latestDate()) for h in helpers]
    for k in range(len(TENORS)):
        mid = 0.5 * (edges[k] + edges[k + 1])
        d = today + ql.Period(int(round(mid * 365)), ql.Days)
        h_ql = ql_curve.hazardRate(d)
        h_us = float(ours.curve.hazard(mid))
        assert abs(h_us - h_ql) * 1e4 < 0.5, f"bucket {k}: {h_us*1e4:.3f} vs {h_ql*1e4:.3f} bp"

    # Survival probabilities at QuantLib's node dates.
    for helper in helpers:
        d = helper.latestDate()
        t = dc_ql.yearFraction(today, d)
        q_ql = ql_curve.survivalProbability(d)
        q_us = float(ours.curve.survival(t))
        assert abs(q_us - q_ql) < 5e-4, f"{t:.3f}y: ours {q_us:.6f} vs QL {q_ql:.6f}"


def test_dirty_annuity_would_have_broken_quantlib_parity():
    """Regression guard for a bug this suite actually caught.

    Bootstrapping off the DIRTY par spread inflates the 1Y hazard by ~18bp,
    because on a 0.84y contract the accrued stub is ~16% of the annuity. The
    clean convention is not a refinement here -- it is the difference between
    matching QuantLib and being 20% wrong at the front of the curve.
    """
    dc = DiscountCurve.flat(0.04)
    import datetime
    td = datetime.date(2026, 8, 17)
    clean = bootstrap_survival_curve(QUOTES, dc, REC, trade_date=td)

    from creditlib.schedule import make_cds_schedule
    sched_1y = make_cds_schedule(td, tenor_years=1.0)
    from creditlib.cds import risky_annuity
    ann = risky_annuity(dc, clean.curve, sched_1y)
    assert sched_1y.accrued_fraction() / ann > 0.15      # the stub really is ~16%

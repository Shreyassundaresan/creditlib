"""Step 1 tests: analytic identities, no-arbitrage conditions, QuantLib parity."""

import numpy as np
import pytest

from creditlib import CreditCurve, DiscountCurve, check_credit_curve, check_discount_curve

TIMES = [0.5, 1.0, 3.0, 5.0, 7.0, 10.0]
HAZARDS = [0.0080, 0.0095, 0.0140, 0.0185, 0.0210, 0.0230]


# ------------------------------------------------------------------ analytic

def test_flat_hazard_matches_closed_form():
    """h constant => Q(t) = exp(-h t) exactly."""
    h = 0.02
    c = CreditCurve.flat_hazard(h)
    t = np.array([0.0, 0.25, 1.0, 4.7, 30.0])
    assert np.allclose(c.survival(t), np.exp(-h * t), atol=1e-15)


def test_survival_at_zero_is_one():
    c = CreditCurve(TIMES, HAZARDS)
    assert c.survival(0.0) == pytest.approx(1.0, abs=1e-15)


def test_cumulative_hazard_is_piecewise_linear():
    """H is continuous and its slope on each segment equals that segment's hazard."""
    c = CreditCurve(TIMES, HAZARDS)
    for i, (t_lo, t_hi) in enumerate(zip([0.0] + TIMES[:-1], TIMES)):
        mid_lo, mid_hi = t_lo + 0.1 * (t_hi - t_lo), t_lo + 0.9 * (t_hi - t_lo)
        slope = (c.cumulative_hazard(mid_hi) - c.cumulative_hazard(mid_lo)) / (mid_hi - mid_lo)
        assert slope == pytest.approx(HAZARDS[i], rel=1e-12)


def test_conditional_default_prob_identity():
    """P(tau <= t2 | tau > t1) = 1 - Q(t2)/Q(t1), and survival multiplies."""
    c = CreditCurve(TIMES, HAZARDS)
    t1, t2 = 2.0, 6.0
    assert c.conditional_default_prob(t1, t2) == pytest.approx(1 - c.survival(t2) / c.survival(t1))
    # Q(t2) = Q(t1) * P(survive t1->t2)
    assert c.survival(t2) == pytest.approx(c.survival(t1) * (1 - c.conditional_default_prob(t1, t2)))


def test_flat_extrapolation_beyond_last_knot():
    c = CreditCurve(TIMES, HAZARDS)
    extra = 5.0
    expected = c.survival(TIMES[-1]) * np.exp(-HAZARDS[-1] * extra)
    assert c.survival(TIMES[-1] + extra) == pytest.approx(expected, rel=1e-14)


# --------------------------------------------------------------- round trips

def test_survival_round_trip():
    c = CreditCurve(TIMES, HAZARDS)
    q = c.survival(np.array(TIMES))
    c2 = CreditCurve.from_survival_probabilities(TIMES, q)
    assert np.allclose(c2.rates, HAZARDS, atol=1e-13)


def test_discount_round_trip():
    zeros = [0.041, 0.039, 0.038, 0.0385, 0.0395, 0.0405]
    d = DiscountCurve.from_zero_rates(TIMES, zeros)
    assert np.allclose(d.zero_rate(np.array(TIMES)), zeros, atol=1e-13)
    d2 = DiscountCurve.from_discount_factors(TIMES, d.df(np.array(TIMES)))
    t = np.linspace(0.0, TIMES[-1], 501)
    assert np.allclose(d2.discount_factor(t), d.discount_factor(t), atol=1e-14)


def test_forward_rate_composition():
    """D(t1) * exp(-f(t1,t2)*(t2-t1)) = D(t2)."""
    d = DiscountCurve.from_zero_rates(TIMES, [0.041, 0.039, 0.038, 0.0385, 0.0395, 0.0405])
    t1, t2 = 1.5, 4.25
    f = d.forward_rate(t1, t2)
    assert d.df(t1) * np.exp(-f * (t2 - t1)) == pytest.approx(d.df(t2), rel=1e-14)


# ------------------------------------------------------------- no-arbitrage

def test_no_arbitrage_checks_pass():
    check_credit_curve(CreditCurve(TIMES, HAZARDS), t_max=15.0).raise_if_failed()
    check_discount_curve(DiscountCurve.flat(0.04), t_max=15.0).raise_if_failed()


def test_negative_hazard_rejected():
    with pytest.raises(ValueError, match="negative hazard"):
        CreditCurve([1.0, 2.0], [0.01, -0.001])


def test_non_monotone_survival_rejected():
    """A survival curve that rises implies a negative hazard and must not build."""
    with pytest.raises(ValueError, match="negative hazard"):
        CreditCurve.from_survival_probabilities([1.0, 2.0], [0.98, 0.99])


def test_negative_forwards_allowed_on_discount_curve():
    """D(t) > 1 is a negative rate, not an arbitrage. The monotonicity assert
    is a data-quality guard and must be opted out of explicitly."""
    d = DiscountCurve.from_zero_rates([1.0, 2.0], [-0.005, 0.0025],
                                      require_monotone_df=False)
    assert d.df(1.0) > 1.0
    check_discount_curve(d, t_max=2.0).raise_if_failed()


def test_malformed_inputs_rejected():
    with pytest.raises(ValueError):
        CreditCurve([2.0, 1.0], [0.01, 0.02])       # not increasing
    with pytest.raises(ValueError):
        CreditCurve([0.0, 1.0], [0.01, 0.02])       # knot at zero
    with pytest.raises(ValueError):
        CreditCurve([1.0, 2.0], [0.01])             # length mismatch


# ----------------------------------------------------------------- QuantLib

def test_survival_matches_quantlib():
    """QuantLib HazardRateCurve uses BackwardFlat interpolation on hazards --
    identical to our (t_{i-1}, t_i] convention. Parity here proves the
    convention, not just the arithmetic."""
    ql = pytest.importorskip("QuantLib")

    today = ql.Date(17, 8, 2026)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()

    dates = [today] + [today + ql.Period(int(round(t * 365)), ql.Days) for t in TIMES]
    hazards = [HAZARDS[0]] + list(HAZARDS)
    ql_curve = ql.HazardRateCurve(dates, hazards, dc)

    ours = CreditCurve([dc.yearFraction(today, d) for d in dates[1:]], HAZARDS)

    for d in dates[1:]:
        t = dc.yearFraction(today, d)
        assert ours.survival(t) == pytest.approx(ql_curve.survivalProbability(d), rel=1e-12)

    # Also check interior points, not just the knots.
    for t in [0.13, 0.77, 2.4, 4.1, 6.3, 9.9]:
        d = today + ql.Period(int(round(t * 365)), ql.Days)
        tt = dc.yearFraction(today, d)
        assert ours.survival(tt) == pytest.approx(ql_curve.survivalProbability(d), rel=1e-12)


def test_discount_matches_quantlib():
    """Weaker claim than the hazard test, deliberately. QuantLib's ZeroCurve
    interpolates zero rates linearly, which is NOT piecewise-flat forwards, so
    the two agree only at the knots. That is all this asserts. Full-curve
    parity on the discount side arrives in the bootstrap step, against
    PiecewiseLogLinearDiscount."""
    ql = pytest.importorskip("QuantLib")

    today = ql.Date(17, 8, 2026)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    zeros = [0.041, 0.039, 0.038, 0.0385, 0.0395, 0.0405]

    dates = [today] + [today + ql.Period(int(round(t * 365)), ql.Days) for t in TIMES]
    ql_curve = ql.ZeroCurve(dates, [zeros[0]] + zeros, dc, ql.NullCalendar(),
                            ql.Linear(), ql.Continuous)

    ours = DiscountCurve.from_zero_rates([dc.yearFraction(today, d) for d in dates[1:]], zeros)

    for d in dates[1:]:
        t = dc.yearFraction(today, d)
        assert ours.df(t) == pytest.approx(ql_curve.discount(d), rel=1e-12)

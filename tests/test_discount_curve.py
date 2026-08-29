"""Discount curve tests: identities, interpolation behaviour, QuantLib parity."""

import numpy as np
import pytest

from creditlib.discount_curve import (
    ANNUAL,
    CONTINUOUS,
    QUARTERLY,
    SEMIANNUAL,
    SIMPLE,
    DiscountCurve,
)

TIMES = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 30.0]
ZEROS = [0.0530, 0.0518, 0.0472, 0.0421, 0.0399, 0.0392, 0.0398, 0.0410, 0.0425]


def curve(**kw) -> DiscountCurve:
    return DiscountCurve.from_zero_rates(TIMES, ZEROS, **kw)


# -------------------------------------------------------------- fundamentals

def test_df_at_zero_is_one():
    assert curve().discount_factor(0.0) == pytest.approx(1.0, abs=1e-15)


def test_dfs_strictly_positive_and_decreasing():
    c = curve()
    t = np.linspace(0.0, 30.0, 5001)
    d = c.discount_factor(t)
    assert np.all(d > 0.0)
    assert np.all(np.diff(d) <= 1e-14)


def test_knots_reproduced_exactly():
    c = curve()
    assert np.allclose(c.zero_rate(np.array(TIMES), CONTINUOUS), ZEROS, atol=1e-14)


# ---------------------------------------------------------- rate conversions

def test_zero_rate_convention_round_trips():
    """Each compounding convention must invert its own definition."""
    c = curve()
    t = np.array([0.4, 1.7, 4.2, 12.0])
    d = c.discount_factor(t)

    assert np.allclose(np.exp(-c.zero_rate(t, CONTINUOUS) * t), d, rtol=1e-14)
    assert np.allclose(1.0 / (1.0 + c.zero_rate(t, SIMPLE) * t), d, rtol=1e-14)
    for m, name in [(1, ANNUAL), (2, SEMIANNUAL), (4, QUARTERLY)]:
        z = c.zero_rate(t, name)
        assert np.allclose((1.0 + z / m) ** (-m * t), d, rtol=1e-14)


def test_continuous_annual_identity():
    """1 + z_annual = exp(z_continuous)."""
    c = curve()
    t = np.array([0.6, 3.3, 9.1])
    assert np.allclose(1.0 + c.zero_rate(t, ANNUAL), np.exp(c.zero_rate(t, CONTINUOUS)), rtol=1e-13)


def test_discrete_converges_to_continuous():
    c = curve()
    z_cc = c.zero_rate(5.0, CONTINUOUS)
    prev = abs(c.zero_rate(5.0, 1) - z_cc)
    for m in (4, 52, 365, 10_000):
        err = abs(c.zero_rate(5.0, m) - z_cc)
        assert err < prev
        prev = err
    assert prev < 1e-5


# ------------------------------------------------------------- forward rates

def test_forward_replication_identity():
    """The derivation itself: D(t1) * exp(-f*(t2-t1)) must equal D(t2)."""
    c = curve()
    for t1, t2 in [(0.0, 0.25), (1.0, 2.0), (2.4, 6.8), (10.0, 30.0)]:
        f = c.forward_rate(t1, t2, CONTINUOUS)
        assert c.discount_factor(t1) * np.exp(-f * (t2 - t1)) == pytest.approx(
            c.discount_factor(t2), rel=1e-14
        )


def test_simple_forward_matches_definition():
    c = curve()
    t1, t2 = 1.0, 1.25
    f = c.forward_rate(t1, t2, SIMPLE)
    growth = c.discount_factor(t1) / c.discount_factor(t2)
    assert f == pytest.approx((growth - 1.0) / (t2 - t1), rel=1e-14)


def test_accrual_override_changes_simple_forward():
    """A contract accruing ACT/360 over a period the curve measures ACT/365F
    pays a different amount. The override is the hook for that.

    91 days: ACT/360 -> 0.25278, ACT/365F -> 0.24932. The 360-day year gives
    the LARGER year fraction, so for identical growth D(t1)/D(t2) the quoted
    simple rate is LOWER. Getting this direction backwards is the standard
    mistake -- 'divide by 360' feels like it should shrink the fraction.
    """
    c = curve()
    t1, t2 = 1.0, 1.25
    f_curve = c.forward_rate(t1, t2, SIMPLE)
    f_act360 = c.forward_rate(t1, t2, SIMPLE, accrual=91 / 360)
    assert f_act360 != pytest.approx(f_curve)
    assert 91 / 360 > 91 / 365
    assert f_act360 < f_curve


def test_forwards_constant_within_bucket():
    """The defining property of log-linear DF interpolation."""
    c = curve()
    lo, hi = 3.0, 5.0
    f_ref = c.forward_rate(lo, hi)
    for a, b in [(3.1, 3.4), (3.9, 4.05), (4.5, 4.99), (3.0, 4.0)]:
        assert c.forward_rate(a, b) == pytest.approx(f_ref, rel=1e-13)
    assert c.instantaneous_forward(4.2) == pytest.approx(f_ref, rel=1e-13)


def test_zero_is_average_of_instantaneous_forwards():
    """z_c(t) = (1/t) * integral_0^t f(s) ds."""
    c = curve()
    for t in (0.8, 4.0, 11.5):
        grid = np.linspace(1e-9, t, 400_001)
        avg = np.trapezoid(c.instantaneous_forward(grid), grid) / t
        assert avg == pytest.approx(c.zero_rate(t, CONTINUOUS), rel=1e-6)


# ---------------------------------------------------- interpolation contrast

def test_linear_zeros_can_imply_negative_forwards():
    """Motivating defect of linear-on-zero-rates interpolation: a steep
    inversion produces a negative implied forward from strictly positive
    zero rates. f(t1,t2) = (z2*t2 - z1*t1)/(t2 - t1)."""
    z1, t1, z2, t2 = 0.05, 1.0, 0.02, 2.0
    f = (z2 * t2 - z1 * t1) / (t2 - t1)
    assert f < 0.0
    # Our curve reproduces exactly this forward, because it is a property of
    # the quotes, not of the interpolation -- and it builds only with the
    # monotonicity check disabled.
    c = DiscountCurve.from_zero_rates([t1, t2], [z1, z2], require_monotone_df=False)
    assert c.forward_rate(t1, t2) == pytest.approx(f, rel=1e-13)
    with pytest.raises(AssertionError, match="discount factors increase"):
        DiscountCurve.from_zero_rates([t1, t2], [z1, z2])


# ------------------------------------------------------------ extrapolation

def test_flat_forward_extrapolation():
    c = curve()
    f_last = c.forward_rate(10.0, 30.0)
    extra = 7.0
    expected = c.discount_factor(30.0) * np.exp(-f_last * extra)
    assert c.discount_factor(30.0 + extra) == pytest.approx(expected, rel=1e-14)
    # Not flat in D: that would assert a zero forward rate.
    assert c.discount_factor(37.0) < c.discount_factor(30.0)


# ------------------------------------------------------------- input hygiene

def test_bad_inputs_rejected():
    with pytest.raises(ValueError):
        DiscountCurve.from_zero_rates([2.0, 1.0], [0.04, 0.04])      # not increasing
    with pytest.raises(ValueError):
        DiscountCurve.from_zero_rates([0.0, 1.0], [0.04, 0.04])      # knot at zero
    with pytest.raises(ValueError):
        DiscountCurve.from_zero_rates([1.0, 2.0], [0.04])            # length mismatch
    with pytest.raises(ValueError):
        DiscountCurve.from_discount_factors([1.0], [-0.5])           # negative DF
    with pytest.raises(ValueError):
        curve().forward_rate(2.0, 1.0)                               # t2 <= t1
    with pytest.raises(ValueError):
        curve().zero_rate(-1.0)                                      # negative t
    with pytest.raises(ValueError):
        curve().zero_rate(1.0, "fortnightly")                        # unknown convention


def test_negative_rate_curve_builds_when_flag_disabled():
    c = DiscountCurve.from_zero_rates([1.0, 5.0], [-0.004, 0.002], require_monotone_df=False)
    assert c.discount_factor(1.0) > 1.0
    assert np.all(c.discount_factor(np.linspace(0, 5, 501)) > 0.0)


# ----------------------------------------------------------------- QuantLib

def test_matches_quantlib_loglinear_discount_curve():
    """ql.DiscountCurve is InterpolatedDiscountCurve<LogLinear> -- the exact
    same interpolation scheme, so parity should hold everywhere, not just at
    the knots."""
    ql = pytest.importorskip("QuantLib")

    today = ql.Date(17, 8, 2026)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()

    dates = [today] + [today + ql.Period(int(round(t * 365)), ql.Days) for t in TIMES]
    times = [dc.yearFraction(today, d) for d in dates[1:]]

    ours = DiscountCurve.from_zero_rates(times, ZEROS)
    dfs = [1.0] + list(ours.discount_factor(np.array(times)))
    ql_curve = ql.DiscountCurve(dates, dfs, dc)

    for t in [0.07, 0.25, 0.6, 1.0, 1.9, 3.7, 5.0, 8.4, 10.0, 22.0, 30.0]:
        d = today + ql.Period(int(round(t * 365)), ql.Days)
        tt = dc.yearFraction(today, d)
        assert ours.discount_factor(tt) == pytest.approx(ql_curve.discount(d), rel=1e-12)


def test_zero_and_forward_rates_match_quantlib():
    ql = pytest.importorskip("QuantLib")

    today = ql.Date(17, 8, 2026)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()

    dates = [today] + [today + ql.Period(int(round(t * 365)), ql.Days) for t in TIMES]
    times = [dc.yearFraction(today, d) for d in dates[1:]]
    ours = DiscountCurve.from_zero_rates(times, ZEROS)
    dfs = [1.0] + list(ours.discount_factor(np.array(times)))
    ql_curve = ql.DiscountCurve(dates, dfs, dc)

    for t in [0.5, 2.0, 4.4, 9.0, 18.0]:
        d = today + ql.Period(int(round(t * 365)), ql.Days)
        tt = dc.yearFraction(today, d)
        assert ours.zero_rate(tt, CONTINUOUS) == pytest.approx(
            ql_curve.zeroRate(d, dc, ql.Continuous).rate(), rel=1e-12
        )
        assert ours.zero_rate(tt, ANNUAL) == pytest.approx(
            ql_curve.zeroRate(d, dc, ql.Compounded, ql.Annual).rate(), rel=1e-12
        )

    for t1, t2 in [(1.0, 2.0), (3.0, 5.0), (5.0, 7.0)]:
        d1 = today + ql.Period(int(round(t1 * 365)), ql.Days)
        d2 = today + ql.Period(int(round(t2 * 365)), ql.Days)
        tt1, tt2 = dc.yearFraction(today, d1), dc.yearFraction(today, d2)
        assert ours.forward_rate(tt1, tt2, CONTINUOUS) == pytest.approx(
            ql_curve.forwardRate(d1, d2, dc, ql.Continuous).rate(), rel=1e-12
        )

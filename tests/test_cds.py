"""CDS valuation tests.

The two hand-checked anchors:

  A. One annual payment, no accrual-on-default, flat r and h. Every quantity
     has a closed form derivable on paper in three lines.
  B. Continuously paid premium with flat r, h gives s_par = h(1-R) EXACTLY,
     independent of r and T. Discrete schedules must converge to it.
"""

from datetime import date

import numpy as np
import pytest

from creditlib import (
    CreditCurve,
    DiscountCurve,
    SurvivalCurve,
    cs01,
    flat_schedule,
    ir_dv01,
    jump_to_default,
    make_cds_schedule,
    par_spread,
    premium_leg_pv,
    price_cds,
    protection_leg_pv,
    rec01,
    risky_annuity,
)

R_FLAT, H_FLAT, REC = 0.05, 0.02, 0.40


def flat_setup():
    return DiscountCurve.flat(R_FLAT), CreditCurve.flat_hazard(H_FLAT)


# ------------------------------------------------------------- HAND CHECK A

def test_hand_check_A_single_annual_payment():
    """Closed form, computable on paper.

        PV_prot = (1-R) * h/(r+h) * (1 - e^{-(r+h)T})
                = 0.60 * (0.02/0.07) * (1 - e^{-0.07}) = 0.011589630873
        RPV01   = 1.0 * D(1)Q(1) = e^{-0.07}          = 0.932393819906
        s_par   = 0.011589631 / 0.932393820          = 124.2997 bp
    """
    dc, cc = flat_setup()
    sch = flat_schedule(1.0, frequency=1, accrual_basis=365.0)
    p = price_cds(dc, cc, 0.0124, REC, sch, include_accrual=False)

    c = R_FLAT + H_FLAT
    hand_prot = (1 - REC) * (H_FLAT / c) * (1 - np.exp(-c * 1.0))
    hand_ann = np.exp(-c * 1.0)

    assert p.protection_leg_pv == pytest.approx(hand_prot, abs=1e-15)
    assert p.risky_annuity == pytest.approx(hand_ann, abs=1e-15)
    assert p.par_spread == pytest.approx(hand_prot / hand_ann, rel=1e-14)
    assert p.par_spread * 1e4 == pytest.approx(124.299739, abs=1e-5)


def test_hand_check_A_mtm_and_sign_convention():
    """MTM_buyer = (s_par - s_contract) * RPV01, and the seller is the negative."""
    dc, cc = flat_setup()
    sch = flat_schedule(1.0, frequency=1, accrual_basis=365.0)
    s = 0.0100
    buy = price_cds(dc, cc, s, REC, sch, notional=10e6, include_accrual=False)
    sell = price_cds(dc, cc, s, REC, sch, notional=10e6,
                     protection_buyer=False, include_accrual=False)

    assert buy.mtm == pytest.approx(
        (buy.par_spread - s) * buy.risky_annuity * buy.notional, rel=1e-12
    )
    assert sell.mtm == pytest.approx(-buy.mtm, rel=1e-15)
    # Par spread 124bp > 100bp coupon, so buying protection cheap is a gain.
    assert buy.mtm > 0.0


def test_pricing_at_par_gives_zero_mtm():
    dc, cc = flat_setup()
    sch = flat_schedule(5.0, frequency=4, accrual_basis=365.0)
    s = par_spread(dc, cc, REC, sch)
    p = price_cds(dc, cc, s, REC, sch, notional=10e6)
    assert p.mtm == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------------------- HAND CHECK B

def test_hand_check_B_credit_triangle_is_exact_in_the_limit():
    """Discrete premium must converge to s_par = h(1-R) = 120bp as frequency
    rises, and the convergence must be monotone."""
    dc, cc = flat_setup()
    target = H_FLAT * (1 - REC)
    errs = []
    for f in (1, 2, 4, 12, 52, 365):
        s = par_spread(dc, cc, REC, flat_schedule(5.0, frequency=f, accrual_basis=365.0))
        errs.append(abs(s - target))
    assert all(b < a for a, b in zip(errs, errs[1:])), "convergence not monotone"
    assert errs[-1] * 1e4 < 0.01, "not within 0.01bp at daily frequency"


def test_credit_triangle_independent_of_r_and_T():
    """s_par = h(1-R) in the continuous limit holds for ANY r and T."""
    cc = CreditCurve.flat_hazard(H_FLAT)
    for r in (0.0, 0.02, 0.08):
        for T in (1.0, 5.0, 10.0):
            dc = DiscountCurve.flat(r) if r > 0 else DiscountCurve.flat(1e-12)
            s = par_spread(dc, cc, REC, flat_schedule(T, frequency=365, accrual_basis=365.0))
            assert s == pytest.approx(H_FLAT * (1 - REC), rel=2e-4)


def test_accrual_cancels_the_hazard_part_of_discretisation_error():
    """Sharp claim about WHY the accrual term exists, made testable.

    Over a period of length d, the discrete coupon undervalues the continuous
    one by ~ (f+h) d^2/2 * DQ. The accrual term contributes ~ h d^2/2 * DQ.
    So it cancels exactly the hazard component and leaves the discounting
    component: residual error / no-accrual error -> f/(f+h) = 0.05/0.07.
    """
    dc, cc = flat_setup()
    target = H_FLAT * (1 - REC)
    sch = flat_schedule(5.0, frequency=4, accrual_basis=365.0)
    err_no = par_spread(dc, cc, REC, sch, include_accrual=False) - target
    err_yes = par_spread(dc, cc, REC, sch, include_accrual=True) - target

    assert 0.0 < err_yes < err_no
    assert err_yes / err_no == pytest.approx(R_FLAT / (R_FLAT + H_FLAT), rel=5e-3)


def test_accrual_term_is_positive_and_small():
    dc, cc = flat_setup()
    sch = flat_schedule(5.0, frequency=4, accrual_basis=365.0)
    coupon, accr = premium_leg_pv(dc, cc, sch, 0.012, notional=10e6)
    assert accr > 0.0
    assert accr / coupon < 0.02          # order of h * (period/2)


# ----------------------------------------------------- integration machinery

def test_exact_integral_matches_fine_quadrature():
    """The closed-form protection integral against a brute-force Riemann sum."""
    dc = DiscountCurve.from_zero_rates([1.0, 3.0, 5.0, 10.0], [0.045, 0.040, 0.039, 0.041])
    cc = CreditCurve([0.5, 1.0, 3.0, 5.0, 7.0], [0.006, 0.009, 0.015, 0.021, 0.026])
    sch = flat_schedule(5.0, frequency=4, accrual_basis=365.0)

    exact = protection_leg_pv(dc, cc, sch, REC)

    # The closed form is exact; the Riemann sum is the approximation. The
    # integrand is discontinuous at every hazard knot, so trapezoid converges
    # slowly. Assert CONVERGENCE toward our number rather than agreement at
    # one grid size -- that is the correct direction of the claim.
    errs = []
    for n in (20_001, 200_001, 2_000_001):
        t = np.linspace(0.0, 5.0, n)
        brute = (1 - REC) * np.trapezoid(
            dc.discount_factor(t) * cc.hazard(t) * cc.survival(t), t
        )
        errs.append(abs(brute - exact) / exact)
    assert all(b < a for a, b in zip(errs, errs[1:])), f"no convergence: {errs}"
    assert errs[-1] < 1e-6


def test_no_discounting_gives_undiscounted_expected_loss():
    """f = 0 collapses the protection leg to (1-R)(1 - Q(T))."""
    dc = DiscountCurve.flat(0.0)
    cc = CreditCurve([1.0, 5.0], [0.01, 0.02])
    sch = flat_schedule(5.0, frequency=4, accrual_basis=365.0)
    expected = (1 - REC) * (1.0 - cc.survival(5.0))
    assert protection_leg_pv(dc, cc, sch, REC) == pytest.approx(expected, rel=1e-13)


def test_zero_hazard_gives_riskless_annuity_and_no_protection():
    dc = DiscountCurve.flat(0.04)
    cc = CreditCurve.flat_hazard(0.0)
    sch = flat_schedule(5.0, frequency=4, accrual_basis=365.0)
    assert protection_leg_pv(dc, cc, sch, REC) == pytest.approx(0.0, abs=1e-15)
    riskless = sum(d * dc.discount_factor(t) for d, t in zip(sch.delta, sch.t_end))
    assert risky_annuity(dc, cc, sch) == pytest.approx(riskless, rel=1e-14)


def test_exact_and_midpoint_agree_closely():
    """Midpoint is the usual grid approach. It should be close but not equal --
    if it were equal, one of the two implementations would be wrong."""
    dc, cc = flat_setup()
    sch = flat_schedule(5.0, frequency=4, accrual_basis=365.0)
    s_ex = par_spread(dc, cc, REC, sch, method="exact")
    s_mp = par_spread(dc, cc, REC, sch, method="midpoint")
    assert s_ex != pytest.approx(s_mp, rel=1e-12)
    assert abs(s_ex - s_mp) * 1e4 < 0.05


def test_recovery_monotonicity():
    """Higher recovery -> smaller loss given default -> lower par spread."""
    dc, cc = flat_setup()
    sch = flat_schedule(5.0, frequency=4, accrual_basis=365.0)
    spreads = [par_spread(dc, cc, r, sch) for r in (0.0, 0.2, 0.4, 0.6, 0.8)]
    assert all(b < a for a, b in zip(spreads, spreads[1:]))


def test_hazard_monotonicity():
    dc = DiscountCurve.flat(R_FLAT)
    sch = flat_schedule(5.0, frequency=4, accrual_basis=365.0)
    spreads = [par_spread(dc, CreditCurve.flat_hazard(h), REC, sch)
               for h in (0.005, 0.01, 0.02, 0.05)]
    assert all(b > a for a, b in zip(spreads, spreads[1:]))


# ------------------------------------------------------------ risk measures

def test_cs01_analytic_matches_reprice_to_second_order():
    """Directional claim holds AT PAR on a flat curve. Off par, or on a sloped
    curve, the sign of the second-order gap depends on the shape -- see the
    worked example, where analytic comes in 0.6% BELOW reprice."""
    dc, cc = flat_setup()
    sch = flat_schedule(5.0, frequency=4, accrual_basis=365.0)
    s = par_spread(dc, cc, REC, sch)
    an = cs01(dc, cc, s, REC, sch, notional=10e6, method="analytic")
    rp = cs01(dc, cc, s, REC, sch, notional=10e6, method="reprice")
    assert rp == pytest.approx(an, rel=2e-3)
    # Buying protection gains when spreads widen.
    assert rp > 0.0
    # Analytic overstates: widening lowers Q, which lowers the annuity.
    assert an > rp


def test_cs01_sign_flips_for_seller():
    dc, cc = flat_setup()
    sch = flat_schedule(5.0, frequency=4, accrual_basis=365.0)
    s = par_spread(dc, cc, REC, sch)
    buy = cs01(dc, cc, s, REC, sch, notional=10e6, protection_buyer=True)
    sell = cs01(dc, cc, s, REC, sch, notional=10e6, protection_buyer=False)
    assert sell == pytest.approx(-buy, rel=1e-9)


def test_ir_dv01_is_small_relative_to_cs01():
    """The legs largely offset under a rate shift. Order of magnitude matters
    more than the exact figure."""
    dc, cc = flat_setup()
    sch = flat_schedule(5.0, frequency=4, accrual_basis=365.0)
    s = par_spread(dc, cc, REC, sch)
    c = cs01(dc, cc, s, REC, sch, notional=10e6)
    i = ir_dv01(dc, cc, s, REC, sch, notional=10e6)
    assert abs(i) < 0.05 * abs(c)


def test_jump_to_default_at_par():
    """At par the MTM is zero, so JTD is the full loss-given-default less
    whatever premium has accrued."""
    dc, cc = flat_setup()
    sch = flat_schedule(5.0, frequency=4, accrual_basis=365.0)
    s = par_spread(dc, cc, REC, sch)
    jtd = jump_to_default(dc, cc, s, REC, sch, notional=10e6)
    assert jtd == pytest.approx(10e6 * (1 - REC), rel=1e-6)


def test_jtd_and_cs01_have_opposite_sign_for_seller():
    """A protection seller is short CS01 and short JTD -- both hurt on widening
    and on default. The point of reporting both is that they are not
    substitutes."""
    dc, cc = flat_setup()
    sch = flat_schedule(5.0, frequency=4, accrual_basis=365.0)
    s = par_spread(dc, cc, REC, sch)
    assert cs01(dc, cc, s, REC, sch, notional=10e6, protection_buyer=False) < 0
    assert jump_to_default(dc, cc, s, REC, sch, notional=10e6, protection_buyer=False) < 0


def test_rec01_direction():
    """Higher recovery hurts the protection buyer."""
    dc, cc = flat_setup()
    sch = flat_schedule(5.0, frequency=4, accrual_basis=365.0)
    s = par_spread(dc, cc, REC, sch)
    assert rec01(dc, cc, s, REC, sch, notional=10e6) < 0.0


# ---------------------------------------------------------------- schedule

def test_imm_schedule_structure():
    sch = make_cds_schedule(date(2026, 8, 17), tenor_years=5.0)
    assert sch.maturity.month in (3, 6, 9, 12) and sch.maturity.day == 20
    assert sch.accrual_start[0] == date(2026, 6, 20)      # previous IMM
    assert sch.maturity == date(2031, 6, 20)              # tenor from ACCRUAL START
    assert sch.n == 20                                    # stub + 19 quarters
    assert all(0.24 < d < 0.26 for d in sch.delta)        # ACT/360 quarterly
    # ACT/360 accrual exceeds ACT/365F curve time for the same period.
    assert np.all(sch.delta > (sch.t_end - sch.t_start) - 1e-12)


def test_accrued_rebate_is_positive_mid_period():
    sch = make_cds_schedule(date(2026, 8, 17), tenor_years=5.0)
    assert sch.accrued_fraction() == pytest.approx(58 / 360, abs=1e-12)
    on_imm = make_cds_schedule(date(2026, 9, 20), tenor_years=5.0)
    assert on_imm.accrued_fraction() == pytest.approx(0.0, abs=1e-15)


def test_survival_curve_alias():
    assert SurvivalCurve is CreditCurve


def test_bad_inputs():
    dc, cc = flat_setup()
    sch = flat_schedule(5.0, frequency=4, accrual_basis=365.0)
    with pytest.raises(ValueError):
        price_cds(dc, cc, 0.01, 1.0, sch)                 # recovery == 1
    with pytest.raises(ValueError):
        price_cds(dc, cc, 0.01, REC, sch, method="simpson")
    with pytest.raises(ValueError):
        make_cds_schedule(date(2026, 8, 17))              # no tenor or maturity
    with pytest.raises(ValueError):
        make_cds_schedule(date(2026, 8, 17), maturity=date(2031, 8, 17))  # not IMM


# ----------------------------------------------------------------- QuantLib

def test_par_spread_matches_quantlib_midpoint_engine():
    """QuantLib's MidPointCdsEngine uses the midpoint approximation, so parity
    is expected against method='midpoint', and a small controlled gap against
    method='exact'."""
    ql = pytest.importorskip("QuantLib")

    today = ql.Date(17, 8, 2026)
    ql.Settings.instance().evaluationDate = today
    dc_ql = ql.Actual365Fixed()

    flat_r, flat_h = 0.04, 0.03
    disc = ql.YieldTermStructureHandle(ql.FlatForward(today, flat_r, dc_ql, ql.Continuous))
    prob = ql.DefaultProbabilityTermStructureHandle(
        ql.FlatHazardRate(today, ql.QuoteHandle(ql.SimpleQuote(flat_h)), dc_ql)
    )

    maturity = ql.Date(20, 9, 2031)
    ql_sched = ql.Schedule(ql.Date(20, 6, 2026), maturity, ql.Period(ql.Quarterly),
                           ql.WeekendsOnly(), ql.Following, ql.Unadjusted,
                           ql.DateGeneration.CDS2015, False)
    cds = ql.CreditDefaultSwap(ql.Protection.Buyer, 10e6, 0.01, ql_sched,
                               ql.Following, ql.Actual360(), True, True)
    cds.setPricingEngine(ql.MidPointCdsEngine(prob, REC, disc))
    ql_par = cds.fairSpread()

    ours = make_cds_schedule(date(2026, 8, 17), maturity=date(2031, 9, 20),
                             valuation_date=date(2026, 8, 17))
    dc = DiscountCurve.flat(flat_r)
    cc = CreditCurve.flat_hazard(flat_h)

    # QuantLib's CreditDefaultSwap.fairSpread() leaves the accrued stub IN the
    # annuity, so it is the DIRTY spread -- unlike SpreadCdsHelper, which uses
    # the clean convention when bootstrapping. Compare like with like.
    s_mid = par_spread(dc, cc, REC, ours, method="midpoint", clean=False)
    s_exact = par_spread(dc, cc, REC, ours, method="exact", clean=False)

    # Within a basis point (residual is business-day adjustment, not modelled).
    assert abs(s_mid - ql_par) * 1e4 < 1.0
    assert abs(s_exact - ql_par) * 1e4 < 1.0
    # And the clean spread sits materially above it.
    assert par_spread(dc, cc, REC, ours, clean=True) > s_exact
    # Dirty spread ~172bp; clean ~178bp; credit triangle 180bp. All three
    # differ for reasons the decomposition test pins down exactly.
    assert 170.0 < s_exact * 1e4 < 174.0


def test_imm_par_spread_gap_to_credit_triangle_decomposes():
    """A seasoned IMM CDS prices ~8bp below h(1-R)=180bp. Two causes, both
    contract features:

      1. Accrual is ACT/360 while curve time is ACT/365F, so every delta_i is
         365/360 too large relative to curve time. The annuity is inflated by
         that factor and the par spread deflated by its reciprocal.
      2. The schedule is seasoned: accrual began at the previous IMM (20 Jun)
         but protection only runs from today (17 Aug). The annuity carries
         58/360 of a coupon for which no protection is provided -- that is
         precisely the accrued rebate.

    Netting the accrued (clean=True, the market convention and the default)
    removes wedge 2 entirely, leaving only wedge 1. Both are asserted below.
    """
    dc = DiscountCurve.flat(0.04)
    cc = CreditCurve.flat_hazard(0.03)
    triangle = 0.03 * (1 - REC)

    s_365 = par_spread(dc, cc, REC, flat_schedule(5.0, 4, accrual_basis=365.0))
    s_360 = par_spread(dc, cc, REC, flat_schedule(5.0, 4, accrual_basis=360.0))

    assert s_365 == pytest.approx(triangle, rel=6e-3)          # quarterly discretisation
    assert s_360 / s_365 == pytest.approx(360 / 365, rel=1e-9)  # wedge 1, exact

    sch = make_cds_schedule(date(2026, 8, 17), maturity=date(2031, 9, 20))
    ann = risky_annuity(dc, cc, sch)

    # CLEAN (default): wedge 2 netted out, only the ACT/360 wedge survives.
    s_clean = par_spread(dc, cc, REC, sch, clean=True)
    assert s_clean == pytest.approx(s_360, rel=2e-3)

    # DIRTY: the annuity carries a stub of coupon buying no protection.
    s_dirty = par_spread(dc, cc, REC, sch, clean=False)
    predicted = s_clean * (1 - sch.accrued_fraction() / ann)
    assert s_dirty == pytest.approx(predicted, rel=1e-9)
    assert s_dirty < s_clean
    assert (s_clean - s_dirty) * 1e4 > 5.0        # worth >5bp -- not a rounding issue

    # Traded ON an IMM date there is no stub, so clean and dirty coincide.
    fresh = make_cds_schedule(date(2026, 9, 20), maturity=date(2031, 9, 20))
    assert fresh.accrued_fraction() == 0.0
    assert par_spread(dc, cc, REC, fresh, clean=True) == pytest.approx(
        par_spread(dc, cc, REC, fresh, clean=False), rel=1e-15
    )


# ------------------------------------- seasoned-schedule regression (clean MTM)

def _seasoned():
    """A schedule with a real accrued stub. The synthetic `flat_schedule` used
    elsewhere has accrued_fraction() == 0, so clean and dirty conventions
    coincide there and any inconsistency between them is invisible."""
    dc, cc = flat_setup()
    sch = make_cds_schedule(date(2026, 8, 17), tenor_years=5.0)
    assert sch.accrued_fraction() > 0.0, "fixture must be seasoned"
    return dc, cc, sch


def test_mtm_identity_holds_on_a_seasoned_schedule():
    """MTM = (s_par - s) * A_clean * N.

    Regression: `mtm` was computed on the DIRTY annuity while `par_spread` used
    the CLEAN one, so this identity failed by exactly the accrued rebate on any
    seasoned schedule. Caught by a user run, not by the suite -- every prior
    MTM test used flat_schedule.
    """
    dc, cc, sch = _seasoned()
    s = 0.0100
    p = price_cds(dc, cc, s, REC, sch, notional=10e6)
    assert p.mtm == pytest.approx(
        (p.par_spread - s) * p.risky_annuity_clean * p.notional, rel=1e-12
    )


def test_pricing_at_par_gives_zero_mtm_on_a_seasoned_schedule():
    dc, cc, sch = _seasoned()
    s = par_spread(dc, cc, REC, sch)
    p = price_cds(dc, cc, s, REC, sch, notional=10e6)
    assert p.mtm == pytest.approx(0.0, abs=1e-6)


def test_clean_mtm_equals_stream_pv_plus_accrued():
    """The two views differ by exactly the accrued rebate."""
    dc, cc, sch = _seasoned()
    p = price_cds(dc, cc, 0.0100, REC, sch, notional=10e6)
    assert p.mtm == pytest.approx(p.stream_pv + p.accrued, rel=1e-12)
    assert p.risky_annuity_clean == pytest.approx(
        p.risky_annuity - sch.accrued_fraction(), rel=1e-15
    )
    assert p.stream_pv == pytest.approx(p.protection_leg_pv - p.premium_leg_pv, rel=1e-12)


def test_clean_and_dirty_coincide_when_traded_on_an_imm_date():
    """No stub, so the two conventions must agree exactly."""
    dc, cc = flat_setup()
    fresh = make_cds_schedule(date(2026, 9, 20), tenor_years=5.0)
    assert fresh.accrued_fraction() == 0.0
    p = price_cds(dc, cc, 0.0100, REC, fresh, notional=10e6)
    assert p.mtm == pytest.approx(p.stream_pv, rel=1e-15)
    assert p.risky_annuity_clean == pytest.approx(p.risky_annuity, rel=1e-15)


def test_jtd_uses_stream_pv_not_clean_mark():
    """On default the accrued stub is OWED, not rebated, so JTD must net against
    the cashflow stream. Netting against the clean mark double-counts it."""
    dc, cc, sch = _seasoned()
    p = price_cds(dc, cc, 0.0100, REC, sch, notional=10e6)
    jtd = jump_to_default(dc, cc, 0.0100, REC, sch, notional=10e6)
    assert jtd == pytest.approx(10e6 * (1 - REC) - p.accrued - p.stream_pv, rel=1e-12)
    assert jtd != pytest.approx(10e6 * (1 - REC) - p.accrued - p.mtm, rel=1e-9)


def test_seller_signs_flip_on_both_views():
    dc, cc, sch = _seasoned()
    buy = price_cds(dc, cc, 0.0100, REC, sch, notional=10e6, protection_buyer=True)
    sell = price_cds(dc, cc, 0.0100, REC, sch, notional=10e6, protection_buyer=False)
    assert sell.mtm == pytest.approx(-buy.mtm, rel=1e-15)
    assert sell.stream_pv == pytest.approx(-buy.stream_pv, rel=1e-15)


def test_cs01_unaffected_by_the_clean_convention():
    """Accrued does not depend on the curve, so it cancels in any bump-difference.
    CS01 must be identical under either MTM convention."""
    dc, cc, sch = _seasoned()
    s = par_spread(dc, cc, REC, sch)
    c = cs01(dc, cc, s, REC, sch, notional=10e6, method="reprice")
    bumped = CreditCurve(cc.times, cc.rates + 1e-4)
    delta_stream = (price_cds(dc, bumped, s, REC, sch, 10e6).stream_pv
                    - price_cds(dc, cc, s, REC, sch, 10e6).stream_pv)
    delta_mtm = (price_cds(dc, bumped, s, REC, sch, 10e6).mtm
                 - price_cds(dc, cc, s, REC, sch, 10e6).mtm)
    assert delta_stream == pytest.approx(delta_mtm, rel=1e-12)
    assert c > 0.0

"""Recovery model tests: anchors, monotonicity, integration, and the
identification property that motivates the whole module."""

from datetime import date

import numpy as np
import pytest

from creditlib import IssuerProfile, RecoveryModel, bank_profile, compare, recovery_map

TD = date(2026, 8, 17)


def neutral(seniority="senior_unsecured", **kw):
    kw.setdefault("tangibility", 0.5)
    return IssuerProfile("test", sector="other", seniority=seniority, **kw)


# ------------------------------------------------------------- calibration

def test_anchors():
    """Calibration targets from the published recovery studies."""
    m = RecoveryModel()
    for sen, target in [("senior_unsecured", 0.40), ("secured", 0.65),
                        ("holdco_senior", 0.24), ("junior_subordinated", 0.10)]:
        assert m.predict(neutral(sen)).point == pytest.approx(target, abs=0.02), sen


def test_seniority_ordering_is_strict():
    m = RecoveryModel()
    order = ["secured", "senior_unsecured", "holdco_senior",
             "subordinated", "junior_subordinated"]
    r = [m.predict(neutral(s)).point for s in order]
    assert all(a > b for a, b in zip(r, r[1:])), r


def test_recovery_always_strictly_inside_unit_interval():
    """The logit link guarantees it. The bootstrap divides by (1-R), so a
    prediction of exactly 1.0 would be a divide-by-zero."""
    m = RecoveryModel()
    for lev in (0.0, 50.0, 500.0):
        for cyc in (-1.0, 1.0):
            for cush in (0.0, 1.0):
                for sen in ("secured", "junior_subordinated"):
                    p = m.predict(neutral(sen, leverage=lev, cycle=cyc, debt_cushion=cush)).point
                    assert 0.0 < p < 1.0


# -------------------------------------------------------------- directions

def test_driver_directions():
    m = RecoveryModel()
    base = m.predict(neutral()).point
    assert m.predict(neutral(debt_cushion=0.4)).point > base      # junior debt absorbs first
    assert m.predict(neutral(leverage=8.0)).point < base          # less value per claim
    assert m.predict(neutral(cycle=-1.0)).point < base            # recoveries fall in downturns
    assert m.predict(neutral(tangibility=0.9)).point > base       # hard assets recover more


def test_uncertainty_band_brackets_the_point():
    m = RecoveryModel()
    e = m.predict(neutral())
    assert 0.0 < e.low < e.point < e.high < 1.0
    assert e.lgd == pytest.approx(1 - e.point)


def test_bank_seniority_is_auto_reclassified():
    """Single-name bank CDS references the HOLDCO, whose senior unsecured is
    the TLAC/bail-in layer. Treating it as ordinary corporate senior is the
    largest single error this module exists to prevent."""
    p = IssuerProfile("BigBank", sector="banks", seniority="senior_unsecured", is_bank=True)
    assert p.seniority == "holdco_senior"
    assert "TLAC" in p.notes
    assert RecoveryModel().predict(p).point < 0.30

    assert bank_profile("X").seniority == "holdco_senior"


def test_input_validation():
    with pytest.raises(ValueError):
        IssuerProfile("x", seniority="mezzanine")
    with pytest.raises(ValueError):
        IssuerProfile("x", debt_cushion=1.5)
    with pytest.raises(ValueError):
        IssuerProfile("x", cycle=-3.0)
    with pytest.raises(ValueError):
        IssuerProfile("x", sector="unicorns")


# ------------------------------------------------------------------- fit

def test_fit_requires_enough_data():
    m = RecoveryModel()
    with pytest.raises(ValueError, match="at least 12"):
        m.fit([(neutral(), 0.4)] * 5)


def _planted(n=200, coef=2.0, seed=0):
    """Synthetic sample with a known cushion coefficient."""
    rng = np.random.default_rng(seed)
    obs = []
    for _ in range(n):
        c = float(rng.uniform(0, 0.5))
        z = -0.4 + coef * c + float(rng.normal(0, 0.05))
        obs.append((neutral(debt_cushion=c), 1 / (1 + np.exp(-z))))
    return obs


def test_fit_recovers_a_planted_signal():
    """Synthetic check that the estimator works. Says nothing about real
    recoveries — only that the algebra is right."""
    m = RecoveryModel().fit(_planted(), l2=0.01)
    assert m.fitted
    assert m.c["cush"] == pytest.approx(2.0, abs=0.1)


def test_ridge_shrinks_toward_zero_as_intended():
    """The default l2=1.0 deliberately shrinks coefficients. Recovery samples
    are small and noisy, so unregularised estimates overfit. Shrinkage is the
    feature; this test pins the direction so it can't be lost silently."""
    loose = RecoveryModel().fit(_planted(), l2=0.01).c["cush"]
    tight = RecoveryModel().fit(_planted(), l2=1.0).c["cush"]
    assert 0 < tight < loose
    assert loose == pytest.approx(2.0, abs=0.1)


def test_fit_sets_band_from_residuals():
    """The uncertainty band should come from fit quality, not stay at the
    hand-set default."""
    m = RecoveryModel(band=0.55)
    m.fit(_planted(), l2=0.01)
    assert m.band < 0.55


def test_model_reports_unfitted_by_default():
    assert RecoveryModel().fitted is False


# ------------------------------------------------------- engine integration

def test_uniform_recovery_leaves_ranking_invariant():
    """The baseline property: h = s/(1-R) is a monotone transform of s."""
    sp = {"A": 58, "B": 67, "C": 85, "D": 148}
    orders = []
    for R in (0.40, 0.25, 0.10):
        orders.append(list(compare(sp, recovery=R, trade_date=TD).table["name"]))
    assert orders[0] == orders[1] == orders[2]


def test_name_specific_recovery_breaks_the_invariance():
    """The whole point of the module. Two names at nearly the same spread
    invert once loss given default differs."""
    sp = {"SecuredCo": 57, "BankCo": 58}
    Rmap = {"SecuredCo": 0.75, "BankCo": 0.24}
    flat = list(compare(sp, recovery=0.40, trade_date=TD).table["name"])
    mod = list(compare(sp, recovery=Rmap, trade_date=TD).table["name"])
    assert flat != mod
    # High recovery -> each default is cheap -> the same spread needs MORE of them.
    t = compare(sp, recovery=Rmap, trade_date=TD).table.set_index("name")
    assert t.loc["SecuredCo", "PD_5y"] > t.loc["BankCo", "PD_5y"]


def test_recovery_map_feeds_straight_into_compare():
    m = RecoveryModel()
    profs = [bank_profile("JPM", cushion=0.22), bank_profile("C", cushion=0.25),
             IssuerProfile("Duke", sector="utilities", leverage=4.8)]
    Rmap = recovery_map(m, profs)
    assert set(Rmap) == {"JPM", "C", "Duke"}
    out = compare({"JPM": 58, "C": 85, "Duke": 62}, recovery=Rmap, trade_date=TD)
    assert out.per_name_recovery
    assert "recovery" in out.table.columns
    assert out.table["PD_5y"].is_monotonic_increasing


def test_missing_recovery_is_rejected_not_defaulted():
    with pytest.raises(ValueError, match="no recovery supplied"):
        compare({"A": 58, "B": 67}, recovery={"A": 0.4}, trade_date=TD)


def test_quotes_still_reprice_exactly_under_name_specific_recovery():
    sp = {"A": 58, "B": 85, "C": 212}
    Rmap = {"A": 0.24, "B": 0.51, "C": 0.43}
    out = compare(sp, recovery=Rmap, trade_date=TD)
    for nm, a in out.analyses.items():
        assert a.max_rel_error < 1e-13


def test_twenty_names_run():
    sp = {f"N{i:02d}": 40 + 18 * i for i in range(20)}
    out = compare(sp, recovery=0.40, trade_date=TD)
    assert len(out.table) == 20
    assert out.table["PD_5y"].is_monotonic_increasing

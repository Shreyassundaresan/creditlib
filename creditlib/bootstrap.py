"""Bootstrap a piecewise-flat hazard curve from quoted CDS par spreads.

The inverse of Step 4. Given quotes s_k at maturities T_1 < ... < T_n, solve

    Phi_k(h_1, ..., h_n) = PV_prot_k - s_k * A_k = 0,    k = 1..n

using the Step 4 pricer unchanged. There is no new pricing mathematics here:
the bootstrap is the pricer run backwards.

Why sequentially
----------------
The k-th CDS matures at T_k, so both legs integrate against Q(t) only for
t <= T_k, and Q on that range depends only on h_1..h_k. Hence

    d Phi_k / d h_j = 0   for all j > k

The Jacobian is LOWER TRIANGULAR, so sequential bootstrapping is precisely
forward substitution on a triangular system. Consequences:

  * n one-dimensional root-finds. No Jacobian, no vector initial guess, no
    divergence risk that a multivariate Newton would carry.
  * Locality: re-quoting the 10Y cannot move h_1..h_4. Without this, bucketed
    CS01 would not mean anything.
  * Knots sit AT the quote maturities, so the system is square and every input
    reprices exactly -- not least-squares, which would smear misfit across all
    tenors.
  * Errors propagate forward only. A stale 1Y quote poisons every hazard after
    it and none before it.

Why solve Phi rather than s_par(h) - s_mkt
------------------------------------------
Both have the same root and the same monotonicity, but the ratio form
degenerates. At the first tenor A -> 0 as h grows, so s_par is unbounded
(s_par ~ 3.0 at h = 5 on a 1Y contract) while Phi stays bounded and is
measurably less curved. Brent copes with either; Phi is the better-conditioned
object and costs nothing.

Monotonicity: raising h_k raises PV_prot (more default probability in the
final bucket) and lowers the coupon part of A (less expected premium). Both
push Phi the same way, so Phi is strictly increasing in h_k and has at most
one root. The accrual part of A also rises with h -- an O(h * delta^2) term,
dominated over any realistic range, and verified numerically rather than
asserted.

The recovery degeneracy
-----------------------
CDS spreads identify EXPECTED LOSS, roughly h * (1 - R) -- not h. Bootstrapping
the same quotes at R = 40% and R = 20% reprices every input identically while
implying hazards that differ by a factor of 0.80/0.60. A "CDS-implied default
probability" is only ever as good as the exogenous recovery assumption bolted
onto it, and that assumption is not observable from the spread curve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .cds import par_spread, protection_leg_pv, risky_annuity
from .curves import CreditCurve
from .discount_curve import DiscountCurve
from .schedule import CdsSchedule, flat_schedule, make_cds_schedule

_RTOL = 4 * np.finfo(float).eps      # Brent's floor; below this is sub-float.
_H_CEILING = 20.0                    # 2000% intensity ~ 18-day expected life.


class BootstrapError(RuntimeError):
    """No admissible non-negative hazard reproduces a quote."""

    def __init__(self, message: str, tenor_index: int, tenor: float,
                 quoted: float, bound: float | None = None):
        self.tenor_index = tenor_index
        self.tenor = tenor
        self.quoted = quoted
        self.bound = bound
        super().__init__(message)


class InvertedCurveError(BootstrapError):
    """Quote too LOW: reproducing it needs a negative forward hazard."""


class SpreadCeilingError(BootstrapError):
    """Quote too HIGH: above the certain-default ceiling."""


@dataclass
class BootstrapResult:
    curve: CreditCurve
    recovery: float
    tenors: np.ndarray
    knot_times: np.ndarray
    quoted_spreads: np.ndarray
    repriced_spreads: np.ndarray
    hazards: np.ndarray
    survival: np.ndarray
    diagnostics: list = field(default_factory=list)

    @property
    def max_abs_error_bp(self) -> float:
        return float(np.max(np.abs(self.repriced_spreads - self.quoted_spreads)) * 1e4)

    @property
    def max_rel_error(self) -> float:
        return float(np.max(np.abs(self.repriced_spreads / self.quoted_spreads - 1.0)))

    def implied_forward_spread(self, k: int) -> float:
        """Credit-triangle reading of bucket k: h_k * (1 - R)."""
        return float(self.hazards[k] * (1.0 - self.recovery))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "tenor": self.tenors,
            "quoted_bp": self.quoted_spreads * 1e4,
            "repriced_bp": self.repriced_spreads * 1e4,
            "err_bp": (self.repriced_spreads - self.quoted_spreads) * 1e4,
            "hazard_bp": self.hazards * 1e4,
            "fwd_spread_bp": self.hazards * (1.0 - self.recovery) * 1e4,
            "survival": self.survival,
            "cum_PD": 1.0 - self.survival,
        })

    def __str__(self) -> str:
        body = self.to_frame().to_string(
            index=False,
            formatters={
                "tenor": lambda x: f"{x:6.1f}",
                "quoted_bp": lambda x: f"{x:10.4f}",
                "repriced_bp": lambda x: f"{x:12.4f}",
                "err_bp": lambda x: f"{x:11.2e}",
                "hazard_bp": lambda x: f"{x:11.4f}",
                "fwd_spread_bp": lambda x: f"{x:14.4f}",
                "survival": lambda x: f"{x:10.6f}",
                "cum_PD": lambda x: f"{x:8.6f}",
            },
        )
        return (f"{body}\n\nmax repricing error: {self.max_abs_error_bp:.3e} bp "
                f"({self.max_rel_error:.2e} relative), R = {self.recovery:.0%}")


def _default_schedule_fn(trade_date: date | None):
    if trade_date is None:
        return lambda tenor: flat_schedule(tenor, frequency=4, accrual_basis=365.0)
    return lambda tenor: make_cds_schedule(trade_date, tenor_years=tenor)


def _normalise_quotes(spreads) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(spreads, Mapping):
        keys = sorted(spreads)
        tenors = np.array(keys, dtype=float)
        values = np.array([spreads[k] for k in keys], dtype=float)
    else:
        arr = np.asarray(spreads, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("spreads must be a {tenor: spread} mapping or an (n, 2) array")
        order = np.argsort(arr[:, 0])
        tenors, values = arr[order, 0], arr[order, 1]

    if tenors.size == 0:
        raise ValueError("no quotes supplied")
    if np.any(tenors <= 0.0):
        raise ValueError("quote maturities must be positive")
    if tenors.size > 1 and np.any(np.diff(tenors) <= 0.0):
        raise ValueError("quote maturities must be strictly increasing and distinct")
    if np.any(values <= 0.0):
        raise ValueError("quoted spreads must be strictly positive")
    return tenors, values


def bootstrap_survival_curve(
    spreads,
    discount_curve: DiscountCurve,
    recovery: float,
    trade_date: date | None = None,
    schedule_fn: Callable[[float], CdsSchedule] | None = None,
    include_accrual: bool = True,
    method: str = "exact",
) -> BootstrapResult:
    """Solve for the piecewise-flat hazard curve repricing every quote to par.

    Parameters
    ----------
    spreads : {tenor_years: par_spread} mapping, or an (n, 2) array of
        (tenor, spread) pairs. Decimal, e.g. 0.0085 for 85bp.
    discount_curve : from Step 2.
    recovery : assumed recovery in [0, 1). An assumption, not a quote -- see
        the module docstring on the h * (1-R) degeneracy.
    trade_date : if supplied, real IMM schedules are built per tenor and knots
        land on actual IMM maturity times. If None, synthetic quarterly
        schedules on exact year fractions are used.

    Raises
    ------
    InvertedCurveError / SpreadCeilingError : both subclasses of
        BootstrapError, carrying the offending tenor and the achievable bound.
    """
    if not 0.0 <= recovery < 1.0:
        raise ValueError("recovery must be in [0, 1)")

    tenors, quotes = _normalise_quotes(spreads)
    build = schedule_fn or _default_schedule_fn(trade_date)
    schedules = [build(float(t)) for t in tenors]

    # Knots must sit at the ACTUAL schedule maturities, not the nominal tenors,
    # or the system stops being square.
    knot_times = np.array([s.maturity_time for s in schedules], dtype=float)
    if knot_times.size > 1 and np.any(np.diff(knot_times) <= 0.0):
        raise ValueError("schedule maturities are not strictly increasing")

    hazards: list[float] = []
    diagnostics: list[dict] = []

    for k, (s_mkt, sched) in enumerate(zip(quotes, schedules)):

        def curve_with(h_k: float) -> CreditCurve:
            return CreditCurve(knot_times[: k + 1], np.array(hazards + [float(h_k)]))

        def phi(h_k: float) -> float:
            """PV_prot - s * A. Strictly increasing in h_k."""
            cc = curve_with(h_k)
            prot = protection_leg_pv(discount_curve, cc, sched, recovery, 1.0, method)
            ann = risky_annuity(discount_curve, cc, sched, 1.0, include_accrual, method)
            # CLEAN annuity: net the accrued rebate. See par_spread(clean=...).
            return prot - s_mkt * (ann - sched.accrued_fraction())

        # --- lower bracket: h_k = 0 leaves the marginal bucket default-free --
        phi_lo = phi(0.0)
        if phi_lo > 0.0:
            floor = par_spread(discount_curve, curve_with(0.0), recovery, sched,
                               include_accrual, method)
            prev = tenors[k - 1] if k else 0.0
            raise InvertedCurveError(
                f"{tenors[k]:g}Y quoted at {s_mkt*1e4:.2f}bp is BELOW the "
                f"{floor*1e4:.2f}bp floor implied by the shorter quotes. Reproducing "
                f"it needs a negative forward hazard over ({prev:g}Y, {tenors[k]:g}Y] "
                f"— a negative forward default probability. Arbitrage: sell the "
                f"{prev:g}Y leg, buy the {tenors[k]:g}Y leg. The curve is inverted "
                f"beyond the no-arbitrage bound.",
                tenor_index=k, tenor=float(tenors[k]), quoted=float(s_mkt),
                bound=float(floor),
            )

        # --- upper bracket: seed from the credit triangle, expand to sign flip
        h_hi = max(2.0 * s_mkt / (1.0 - recovery), 1e-4)
        expansions = 0
        while phi(h_hi) < 0.0:
            h_hi *= 2.0
            expansions += 1
            if h_hi > _H_CEILING:
                ceiling = par_spread(discount_curve, curve_with(_H_CEILING), recovery,
                                     sched, include_accrual, method)
                raise SpreadCeilingError(
                    f"{tenors[k]:g}Y quoted at {s_mkt*1e4:.2f}bp EXCEEDS the "
                    f"{ceiling*1e4:.2f}bp certain-default ceiling implied by the "
                    f"shorter quotes at R={recovery:.0%}. No non-negative hazard "
                    f"prices protection above the maximum possible loss. Either the "
                    f"quote set is arbitrageable, or the recovery assumption is too "
                    f"high, or this name is distressed and belongs in the "
                    f"points-upfront parameterisation rather than a running spread.",
                    tenor_index=k, tenor=float(tenors[k]), quoted=float(s_mkt),
                    bound=float(ceiling),
                )

        h_k = brentq(phi, 0.0, h_hi, xtol=1e-16, rtol=_RTOL, maxiter=200)
        # Evaluate the residual BEFORE appending: phi closes over `hazards`,
        # which the append would lengthen out from under it.
        phi_at_root = float(phi(h_k))
        hazards.append(float(h_k))
        diagnostics.append({
            "tenor": float(tenors[k]),
            "hazard": float(h_k),
            "seed": float(2.0 * s_mkt / (1.0 - recovery)),
            "bracket_hi": float(h_hi),
            "expansions": expansions,
            "phi_at_root": phi_at_root,
        })

    curve = CreditCurve(knot_times, np.array(hazards))
    repriced = np.array([
        par_spread(discount_curve, curve, recovery, sched, include_accrual, method)
        for sched in schedules
    ])

    return BootstrapResult(
        curve=curve,
        recovery=recovery,
        tenors=tenors,
        knot_times=knot_times,
        quoted_spreads=quotes,
        repriced_spreads=repriced,
        hazards=np.array(hazards),
        survival=np.asarray(curve.survival(knot_times)),
        diagnostics=diagnostics,
    )


def spread_bounds(
    spreads,
    discount_curve: DiscountCurve,
    recovery: float,
    tenor_index: int,
    trade_date: date | None = None,
    schedule_fn: Callable[[float], CdsSchedule] | None = None,
) -> tuple[float, float]:
    """(floor, ceiling) of quotes bootstrappable at `tenor_index`, given the
    shorter quotes.

    floor   : h_k = 0        -- no marginal default risk in the bucket.
    ceiling : h_k -> infinity -- default certain immediately after T_{k-1}.

    A quote outside this window is an arbitrage against the shorter quotes,
    not a numerical difficulty.
    """
    tenors, quotes = _normalise_quotes(spreads)
    if not 0 <= tenor_index < tenors.size:
        raise IndexError("tenor_index out of range")

    build = schedule_fn or _default_schedule_fn(trade_date)
    schedules = [build(float(t)) for t in tenors]
    knot_times = np.array([s.maturity_time for s in schedules], dtype=float)

    if tenor_index == 0:
        prior: list[float] = []
    else:
        head = bootstrap_survival_curve(
            {float(t): float(s) for t, s in zip(tenors[:tenor_index], quotes[:tenor_index])},
            discount_curve, recovery, trade_date, schedule_fn,
        )
        prior = list(head.hazards)

    sched = schedules[tenor_index]
    times = knot_times[: tenor_index + 1]
    lo = par_spread(discount_curve, CreditCurve(times, np.array(prior + [0.0])),
                    recovery, sched)
    hi = par_spread(discount_curve, CreditCurve(times, np.array(prior + [_H_CEILING])),
                    recovery, sched)
    return float(lo), float(hi)


def screen_quotes(spreads, recovery: float) -> list[str]:
    """Cheap pre-flight screen using the credit triangle, H(T) ~ s*T/(1-R).

    Flags tenor pairs where cumulative hazard would fall. Approximate — the
    real test is whether the bootstrap finds a non-negative root — but it
    catches obviously bad marks before any solving happens.
    """
    tenors, quotes = _normalise_quotes(spreads)
    problems = []
    for (t1, s1), (t2, s2) in zip(zip(tenors, quotes), zip(tenors[1:], quotes[1:])):
        if s2 * t2 < s1 * t1:
            problems.append(
                f"{t1:g}Y {s1*1e4:.0f}bp -> {t2:g}Y {s2*1e4:.0f}bp: cumulative hazard "
                f"falls ({s1*t1/(1-recovery)*1e4:.0f} -> {s2*t2/(1-recovery)*1e4:.0f} bp-years)"
            )
    return problems

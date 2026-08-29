"""Discount curve: log-linear interpolation on discount factors.

The discount factor D(t) is the primitive. Zero rates and forward rates are
D re-expressed in rate units; no information is added by either. The class
therefore stores log discount factors and nothing else, and converts on the
way out.

Interpolation
-------------
Linear in log D between knots:

    ln D(t) = (1-w) ln D(t_i) + w ln D(t_{i+1}),   w = (t - t_i)/(t_{i+1} - t_i)

Consequences, in the order they matter:

  1. D(t) = exp(...) > 0 structurally. The only genuine no-arbitrage
     condition on a discount curve cannot be violated by construction.
  2. f(t) = -d/dt ln D(t) is CONSTANT on each bucket. Log-linear DFs are the
     flattest forward curve consistent with the quotes -- minimal information
     injected between knots.
  3. Sampling-independent: any forward computed between two points inside one
     bucket returns that bucket's forward.
  4. Risk is local. Bumping one input quote moves forwards only in the two
     adjacent buckets, so bucketed DV01 does not smear across tenors. This is
     the reason desks use it.

Accepted cost: the forward curve is a step function, discontinuous at knots.
Irrelevant for discounting and linear products. If smooth forwards were
required (convexity-sensitive exotics), monotone-convex or tension splines
would be the move, at the price of losing (2) and weakening (4).

Extrapolation
-------------
Front stub [0, t_1]: the origin (0, ln D = 0) is a genuine knot, so the first
bucket is interpolated, not extrapolated.
Beyond t_n: the last bucket's forward is held flat. Never hold D flat -- that
implies a zero forward rate, which is an economic statement nobody intends.

Time convention
---------------
`times` are year fractions under ONE convention, ACT/365F by default (see
daycount.py). Contract accrual fractions (ACT/360 for CDS premium, 30/360 for
USD fixed legs) are a cashflow-layer concern and never enter this class.
"""

from __future__ import annotations

from datetime import date
from typing import Sequence

import numpy as np
import pandas as pd

from .daycount import ACT_365F, year_fraction

# Compounding conventions accepted by zero_rate / forward_rate.
CONTINUOUS = "continuous"
SIMPLE = "simple"
ANNUAL = "annual"
SEMIANNUAL = "semiannual"
QUARTERLY = "quarterly"
MONTHLY = "monthly"

_FREQUENCY = {ANNUAL: 1, SEMIANNUAL: 2, QUARTERLY: 4, MONTHLY: 12}


def _compounding_frequency(compounding) -> int | None:
    """Return the periods-per-year for a discrete convention, else None."""
    if isinstance(compounding, (int, np.integer)) and not isinstance(compounding, bool):
        if compounding < 1:
            raise ValueError("integer compounding frequency must be >= 1")
        return int(compounding)
    if compounding in _FREQUENCY:
        return _FREQUENCY[compounding]
    if compounding in (CONTINUOUS, SIMPLE):
        return None
    raise ValueError(
        f"unknown compounding {compounding!r}; expected one of "
        f"{[CONTINUOUS, SIMPLE, *_FREQUENCY]} or a positive integer frequency"
    )


class DiscountCurve:
    """Discount curve on log-linearly interpolated discount factors.

    Construct via `from_discount_factors`, `from_zero_rates`, `from_dates`, or
    `flat`. The __init__ signature takes log discount factors directly and is
    mostly for internal use.

    Parameters
    ----------
    times : year fractions, strictly increasing, all > 0.
    log_dfs : natural logs of the discount factors at `times`.
    require_monotone_df : if True (default), assert D is non-increasing.
        NOTE: this is a DATA QUALITY check, not a no-arbitrage condition.
        D(t) > 1 is exactly what a negative rate looks like and is perfectly
        admissible; EUR/JPY/CHF curves traded there for years. Set False when
        building a curve that legitimately has negative forwards.
    """

    def __init__(
        self,
        times: Sequence[float],
        log_dfs: Sequence[float],
        require_monotone_df: bool = True,
        day_count: str = ACT_365F,
    ) -> None:
        times = np.asarray(times, dtype=float).ravel()
        log_dfs = np.asarray(log_dfs, dtype=float).ravel()

        if times.size == 0:
            raise ValueError("curve needs at least one knot")
        if times.size != log_dfs.size:
            raise ValueError(f"times ({times.size}) and log_dfs ({log_dfs.size}) length mismatch")
        if not np.all(np.isfinite(times)) or not np.all(np.isfinite(log_dfs)):
            raise ValueError("times and log_dfs must be finite")
        if times[0] <= 0.0:
            raise ValueError("knot times must be strictly positive; t=0 is implicit")
        if times.size > 1 and np.any(np.diff(times) <= 0.0):
            raise ValueError("times must be strictly increasing")

        # The origin is a real knot: D(0) = 1 exactly, so ln D(0) = 0.
        self._t = np.concatenate(([0.0], times))
        self._log_df = np.concatenate(([0.0], log_dfs))
        self.times = times
        self.day_count = day_count
        self.require_monotone_df = require_monotone_df

        # Piecewise-constant instantaneous forward on each bucket:
        #   f_i = -(ln D_i - ln D_{i-1}) / (t_i - t_{i-1})
        self._fwd = -np.diff(self._log_df) / np.diff(self._t)

        self._validate()

    # ------------------------------------------------------------- validation

    def _validate(self) -> None:
        """Assert-based sanity checks. Run at construction.

        Caveat worth knowing: `assert` is stripped under `python -O`. That is
        acceptable for a construction-time invariant on a research library; a
        production risk system would raise explicitly instead.
        """
        assert abs(self.discount_factor(0.0) - 1.0) < 1e-14, (
            f"D(0) must equal 1, got {self.discount_factor(0.0)!r}"
        )

        grid = np.linspace(0.0, float(self._t[-1]), 2001)
        dfs = self.discount_factor(grid)

        # Genuine no-arbitrage condition: a strictly positive payoff cannot
        # have a non-positive price.
        assert np.all(dfs > 0.0), f"non-positive discount factor: min={dfs.min():.6e}"

        assert np.all(np.isfinite(dfs)), "non-finite discount factor on the curve"

        if self.require_monotone_df:
            worst = np.max(np.diff(dfs))
            assert worst <= 1e-14, (
                f"discount factors increase (max rise {worst:.3e}), implying a negative "
                f"forward rate. This is a data-quality check, not an arbitrage bound -- "
                f"pass require_monotone_df=False if negative rates are intended."
            )

        # Interpolation must reproduce the inputs at the knots.
        assert np.allclose(
            self.discount_factor(self.times), np.exp(self._log_df[1:]), rtol=0.0, atol=1e-14
        ), "interpolation does not reproduce input discount factors at knots"

    # ------------------------------------------------------------------- core

    def _interp_log_df(self, t: np.ndarray) -> np.ndarray:
        """Log-linear in D, with flat-forward extrapolation past the last knot."""
        out = np.interp(t, self._t, self._log_df)
        beyond = t > self._t[-1]
        if np.any(beyond):
            out[beyond] = self._log_df[-1] - self._fwd[-1] * (t[beyond] - self._t[-1])
        return out

    def discount_factor(self, t):
        """D(t): present value of 1 unit paid at year fraction t."""
        t_arr = np.asarray(t, dtype=float)
        scalar = t_arr.ndim == 0
        t_arr = np.atleast_1d(t_arr).astype(float)
        if np.any(t_arr < 0.0):
            raise ValueError("t must be non-negative")
        out = np.exp(self._interp_log_df(t_arr))
        return float(out[0]) if scalar else out

    # Terse alias, since this gets called constantly downstream.
    df = discount_factor

    def zero_rate(self, t, compounding: str | int = CONTINUOUS):
        """Zero rate to t under the given compounding convention.

        continuous : D = exp(-z t)          -> z = -ln D / t
        simple     : D = 1 / (1 + z t)      -> z = (1/D - 1) / t
        m-periodic : D = (1 + z/m)^(-m t)   -> z = m (D^(-1/(m t)) - 1)

        At t = 0 every convention converges to the instantaneous forward at
        the short end, which is what is returned.
        """
        t_arr = np.asarray(t, dtype=float)
        scalar = t_arr.ndim == 0
        t_arr = np.atleast_1d(t_arr).astype(float)
        if np.any(t_arr < 0.0):
            raise ValueError("t must be non-negative")

        m = _compounding_frequency(compounding)
        safe_t = np.where(t_arr > 0.0, t_arr, 1.0)
        d = self.discount_factor(t_arr)

        if compounding == CONTINUOUS:
            z = -np.log(d) / safe_t
        elif compounding == SIMPLE:
            z = (1.0 / d - 1.0) / safe_t
        else:
            z = m * (d ** (-1.0 / (m * safe_t)) - 1.0)

        z = np.where(t_arr > 0.0, z, self._fwd[0])
        return float(z[0]) if scalar else z

    def forward_rate(
        self,
        t1: float,
        t2: float,
        compounding: str | int = CONTINUOUS,
        accrual: float | None = None,
    ) -> float:
        """Forward rate over [t1, t2], locked in today.

        From the replication argument: two riskless strategies costing 1 today
        must pay the same at t2, so 1/D(t2) = (1/D(t1)) * growth(t1 -> t2).

        continuous : f = (ln D(t1) - ln D(t2)) / (t2 - t1)
        simple     : f = (D(t1)/D(t2) - 1) / tau
        m-periodic : f = m * ((D(t1)/D(t2))^(1/(m*(t2-t1))) - 1)

        `accrual` overrides tau for the simple case, which is the correct hook
        for a contract whose accrual day count (ACT/360, 30/360) differs from
        the curve's time measure. Defaults to t2 - t1.
        """
        t1, t2 = float(t1), float(t2)
        if t2 <= t1:
            raise ValueError(f"require t2 > t1, got t1={t1}, t2={t2}")
        if t1 < 0.0:
            raise ValueError("t1 must be non-negative")

        dt = t2 - t1
        m = _compounding_frequency(compounding)
        growth = self.discount_factor(t1) / self.discount_factor(t2)

        if compounding == CONTINUOUS:
            return float(np.log(growth) / dt)
        if compounding == SIMPLE:
            tau = dt if accrual is None else float(accrual)
            if tau <= 0.0:
                raise ValueError("accrual must be positive")
            return float((growth - 1.0) / tau)
        return float(m * (growth ** (1.0 / (m * dt)) - 1.0))

    def instantaneous_forward(self, t):
        """f(t) = -d/dt ln D(t). Piecewise constant by construction."""
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        if np.any(t_arr < 0.0):
            raise ValueError("t must be non-negative")
        idx = np.clip(np.searchsorted(self._t[1:], t_arr, side="left"), 0, self._fwd.size - 1)
        out = self._fwd[idx]
        return float(out[0]) if np.asarray(t).ndim == 0 else out

    # ----------------------------------------------------------- construction

    @classmethod
    def from_discount_factors(cls, times, dfs, **kwargs) -> "DiscountCurve":
        dfs = np.asarray(dfs, dtype=float).ravel()
        if np.any(dfs <= 0.0):
            raise ValueError("discount factors must be strictly positive")
        return cls(times, np.log(dfs), **kwargs)

    @classmethod
    def from_zero_rates(cls, times, zeros, compounding: str | int = CONTINUOUS, **kwargs):
        """Build from zero rates under any supported compounding convention.

        Inverts each convention's definition to recover D at the knots, then
        interpolates log-linearly in D. Note this means the interpolation is
        NOT linear in the input zero rates -- deliberately, see module docstring.
        """
        times = np.asarray(times, dtype=float).ravel()
        zeros = np.asarray(zeros, dtype=float).ravel()
        if times.size != zeros.size:
            raise ValueError("times and zeros must have equal length")

        m = _compounding_frequency(compounding)
        if compounding == CONTINUOUS:
            dfs = np.exp(-zeros * times)
        elif compounding == SIMPLE:
            dfs = 1.0 / (1.0 + zeros * times)
        else:
            dfs = (1.0 + zeros / m) ** (-m * times)
        return cls.from_discount_factors(times, dfs, **kwargs)

    @classmethod
    def from_dates(cls, valuation_date: date, dates, values, kind: str = "df",
                   day_count: str = ACT_365F, compounding: str | int = CONTINUOUS, **kwargs):
        """Build from calendar dates, converting to curve time in one place."""
        times = [year_fraction(valuation_date, d, day_count) for d in dates]
        if kind == "df":
            return cls.from_discount_factors(times, values, day_count=day_count, **kwargs)
        if kind == "zero":
            return cls.from_zero_rates(times, values, compounding=compounding,
                                       day_count=day_count, **kwargs)
        raise ValueError("kind must be 'df' or 'zero'")

    @classmethod
    def flat(cls, rate: float, t_max: float = 50.0, compounding: str | int = CONTINUOUS, **kwargs):
        return cls.from_zero_rates([t_max], [rate], compounding=compounding, **kwargs)

    # --------------------------------------------------------------- display

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "t": self.times,
                "df": self.discount_factor(self.times),
                "zero_cc": self.zero_rate(self.times, CONTINUOUS),
                "zero_annual": self.zero_rate(self.times, ANNUAL),
                "fwd_bucket": self._fwd,
            }
        )

    def __repr__(self) -> str:
        return (
            f"DiscountCurve(n={self.times.size}, t_max={self.times[-1]:.2f}, "
            f"df_max_t={self.discount_factor(self.times[-1]):.6f}, dc={self.day_count})"
        )

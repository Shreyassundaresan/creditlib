"""Piecewise-flat term structures (hazard side).

NOTE: the discount curve moved to discount_curve.py, which supersedes the
DiscountCurve that used to live here. The two share the exp(-integral of a
piecewise-flat rate) form, but the discount side needs its own compounding
conventions and accrual hooks, so it earns a dedicated module.

Both objects a CDS needs -- a discount curve and a survival (credit) curve --
are the same mathematical animal:

    value(t) = exp( - integral_0^t rate(s) ds )

with `rate` a piecewise-constant instantaneous rate. For the discount curve
`rate` is the instantaneous forward rate f(s); for the credit curve it is the
hazard rate h(s). One base class implements the integral; the two subclasses
only differ in naming, validation, and the accessors they expose.

Convention
----------
Knot times t_1 < ... < t_n with an implicit t_0 = 0. The rate r_i applies on
the half-open interval (t_{i-1}, t_i]. This is "backward flat" interpolation:
the rate at a knot is the rate of the segment *ending* there. Beyond t_n the
last rate is extrapolated flat. This matches the ISDA CDS Standard Model and
QuantLib's BackwardFlat interpolation on hazard rates.

Because the integral of a piecewise-constant function is piecewise linear and
continuous, log(value) is piecewise linear in t. Piecewise-constant forward
rates and log-linear interpolation of discount factors / survival
probabilities are therefore the *same* assumption, not two competing ones.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


class PiecewiseFlatCurve:
    """A term structure of the form exp(-integral of a piecewise-flat rate).

    Parameters
    ----------
    times : increasing year fractions, all strictly positive.
    rates : instantaneous rate on each segment; rates[i] applies on
            (times[i-1], times[i]] with times[-1] := 0.
    allow_negative : if False, reject negative rates.
    """

    _rate_name = "rate"

    def __init__(
        self,
        times: Sequence[float],
        rates: Sequence[float],
        allow_negative: bool = True,
    ) -> None:
        times = np.asarray(times, dtype=float).ravel()
        rates = np.asarray(rates, dtype=float).ravel()

        if times.size == 0:
            raise ValueError("curve needs at least one knot")
        if times.size != rates.size:
            raise ValueError(f"times ({times.size}) and rates ({rates.size}) must have equal length")
        if not np.all(np.isfinite(times)) or not np.all(np.isfinite(rates)):
            raise ValueError("times and rates must be finite")
        if times[0] <= 0.0:
            raise ValueError("first knot time must be strictly positive")
        if times.size > 1 and np.any(np.diff(times) <= 0.0):
            raise ValueError("times must be strictly increasing")
        if not allow_negative and np.any(rates < 0.0):
            bad = np.flatnonzero(rates < 0.0)
            raise ValueError(
                f"negative {self._rate_name} at segment(s) {bad.tolist()}: {rates[bad].tolist()}"
            )

        self.times = times
        self.rates = rates

        # Knots including the implicit origin, and the cumulative integral there.
        self._knots = np.concatenate(([0.0], times))               # length n+1
        seg_len = np.diff(self._knots)                             # length n
        self._cum = np.concatenate(([0.0], np.cumsum(rates * seg_len)))  # length n+1

    # ------------------------------------------------------------------ core

    def integral(self, t):
        """Cumulative integral of the rate from 0 to t.

        For the discount curve this is the zero rate times t; for the credit
        curve it is the cumulative hazard H(t).
        """
        t_arr = np.asarray(t, dtype=float)
        scalar = t_arr.ndim == 0
        t_arr = np.atleast_1d(t_arr)
        if np.any(t_arr < 0.0):
            raise ValueError("t must be non-negative")

        # Segment index i such that knots[i] < t <= knots[i+1]; clipped at both
        # ends so that t = 0 uses segment 0 (contributing nothing) and t beyond
        # the last knot extrapolates the final rate flat.
        idx = np.searchsorted(self._knots, t_arr, side="left") - 1
        idx = np.clip(idx, 0, self.rates.size - 1)
        out = self._cum[idx] + self.rates[idx] * (t_arr - self._knots[idx])
        return float(out[0]) if scalar else out

    def rate_at(self, t):
        """The instantaneous piecewise-flat rate at time t."""
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        idx = np.clip(np.searchsorted(self.times, t_arr, side="left"), 0, self.rates.size - 1)
        out = self.rates[idx]
        return float(out[0]) if np.asarray(t).ndim == 0 else out

    def _value(self, t):
        """exp(-integral(t)). Discount factor or survival probability."""
        val = np.exp(-np.asarray(self.integral(t)))
        return float(val) if np.asarray(t).ndim == 0 else val

    # --------------------------------------------------------------- display

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "t": self.times,
                self._rate_name: self.rates,
                "value": self._value(self.times),
            }
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(n={self.times.size}, "
            f"t_max={self.times[-1]:.3f}, "
            f"{self._rate_name}=[{', '.join(f'{r:.4%}' for r in self.rates)}])"
        )

    # ---------------------------------------------------------- construction

    @classmethod
    def _rates_from_values(cls, times: Sequence[float], values: Sequence[float]) -> np.ndarray:
        """Invert value(t) = exp(-integral) to recover the piecewise-flat rates.

        r_i = -( ln v_i - ln v_{i-1} ) / (t_i - t_{i-1}),  v_0 = 1, t_0 = 0.
        """
        times = np.asarray(times, dtype=float).ravel()
        values = np.asarray(values, dtype=float).ravel()
        if times.size != values.size:
            raise ValueError("times and values must have equal length")
        if np.any(values <= 0.0):
            raise ValueError("values must be strictly positive to take logs")
        log_v = np.concatenate(([0.0], np.log(values)))
        t_ext = np.concatenate(([0.0], times))
        return -np.diff(log_v) / np.diff(t_ext)


class CreditCurve(PiecewiseFlatCurve):
    """Survival curve for a single reference entity, piecewise-flat hazard rates.

    Q(t) = P(tau > t) = exp(-int_0^t h(s) ds)

    Hazard rates are constrained non-negative. That single constraint is what
    makes the curve arbitrage-free by construction: Q is then automatically
    (a) equal to 1 at t=0, (b) non-increasing, and (c) bounded in (0, 1].
    """

    _rate_name = "hazard"

    def __init__(self, times, rates, allow_negative: bool = False) -> None:
        super().__init__(times, rates, allow_negative=allow_negative)

    def survival(self, t):
        """Q(t) = P(tau > t)."""
        return self._value(t)

    def default_prob(self, t):
        """Unconditional default probability by t, F(t) = 1 - Q(t)."""
        return 1.0 - np.asarray(self.survival(t)) if np.asarray(t).ndim else 1.0 - self.survival(t)

    def cumulative_hazard(self, t):
        """H(t) = int_0^t h(s) ds = -ln Q(t)."""
        return self.integral(t)

    def hazard(self, t):
        """Instantaneous hazard rate h(t)."""
        return self.rate_at(t)

    def conditional_default_prob(self, t1: float, t2: float) -> float:
        """P(tau <= t2 | tau > t1) = 1 - Q(t2)/Q(t1)."""
        if t2 < t1:
            raise ValueError("require t2 >= t1")
        return 1.0 - self.survival(t2) / self.survival(t1)

    @classmethod
    def from_survival_probabilities(cls, times, survivals) -> "CreditCurve":
        return cls(times, cls._rates_from_values(times, survivals))

    @classmethod
    def flat_hazard(cls, h: float, t_max: float = 50.0) -> "CreditCurve":
        return cls([t_max], [h])

    @classmethod
    def from_flat_spread(cls, spread: float, recovery: float, t_max: float = 50.0) -> "CreditCurve":
        """Credit-triangle approximation: h ~= s / (1 - R).

        Deliberately labelled an approximation. It ignores discounting, premium
        accrual on default, and the timing of the protection payment. Useful as
        a bootstrap starting guess and as an order-of-magnitude sanity check --
        never as a pricing input.
        """
        if not 0.0 <= recovery < 1.0:
            raise ValueError("recovery must be in [0, 1)")
        return cls.flat_hazard(spread / (1.0 - recovery), t_max=t_max)

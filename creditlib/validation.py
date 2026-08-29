"""No-arbitrage checks that can be run on any curve, at any stage of the build.

The philosophy here is that these are not unit tests -- they are runtime
assertions that a *constructed* curve is admissible. Every bootstrap in this
library ends by calling `check_credit_curve` on its own output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .curves import CreditCurve
from .discount_curve import DiscountCurve


@dataclass
class CheckResult:
    passed: bool
    failures: list = field(default_factory=list)

    def raise_if_failed(self) -> "CheckResult":
        if not self.passed:
            raise AssertionError("no-arbitrage violation:\n  " + "\n  ".join(self.failures))
        return self

    def __bool__(self) -> bool:
        return self.passed


def _grid(t_max: float, n: int = 2000) -> np.ndarray:
    return np.linspace(0.0, t_max, n)


def check_credit_curve(curve: CreditCurve, t_max: float | None = None, tol: float = 1e-12) -> CheckResult:
    """Assert the four conditions an admissible survival curve must satisfy.

    1. Q(0) = 1                      -- the name has not defaulted at inception.
    2. Q non-increasing              -- P(default in (t1, t2]) >= 0.
    3. 0 < Q(t) <= 1                 -- Q is a probability.
    4. h(t) >= 0                     -- equivalent to (2), enforced structurally.
    """
    fails: list[str] = []
    t_max = float(curve.times[-1]) if t_max is None else t_max
    t = _grid(t_max)
    q = np.asarray(curve.survival(t))

    if abs(curve.survival(0.0) - 1.0) > tol:
        fails.append(f"Q(0) = {curve.survival(0.0):.16f} != 1")

    dq = np.diff(q)
    if np.any(dq > tol):
        i = int(np.argmax(dq))
        fails.append(f"Q increasing near t={t[i]:.4f}: {q[i]:.12f} -> {q[i+1]:.12f}")

    if np.any(q <= 0.0) or np.any(q > 1.0 + tol):
        fails.append(f"Q out of (0, 1]: min={q.min():.6e}, max={q.max():.6e}")

    if np.any(curve.rates < -tol):
        bad = np.flatnonzero(curve.rates < -tol)
        fails.append(f"negative hazard at segments {bad.tolist()}")

    return CheckResult(not fails, fails)


def check_discount_curve(curve: DiscountCurve, t_max: float | None = None, tol: float = 1e-12) -> CheckResult:
    """Assert what is actually required of a discount curve.

    D(0) = 1 and D(t) > 0 are genuine no-arbitrage conditions. D monotone
    decreasing is *not*: it would rule out negative rates, which trade.
    """
    fails: list[str] = []
    t_max = float(curve.times[-1]) if t_max is None else t_max
    d = np.asarray(curve.df(_grid(t_max)))

    if abs(curve.df(0.0) - 1.0) > tol:
        fails.append(f"D(0) = {curve.df(0.0):.16f} != 1")
    if np.any(d <= 0.0):
        fails.append(f"non-positive discount factor: min={d.min():.6e}")

    return CheckResult(not fails, fails)

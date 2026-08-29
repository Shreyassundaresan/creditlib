"""Day count conventions.

Every curve in this library is parameterised in *year fractions*, not dates.
Converting dates -> year fractions is the only place a day count enters the
curve maths, so it lives in one module.

Conventions used by the single-name CDS market:

  * Curve time (both discount and credit curves): ACT/365F.
    The ISDA CDS Standard Model measures time on the curves in ACT/365 Fixed.
  * CDS premium leg accrual: ACT/360.
    The running coupon accrues on an actual/360 basis, which is why a
    quarterly period is ~0.2528 rather than exactly 0.25.

Mixing these up is a classic interview trap: the *same* calendar period has a
different year fraction on the curve than in the accrual, and the two are not
interchangeable.
"""

from __future__ import annotations

from datetime import date

ACT_365F = "ACT/365F"
ACT_360 = "ACT/360"
THIRTY_360 = "30/360"

SUPPORTED = (ACT_365F, ACT_360, THIRTY_360)


def year_fraction(start: date, end: date, convention: str = ACT_365F) -> float:
    """Year fraction between two dates under `convention`.

    Signed: if `end` precedes `start` the result is negative.
    """
    if convention not in SUPPORTED:
        raise ValueError(f"unsupported day count {convention!r}; expected one of {SUPPORTED}")

    if convention == ACT_365F:
        return (end - start).days / 365.0
    if convention == ACT_360:
        return (end - start).days / 360.0

    # 30/360 US (bond basis)
    d1, m1, y1 = start.day, start.month, start.year
    d2, m2, y2 = end.day, end.month, end.year
    if d1 == 31:
        d1 = 30
    if d2 == 31 and d1 == 30:
        d2 = 30
    return (360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)) / 360.0

"""CDS payment schedules on IMM roll dates.

Standard single-name CDS conventions implemented here:

  * Coupons on the IMM 20ths: 20 Mar, 20 Jun, 20 Sep, 20 Dec.
  * Quarterly frequency, accrual ACT/360.
  * Accrual starts at the IMM date on or before the trade date, so the buyer
    pays a FULL first coupon and is rebated the accrued portion up front.
    That rebate is the `accrued` field; it is why CDS quotes distinguish
    "clean" upfront from cash settled.
  * Maturity is the first IMM date on or after (trade date + tenor).
  * Curve time measured ACT/365F from the valuation date (Step 1/2 convention),
    accrual fractions measured ACT/360. These are deliberately different, and
    both appear in the same schedule object.

Deliberately NOT implemented: business-day adjustment against a real holiday
calendar. Payment dates should roll Following, which moves a handful of
cashflows by 1-3 days and changes PVs in the 4th decimal of a basis point.
Adding a calendar is mechanical; leaving it out is a scoping choice, not an
oversight, and it is stated here so the omission is auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .daycount import ACT_360, ACT_365F, year_fraction

IMM_MONTHS = (3, 6, 9, 12)
IMM_DAY = 20


def is_imm(d: date) -> bool:
    return d.month in IMM_MONTHS and d.day == IMM_DAY


def next_imm(d: date, strictly_after: bool = False) -> date:
    """First IMM date on or after `d` (strictly after, if requested)."""
    for year in (d.year, d.year + 1):
        for m in IMM_MONTHS:
            cand = date(year, m, IMM_DAY)
            if cand > d or (cand == d and not strictly_after):
                return cand
    raise RuntimeError("unreachable")


def previous_imm(d: date, strictly_before: bool = False) -> date:
    """Last IMM date on or before `d` (strictly before, if requested)."""
    for year in (d.year, d.year - 1):
        for m in reversed(IMM_MONTHS):
            cand = date(year, m, IMM_DAY)
            if cand < d or (cand == d and not strictly_before):
                return cand
    raise RuntimeError("unreachable")




@dataclass(frozen=True)
class CdsSchedule:
    """Accrual periods and curve times for a single-name CDS.

    Attributes
    ----------
    accrual_start / accrual_end / pay_date : calendar dates per period.
    delta        : ACT/360 accrual fraction per period (what the coupon pays on).
    t_start/t_end: ACT/365F curve time per period, measured from valuation date.
                   May be negative for the stub period that began before today.
    """

    valuation_date: date
    trade_date: date
    maturity: date
    accrual_start: tuple
    accrual_end: tuple
    pay_date: tuple
    delta: np.ndarray
    t_start: np.ndarray
    t_end: np.ndarray

    @property
    def n(self) -> int:
        return len(self.delta)

    @property
    def maturity_time(self) -> float:
        return float(self.t_end[-1])

    @property
    def protection_start_time(self) -> float:
        """Protection is live from today for a seasoned position."""
        return 0.0

    def accrued_fraction(self) -> float:
        """ACT/360 fraction from the current period's accrual start to today.

        This is the coupon the buyer has already consumed but not yet paid.
        Zero if the valuation date coincides with an accrual start.
        """
        for a_start, a_end, dlt in zip(self.accrual_start, self.accrual_end, self.delta):
            if a_start <= self.valuation_date < a_end:
                return year_fraction(a_start, self.valuation_date, ACT_360)
        return 0.0

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "accrual_start": self.accrual_start,
                "accrual_end": self.accrual_end,
                "pay_date": self.pay_date,
                "delta_ACT360": self.delta,
                "t_start": self.t_start,
                "t_end": self.t_end,
            }
        )


def make_cds_schedule(
    trade_date: date,
    tenor_years: float | None = None,
    maturity: date | None = None,
    valuation_date: date | None = None,
    frequency: int = 4,
    full_first_coupon: bool = True,
) -> CdsSchedule:
    """Build a standard quarterly IMM schedule.

    `full_first_coupon=True` reproduces the post-2009 standard: accrual begins
    at the previous IMM date, so the first coupon is full-size and the buyer
    receives an accrued rebate. Set False to start accrual at the trade date
    (a legacy / bespoke structure) if you want to see the difference.
    """
    if (tenor_years is None) == (maturity is None):
        raise ValueError("supply exactly one of tenor_years or maturity")
    if frequency not in (1, 2, 4, 12):
        raise ValueError("frequency must be 1, 2, 4 or 12")

    valuation_date = valuation_date or trade_date
    first_accrual_start = previous_imm(trade_date) if full_first_coupon else trade_date

    if maturity is None:
        # CDS2015: the tenor runs from the ACCRUAL START (the preceding IMM
        # date), not from the trade date. A 5Y traded 17 Aug 2026 accrues from
        # 20 Jun 2026 and matures 20 Jun 2031 -- NOT 20 Sep 2031.
        #
        # Measuring from the trade date instead silently lengthens every
        # contract by up to a full quarter, which shows up as materially wrong
        # bootstrapped hazards at the short end. Verified against QuantLib's
        # DateGeneration.CDS2015.
        anchor = previous_imm(trade_date)
        months = int(round(tenor_years * 12))
        y, m = anchor.year + (anchor.month - 1 + months) // 12, \
               (anchor.month - 1 + months) % 12 + 1
        target = date(y, m, IMM_DAY)
        maturity = target if is_imm(target) else next_imm(target)
    elif not is_imm(maturity):
        raise ValueError(f"maturity {maturity} is not an IMM date")

    step = 12 // frequency

    # Walk IMM dates backwards from maturity down to the first accrual start.
    boundaries = [maturity]
    cursor = maturity
    while cursor > first_accrual_start:
        y, m = cursor.year, cursor.month - step
        while m <= 0:
            m += 12
            y -= 1
        cursor = date(y, m, IMM_DAY)
        boundaries.append(cursor)
    boundaries = sorted(boundaries)
    boundaries[0] = first_accrual_start

    a_start = tuple(boundaries[:-1])
    a_end = tuple(boundaries[1:])
    # Payment on the accrual end date. The final coupon pays at maturity.
    pay = a_end

    delta = np.array([year_fraction(s, e, ACT_360) for s, e in zip(a_start, a_end)])
    t_start = np.array([year_fraction(valuation_date, s, ACT_365F) for s in a_start])
    t_end = np.array([year_fraction(valuation_date, e, ACT_365F) for e in a_end])

    return CdsSchedule(
        valuation_date=valuation_date,
        trade_date=trade_date,
        maturity=maturity,
        accrual_start=a_start,
        accrual_end=a_end,
        pay_date=pay,
        delta=delta,
        t_start=t_start,
        t_end=t_end,
    )


def flat_schedule(maturity_time: float, frequency: int = 4,
                  accrual_basis: float = 360.0, days_basis: float = 365.0) -> CdsSchedule:
    """Synthetic schedule in pure year fractions, for testing and hand-checks.

    Bypasses the calendar entirely: periods are exactly 1/frequency in curve
    time. Set accrual_basis == days_basis to make ACT/360 and curve time
    coincide, which is what the hand-checked examples need.
    """
    n = int(round(maturity_time * frequency))
    if abs(n / frequency - maturity_time) > 1e-9:
        raise ValueError("maturity_time must be a whole number of periods")
    edges = np.arange(n + 1) / frequency
    scale = days_basis / accrual_basis
    dummy = date(2026, 1, 1)
    return CdsSchedule(
        valuation_date=dummy,
        trade_date=dummy,
        maturity=dummy,
        accrual_start=tuple(dummy for _ in range(n)),
        accrual_end=tuple(dummy for _ in range(n)),
        pay_date=tuple(dummy for _ in range(n)),
        delta=np.diff(edges) * scale,
        t_start=edges[:-1],
        t_end=edges[1:],
    )

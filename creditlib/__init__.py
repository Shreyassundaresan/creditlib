"""creditlib -- a single-name CDS pricing engine built from first principles."""

from . import daycount
from .curves import PiecewiseFlatCurve, CreditCurve
from .discount_curve import DiscountCurve
from .schedule import (
    CdsSchedule, make_cds_schedule, flat_schedule, next_imm, previous_imm,
)
from .cds import (
    SurvivalCurve, price_cds, par_spread, risky_annuity,
    protection_leg_pv, premium_leg_pv, cs01, ir_dv01, rec01, jump_to_default,
)
from .bootstrap import (
    bootstrap_survival_curve, BootstrapResult,
    BootstrapError, InvertedCurveError, SpreadCeilingError,
    spread_bounds, screen_quotes,
)
from .analyze import (
    analyze, compare, recovery_sensitivity, default_curve, Analysis, Comparison,
)
from .recovery import (
    RecoveryModel, IssuerProfile, RecoveryEstimate, bank_profile, recovery_map,
    SECTOR_TANGIBILITY, SENIORITY_SCORE,
)
from .validation import check_credit_curve, check_discount_curve

__all__ = [
    "daycount",
    "PiecewiseFlatCurve",
    "CreditCurve",
    "SurvivalCurve",
    "DiscountCurve",
    "CdsSchedule",
    "make_cds_schedule",
    "flat_schedule",
    "next_imm",
    "previous_imm",
    "price_cds",
    "par_spread",
    "risky_annuity",
    "protection_leg_pv",
    "premium_leg_pv",
    "cs01",
    "ir_dv01",
    "rec01",
    "jump_to_default",
    "bootstrap_survival_curve",
    "BootstrapResult",
    "BootstrapError",
    "InvertedCurveError",
    "SpreadCeilingError",
    "spread_bounds",
    "screen_quotes",
    "analyze",
    "compare",
    "recovery_sensitivity",
    "default_curve",
    "Analysis",
    "Comparison",
    "RecoveryModel",
    "IssuerProfile",
    "RecoveryEstimate",
    "bank_profile",
    "recovery_map",
    "check_credit_curve",
    "check_discount_curve",
]

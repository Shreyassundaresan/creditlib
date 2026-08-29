"""Recovery rate prediction from issuer fundamentals.

Why this module exists
----------------------
A CDS spread identifies EXPECTED LOSS, approximately h * (1 - R). One equation,
two unknowns. Neither the hazard rate nor the recovery rate is separately
identified from a single spread -- bootstrapping the same quote at R = 40% and
R = 20% reprices it to machine precision while implying hazards that differ by
a factor of (1-0.20)/(1-0.40) = 1.33.

The market convention of assuming R = 40% for everything does not solve this.
It merely fixes one unknown by fiat, and it does so with a number that is
demonstrably wrong for whole categories of issuer -- most obviously bank
holding company senior debt, which is explicitly designed to absorb losses in
resolution.

This module supplies R from OUTSIDE the CDS market, using issuer fundamentals.
That breaks the degeneracy without circularity: nothing here reads a spread.

The consequence that matters
----------------------------
With a uniform R across names, h = s/(1-R) is a monotone transform of the
spread, so the implied default-probability RANKING is invariant -- recovery
moves levels only. With name-specific R that invariance breaks, and two issuers
at identical spreads can imply materially different default probabilities.

That is the analytical point of the exercise, not a side effect.

Model form
----------
A logit-linear scorecard:

    z = b0 + b_sen[seniority] + b_tang*(tangibility - 0.5)
           + b_cush*debt_cushion - b_lev*(leverage - 3)/10 + b_macro*cycle
    R = 1 / (1 + exp(-z))

The logit keeps R strictly inside (0, 1) for any input, which matters because
the bootstrap divides by (1 - R).

HONEST STATUS: coefficients are CALIBRATED TO PUBLISHED ANCHORS, not estimated
from a default database. This is an expert scorecard with a defensible
functional form, not a fitted statistical model. `RecoveryModel.fit()` is the
path to a real estimate once a recovery dataset is available; until then, any
write-up must describe this as calibrated rather than estimated. Overstating
it is the fastest way to lose credibility on this work.

Anchors used (senior unsecured corporate ~40% is the ISDA convention; the
others are set relative to it in the ordering the published recovery studies
consistently report):

    secured bank debt        ~65%
    senior unsecured         ~40%
    bank holdco senior       ~25%   (TLAC / bail-in layer)
    subordinated             ~20%
    junior subordinated      ~10%
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- constants

SENIORITY_SCORE = {
    "secured": 1.05,
    "senior_secured": 1.05,
    "senior_unsecured": 0.0,
    "holdco_senior": -0.75,
    "subordinated": -1.05,
    "junior_subordinated": -1.85,
}

#: Asset tangibility proxy by sector, 0 (all intangible) to 1 (all hard assets).
#: Drives how much is left to distribute in a liquidation.
SECTOR_TANGIBILITY = {
    "utilities": 0.85, "energy": 0.75, "real_estate": 0.80, "transport": 0.75,
    "materials": 0.70, "industrials": 0.60, "telecom": 0.55, "autos": 0.55,
    "consumer_staples": 0.50, "healthcare": 0.45, "retail": 0.45,
    "consumer_discretionary": 0.40, "media": 0.35, "banks": 0.30,
    "insurance": 0.35, "diversified_financials": 0.30, "software": 0.20,
    "services": 0.30, "pharma": 0.40, "other": 0.45,
}

_B0 = -0.405          # anchors senior unsecured at neutral covariates to ~40%
_B_TANG = 1.30
_B_CUSH = 1.10
_B_LEV = 0.55
_B_MACRO = 0.85


@dataclass
class IssuerProfile:
    """Fundamental inputs for one issuer. Everything optional except name.

    debt_cushion : fraction of total debt that ranks JUNIOR to the reference
        obligation. Junior debt absorbs losses first, so a thick cushion lifts
        recovery on the senior claim. In [0, 1].
    leverage : net debt / EBITDA. Higher leverage means less asset value per
        unit of claim. Neutral is 3.0.
    cycle : macro state in [-1, 1]. Negative is recessionary. Recoveries fall
        in downturns -- recovery and default are positively correlated, which
        is precisely when it hurts.
    """

    name: str
    sector: str = "other"
    seniority: str = "senior_unsecured"
    debt_cushion: float = 0.0
    leverage: float | None = None
    tangibility: float | None = None
    cycle: float = 0.0
    is_bank: bool = False
    notes: str = ""

    def __post_init__(self):
        if self.seniority not in SENIORITY_SCORE:
            raise ValueError(
                f"unknown seniority {self.seniority!r}; expected one of "
                f"{sorted(SENIORITY_SCORE)}"
            )
        if not 0.0 <= self.debt_cushion <= 1.0:
            raise ValueError("debt_cushion must be in [0, 1]")
        if not -1.0 <= self.cycle <= 1.0:
            raise ValueError("cycle must be in [-1, 1]")
        if self.tangibility is not None and not 0.0 <= self.tangibility <= 1.0:
            raise ValueError("tangibility must be in [0, 1]")
        if self.sector not in SECTOR_TANGIBILITY and self.tangibility is None:
            raise ValueError(
                f"unknown sector {self.sector!r} and no tangibility override; "
                f"expected one of {sorted(SECTOR_TANGIBILITY)}"
            )
        if self.is_bank and self.seniority == "senior_unsecured":
            # Single-name bank CDS references the HOLDING COMPANY, whose senior
            # unsecured is the TLAC / bail-in layer. Silently treating it as
            # ordinary corporate senior is the single largest error this module
            # exists to prevent.
            self.seniority = "holdco_senior"
            self.notes = (self.notes + " | seniority auto-set to holdco_senior "
                          "(bank CDS references the holdco; senior there is TLAC)").strip(" |")


@dataclass
class RecoveryEstimate:
    name: str
    point: float
    low: float
    high: float
    drivers: dict = field(default_factory=dict)

    @property
    def lgd(self) -> float:
        return 1.0 - self.point

    def __str__(self) -> str:
        d = "  ".join(f"{k} {v:+.3f}" for k, v in self.drivers.items())
        return (f"{self.name:<22} R = {self.point:.1%}  "
                f"[{self.low:.1%} – {self.high:.1%}]   {d}")


def _logistic(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


class RecoveryModel:
    """Logit-linear recovery scorecard.

    Parameters
    ----------
    band : half-width of the uncertainty interval, in logit units. The default
        of 0.55 produces roughly a +/- 12pt band around a 40% central estimate,
        which is deliberately wide -- published recovery distributions are
        broad and bimodal, and a narrow band here would be false precision.
    """

    def __init__(self, band: float = 0.55, coefficients: Mapping | None = None):
        c = dict(b0=_B0, tang=_B_TANG, cush=_B_CUSH, lev=_B_LEV, macro=_B_MACRO)
        if coefficients:
            c.update(coefficients)
        self.c = c
        self.band = float(band)
        self.fitted = False          # True only after fit() on real data

    # ------------------------------------------------------------- predict

    def score(self, p: IssuerProfile) -> tuple[float, dict]:
        tang = p.tangibility if p.tangibility is not None else SECTOR_TANGIBILITY[p.sector]
        lev = 3.0 if p.leverage is None else p.leverage
        c = self.c
        parts = {
            "seniority": SENIORITY_SCORE[p.seniority],
            "tangibility": c["tang"] * (tang - 0.5),
            "cushion": c["cush"] * p.debt_cushion,
            "leverage": -c["lev"] * (lev - 3.0) / 10.0,
            "cycle": c["macro"] * p.cycle,
        }
        return c["b0"] + sum(parts.values()), parts

    def predict(self, p: IssuerProfile) -> RecoveryEstimate:
        z, parts = self.score(p)
        return RecoveryEstimate(
            name=p.name,
            point=float(_logistic(z)),
            low=float(_logistic(z - self.band)),
            high=float(_logistic(z + self.band)),
            drivers=parts,
        )

    def predict_many(self, profiles) -> dict[str, RecoveryEstimate]:
        return {p.name: self.predict(p) for p in profiles}

    def to_frame(self, profiles) -> pd.DataFrame:
        rows = []
        for p in profiles:
            e = self.predict(p)
            rows.append({
                "name": p.name, "sector": p.sector, "seniority": p.seniority,
                "cushion": p.debt_cushion, "R_low": e.low, "R": e.point,
                "R_high": e.high, "LGD": e.lgd,
            })
        return pd.DataFrame(rows).sort_values("R", ascending=False).reset_index(drop=True)

    # ---------------------------------------------------------------- fit

    def fit(self, observations, l2: float = 1.0) -> "RecoveryModel":
        """Estimate coefficients from realised recoveries.

        `observations` is an iterable of (IssuerProfile, realised_recovery).
        Fits by least squares in logit space with ridge regularisation, which
        is appropriate for the small, noisy samples recovery data comes in.

        Until this is called on a real dataset, the model is CALIBRATED, not
        ESTIMATED, and `fitted` stays False. Say so in any write-up.
        """
        obs = list(observations)
        if len(obs) < 12:
            raise ValueError(
                f"need at least 12 observations to fit; got {len(obs)}. "
                "With fewer, the calibrated defaults are more defensible."
            )
        X, y = [], []
        for p, r in obs:
            if not 0.0 < r < 1.0:
                raise ValueError("realised recovery must be strictly in (0, 1)")
            _, parts = self.score(p)
            tang = p.tangibility if p.tangibility is not None else SECTOR_TANGIBILITY[p.sector]
            lev = 3.0 if p.leverage is None else p.leverage
            X.append([1.0, SENIORITY_SCORE[p.seniority], tang - 0.5,
                      p.debt_cushion, -(lev - 3.0) / 10.0, p.cycle])
            y.append(np.log(r / (1.0 - r)))
        X, y = np.asarray(X), np.asarray(y)
        A = X.T @ X + l2 * np.eye(X.shape[1])
        beta = np.linalg.solve(A, X.T @ y)

        self.c = dict(b0=float(beta[0]), tang=float(beta[2]), cush=float(beta[3]),
                      lev=float(beta[4]), macro=float(beta[5]))
        for k in SENIORITY_SCORE:
            SENIORITY_SCORE[k] *= float(beta[1])
        resid = y - X @ beta
        self.band = float(np.std(resid))
        self.fitted = True
        return self


# ------------------------------------------------------------- convenience

def bank_profile(name: str, cushion: float = 0.0, cycle: float = 0.0) -> IssuerProfile:
    """A US bank holding company. Seniority resolves to the TLAC layer."""
    return IssuerProfile(name=name, sector="banks", is_bank=True,
                         debt_cushion=cushion, cycle=cycle)


def recovery_map(model: RecoveryModel, profiles) -> dict[str, float]:
    """{name: R} ready to pass straight into compare() or bootstrap()."""
    return {p.name: model.predict(p).point for p in profiles}

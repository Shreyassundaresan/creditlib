"""Twenty-name cross-section with model-predicted recovery.

Edit the UNIVERSE block. Everything else derives.
"""
from datetime import date
import pandas as pd
from creditlib import compare, RecoveryModel, IssuerProfile, recovery_map

pd.set_option("display.width", 160)

# ======================== EDIT THIS BLOCK ONLY ============================
# name, 5Y spread (bp), sector, seniority, debt cushion, leverage
UNIVERSE = [
    ("JPMorgan",              58,  "banks",       "senior_unsecured", 0.22, None),
    ("Bank of America",       63,  "banks",       "senior_unsecured", 0.20, None),
    ("U.S. Bancorp",          67,  "banks",       "senior_unsecured", 0.12, None),
    ("Wells Fargo",           72,  "banks",       "senior_unsecured", 0.18, None),
    ("Citigroup",             85,  "banks",       "senior_unsecured", 0.25, None),
    ("Duke Energy",           62,  "utilities",   "senior_unsecured", 0.05, 4.8),
    ("Exxon Mobil",           41,  "energy",      "senior_unsecured", 0.02, 1.1),
    ("Prologis",              57,  "real_estate", "secured",          0.15, 5.2),
    ("Oracle",                89,  "software",    "senior_unsecured", 0.05, 3.6),
    ("Macy's",               395,  "retail",      "senior_unsecured", 0.04, 3.4),
]
CYCLE = 0.0          # -1 recession ... +1 expansion
TRADE = date(2026, 8, 17)
# ==========================================================================

profiles = [
    IssuerProfile(n, sector=sec, seniority=sen, debt_cushion=c, leverage=lev,
                  is_bank=(sec == "banks"), cycle=CYCLE)
    for n, _, sec, sen, c, lev in UNIVERSE
]
spreads = {n: s for n, s, *_ in UNIVERSE}

model = RecoveryModel()
print("PREDICTED RECOVERY")
print(model.to_frame(profiles).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print(f"\nmodel status: {'ESTIMATED' if model.fitted else 'CALIBRATED (not fitted on data)'}")

R = recovery_map(model, profiles)
flat = compare(spreads, recovery=0.40, trade_date=TRADE)
mod = compare(spreads, recovery=R, trade_date=TRADE)

j = (flat.table[["name", "spread_5y_bp", "PD_5y"]].rename(columns={"PD_5y": "PD_flat40"})
     .merge(mod.table[["name", "recovery", "PD_5y"]].rename(columns={"PD_5y": "PD_model"}),
            on="name"))
j["rank_flat"] = j.PD_flat40.rank().astype(int)
j["rank_model"] = j.PD_model.rank().astype(int)
j["move"] = j.rank_flat - j.rank_model
j = j.sort_values("PD_model").reset_index(drop=True)

print("\n\nFLAT 40% vs MODEL RECOVERY")
print(j.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(f"\n{(j['move'] != 0).sum()} of {len(j)} names change rank; "
      f"largest move {j['move'].abs().max()} places")
print("\nCAVEATS: recovery is model output from calibrated (not estimated) coefficients; "
      "\nimplied probabilities are risk-neutral, roughly 10x historical default rates.")

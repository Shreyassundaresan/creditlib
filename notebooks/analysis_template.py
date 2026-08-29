"""Analysis template -- copy this, change the top block, run.

The whole point: one place to edit, everything else derived.
"""
from datetime import date
import matplotlib.pyplot as plt
from creditlib import analyze, compare, recovery_sensitivity

# ============================= EDIT THIS ONLY =============================
NAMES = {                       # spreads in BASIS POINTS
    "JPMorgan":        58,
    "Bank of America": 63,
    "U.S. Bancorp":    67,
    "Wells Fargo":     72,
    "Citigroup":       85,
}
RECOVERY  = 0.40
TRADE     = date(2026, 8, 17)
FOCUS     = "Citigroup"         # name to drill into
# ==========================================================================

# --- cross-section --------------------------------------------------------
x = compare(NAMES, recovery=RECOVERY, trade_date=TRADE)
print(x)

# --- one name in depth ----------------------------------------------------
one = x.analyses[FOCUS]
print("\n" + "=" * 72)
print(one)

# --- recovery sensitivity -------------------------------------------------
print("\nImplied 5Y PD across recovery assumptions:")
print(recovery_sensitivity(NAMES, recoveries=(0.40, 0.25, 0.10))
      .to_string(float_format=lambda v: f"{v:.2%}"))

# --- markdown for your memo ----------------------------------------------
print("\n" + "=" * 72)
print(x.report())
print(one.report())

# --- charts ---------------------------------------------------------------
x.plot(); plt.tight_layout(); plt.show()
one.plot(); plt.tight_layout(); plt.show()

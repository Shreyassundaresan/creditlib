# creditlib

**A single-name CDS pricing engine built from first principles.** Bootstraps a hazard-rate
term structure from quoted CDS spreads, prices both legs by exact closed-form integration,
and computes CS01 / IR DV01 / Rec01 / JTD.

Every formula is derived before it is coded — see [`docs/creditlib_documentation.pdf`](docs/creditlib_documentation.pdf)
(25pp). Every component is validated against no-arbitrage conditions and, where a comparable
object exists, against QuantLib.

```
104 tests passing
QuantLib parity      1e-12 relative
bootstrap repricing  4.4e-16 relative  (two ULP of a float64)
```

**Try it without installing anything** — open these in any browser:

| | |
|---|---|
| [`docs/cds_from_scratch.html`](docs/cds_from_scratch.html) | **Start here.** Full course from zero — 7 chapters, 53 units, ~75 min |
| [`docs/course.html`](docs/course.html) | Shorter engine-focused walkthrough, ~40 min |
| [`docs/desk.html`](docs/desk.html) | Interactive desk — add firms, edit spreads, paste from Excel |
| [`docs/learn.html`](docs/learn.html) | Reference drill: 15 modules + glossary |


## Build order
- [x] **Step 1 — Curves.** Piecewise-flat hazard curve, no-arbitrage checks, QuantLib parity.
- [x] **Step 2 — Discount curve.** Log-linear DF interpolation, compounding conventions, forward replication, QuantLib LogLinear parity.
- [x] **Step 3 — Schedules.** IMM roll dates, ACT/360 accruals, seasoned stubs, accrued rebate.
- [x] **Step 4 — CDS valuation.** Both legs by exact closed-form integration, par spread, CS01 / IR DV01 / Rec01 / JTD, QuantLib parity.
- [x] **Step 5 — Bootstrap.** Triangular forward substitution, Brent per tenor, arbitrage diagnostics, exact repricing, QuantLib parity.
- [x] **Step 6 — Endogenous recovery.** Fundamentals-based recovery model replacing the flat 40% convention; per-name recovery through the bootstrap. See `docs/recovery_design_spec.pdf` and `docs/recovery_findings.pptx`.
- [ ] Step 7 — Fit the recovery model on auction settlement data; senior/sub basis for market-implied recovery.


## Run it in Google Colab (no local setup)

Open `notebooks/creditlib_colab.ipynb` in Colab and run top to bottom. It handles upload,
install, verification, and a live worked example. Colab already has numpy/scipy/pandas/matplotlib;
only QuantLib is installed, and it is validation-only.

## Install and run locally

```bash
# 1. get a virtualenv (any Python >= 3.10)
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 2. install the library, editable, plus test + notebook extras
pip install -e ".[test,notebook]"

# 3. verify -- should print 104 passed
pytest -q

# 4. see it work
python quickstart.py

# 5. the applied analysis
jupyter notebook notebooks/bank_credit_analysis.ipynb
```

`pip install -e .` puts `creditlib` on the path from any directory, so no `PYTHONPATH` juggling.

QuantLib is a **validation-only** dependency, never on the pricing path. If it fails to
install (it has no wheel on some platforms), everything still works — the ~8 parity tests
skip cleanly and you'll see `96 passed, 8 skipped`.

### Minimal usage

```python
from datetime import date
from creditlib import DiscountCurve, bootstrap_survival_curve, make_cds_schedule, price_cds

discount = DiscountCurve.from_zero_rates([1, 3, 5, 7, 10],
                                         [0.0402, 0.0374, 0.0372, 0.0378, 0.0390])

boot = bootstrap_survival_curve(
    spreads={1.0: 0.0055, 3.0: 0.0082, 5.0: 0.0105, 7.0: 0.0118, 10.0: 0.0130},
    discount_curve=discount,
    recovery=0.40,
    trade_date=date(2026, 8, 17),
)
print(boot)                       # hazards, survival, repricing error

sched = make_cds_schedule(date(2026, 8, 17), tenor_years=5.0)
print(price_cds(discount, boot.curve, 0.0100, 0.40, sched, notional=10_000_000))
```

## Learn it / analyse with it

- **`docs/desk.html`** — interactive browser desk. Add firms, edit spreads, drag recovery,
  watch curves and rankings recompute live. Exports markdown / CSV / Python. Self-validates
  its JavaScript pricing core against the Python engine on every load — if the badge isn't
  green, don't trust the page.

- **`docs/learn.html`** — open in any browser. 15 interactive modules with a prediction
  gate before every answer, plus a 27-term glossary. Works offline, saves your place.
- **`notebooks/analysis_template.py`** — edit the top block, run, get tables + markdown + charts.

```python
from creditlib import analyze, compare

compare({"JPM": 58, "BAC": 63, "C": 85})          # spreads in bp
analyze({1: 55, 3: 82, 5: 105}, name="Acme")      # full term structure
```

## Layout
```
creditlib/
  daycount.py    day count conventions
  curves.py         PiecewiseFlatCurve -> CreditCurve
  discount_curve.py DiscountCurve (log-linear on DFs)
  schedule.py       IMM schedules, ACT/360 accruals
  cds.py            legs, price_cds, par_spread, risk measures
  bootstrap.py      survival curve from quoted spreads
  recovery.py       fundamentals-based recovery prediction
  analyze.py        one-line analysis API
  validation.py  runtime no-arbitrage assertions
tests/
  test_curves.py
  test_discount_curve.py
  test_cds.py
  test_bootstrap.py
  test_recovery.py
docs/
  creditlib_documentation.pdf   full derivations, validation, application
  creditlib_documentation.tex
notebooks/
```



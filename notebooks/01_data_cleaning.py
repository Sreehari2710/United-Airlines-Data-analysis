"""
01_data_cleaning.py

Step 1 of the United Airlines Flight Delay Analysis project.

What this does:
1. Loads the raw 3M-row flight dataset (all carriers).
2. Filters it down to United Airlines (AIRLINE_CODE == "UA") flights only.
3. Cleans and standardizes key columns (dates, delay causes, cancellations).
4. Saves a clean, analysis-ready CSV to data/processed/united_flights.csv.

Run from the project root:
    python notebooks/01_data_cleaning.py

(This is written as a plain script so it runs without Jupyter installed.
 It can be pasted into a notebook cell-by-cell later if you prefer working
 in Jupyter.)
"""

import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "flights_sample_3m.csv"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "united_flights.csv"

# ---------------------------------------------------------------------------
# 1. Load raw data
# ---------------------------------------------------------------------------
print(f"Loading raw data from {RAW_PATH} ...")
df = pd.read_csv(RAW_PATH)
print(f"Loaded {len(df):,} total flight records (all airlines).")

# ---------------------------------------------------------------------------
# 2. Filter to United Airlines only
# ---------------------------------------------------------------------------
ua = df[df["AIRLINE_CODE"] == "UA"].copy()
print(f"Filtered to {len(ua):,} United Airlines flights.")

# ---------------------------------------------------------------------------
# 3. Clean and standardize columns
# ---------------------------------------------------------------------------

# Parse the flight date properly
ua["FL_DATE"] = pd.to_datetime(ua["FL_DATE"])

# Add convenience columns for later analysis
ua["YEAR"] = ua["FL_DATE"].dt.year
ua["MONTH"] = ua["FL_DATE"].dt.month
ua["DAY_OF_WEEK"] = ua["FL_DATE"].dt.day_name()

# Delay-cause columns are NaN when there was no delay attributable to that
# cause (not "unknown") -> fill with 0 so they can be summed/aggregated.
delay_cause_cols = [
    "DELAY_DUE_CARRIER",
    "DELAY_DUE_WEATHER",
    "DELAY_DUE_NAS",
    "DELAY_DUE_SECURITY",
    "DELAY_DUE_LATE_AIRCRAFT",
]
ua[delay_cause_cols] = ua[delay_cause_cols].fillna(0)

# Flag whether a flight was "significantly" delayed (industry-standard
# threshold used by BTS/FAA: 15+ minutes late on arrival).
ua["IS_DELAYED_15"] = ua["ARR_DELAY"] >= 15

# Cancelled/diverted flights won't have arrival delay data - keep them,
# but make it explicit so later analysis doesn't silently drop or
# misinterpret them.
ua["CANCELLED"] = ua["CANCELLED"].astype(bool)
ua["DIVERTED"] = ua["DIVERTED"].astype(bool)

# Quick data-quality check
print("\nMissing values in key columns after cleaning:")
print(
    ua[
        ["ARR_DELAY", "DEP_DELAY", "CANCELLED", "DIVERTED"] + delay_cause_cols
    ].isna().sum()
)

# ---------------------------------------------------------------------------
# 4. Save cleaned dataset
# ---------------------------------------------------------------------------
PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
ua.to_csv(PROCESSED_PATH, index=False)
print(f"\nSaved cleaned United Airlines dataset to {PROCESSED_PATH}")
print(f"Rows: {len(ua):,} | Columns: {len(ua.columns)}")
print(f"Date range: {ua['FL_DATE'].min().date()} to {ua['FL_DATE'].max().date()}")

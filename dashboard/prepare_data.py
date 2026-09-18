"""
prepare_data.py

Precomputes all the aggregated numbers the interactive dashboard needs,
and saves them as a single JSON file the dashboard's JavaScript reads.

Why precompute instead of loading the raw CSV into the browser? The
cleaned dataset has ~250,000 rows -- far too much to ship to a webpage.
Instead, we do all the heavy aggregation here in Python (same techniques
as notebooks/02_exploratory_analysis.py), and only send the browser the
small, already-summarized numbers it needs to draw charts.

Run from the project root:
    python dashboard/prepare_data.py
"""

import json
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "united_flights.csv"
OUTPUT_JSON_PATH = Path(__file__).resolve().parent / "data.json"
OUTPUT_JS_PATH = Path(__file__).resolve().parent / "data.js"

DELAY_CAUSE_COLS = [
    "DELAY_DUE_CARRIER",
    "DELAY_DUE_WEATHER",
    "DELAY_DUE_NAS",
    "DELAY_DUE_SECURITY",
    "DELAY_DUE_LATE_AIRCRAFT",
]

print(f"Loading {PROCESSED_PATH} ...")
df = pd.read_csv(PROCESSED_PATH, parse_dates=["FL_DATE"])
flown = df[~df["CANCELLED"] & ~df["DIVERTED"]].copy()

years = sorted(df["YEAR"].unique().tolist())


def overview_stats(flown_subset: pd.DataFrame, full_subset: pd.DataFrame) -> dict:
    return {
        "total_flights": int(len(full_subset)),
        "avg_delay": round(float(flown_subset["ARR_DELAY"].mean()), 1),
        "pct_delayed_15": round(float(flown_subset["IS_DELAYED_15"].mean() * 100), 1),
        "cancellation_rate": round(float(full_subset["CANCELLED"].mean() * 100), 2),
    }


def airport_delay_table(flown_subset: pd.DataFrame, top_n: int = 15) -> list:
    # Same 200-flight minimum used in notebooks/02_exploratory_analysis.py,
    # so the dashboard's "worst airport" matches the README's findings
    # rather than surfacing a different airport due to a looser threshold.
    stats = (
        flown_subset.groupby("ORIGIN")["ARR_DELAY"]
        .agg(avg_delay="mean", flights="count")
        .query("flights >= 200")
        .sort_values("avg_delay", ascending=False)
        .head(top_n)
        .reset_index()
    )
    return [
        {"origin": r.ORIGIN, "avg_delay": round(r.avg_delay, 1), "flights": int(r.flights)}
        for r in stats.itertuples()
    ]


def monthly_delay_table(flown_subset: pd.DataFrame) -> list:
    monthly = flown_subset.groupby("MONTH")["ARR_DELAY"].mean().reindex(range(1, 13))
    return [
        {"month": m, "avg_delay": round(float(v), 1) if pd.notna(v) else None}
        for m, v in monthly.items()
    ]


def cause_breakdown_table(flown_subset: pd.DataFrame) -> list:
    totals = flown_subset[DELAY_CAUSE_COLS].sum()
    total_sum = totals.sum()
    result = []
    for col in DELAY_CAUSE_COLS:
        name = col.replace("DELAY_DUE_", "").replace("_", " ").title()
        minutes = float(totals[col])
        pct = round(minutes / total_sum * 100, 1) if total_sum > 0 else 0
        result.append({"cause": name, "minutes": round(minutes), "pct": pct})
    return sorted(result, key=lambda x: -x["minutes"])


def route_table(flown_subset: pd.DataFrame, top_n: int = 10) -> list:
    # Same 100-flight minimum used in sql/queries.sql, for consistency.
    stats = (
        flown_subset.groupby(["ORIGIN", "DEST"])["ARR_DELAY"]
        .agg(avg_delay="mean", flights="count")
        .query("flights >= 100")
        .sort_values("avg_delay", ascending=False)
        .head(top_n)
        .reset_index()
    )
    return [
        {
            "route": f"{r.ORIGIN} → {r.DEST}",
            "avg_delay": round(r.avg_delay, 1),
            "flights": int(r.flights),
        }
        for r in stats.itertuples()
    ]


# ---------------------------------------------------------------------------
# Build the "ALL" (all years combined) view plus one view per individual year
# ---------------------------------------------------------------------------
dashboard_data = {"years": years, "by_selection": {}}

selections = ["ALL"] + years
for sel in selections:
    if sel == "ALL":
        flown_subset = flown
        full_subset = df
    else:
        flown_subset = flown[flown["YEAR"] == sel]
        full_subset = df[df["YEAR"] == sel]

    dashboard_data["by_selection"][str(sel)] = {
        "overview": overview_stats(flown_subset, full_subset),
        "airport_delay": airport_delay_table(flown_subset),
        "monthly_delay": monthly_delay_table(flown_subset),
        "cause_breakdown": cause_breakdown_table(flown_subset),
        "routes": route_table(flown_subset),
    }

# Yearly trend line is always shown in full, regardless of the year filter
yearly_trend = (
    flown.groupby("YEAR")
    .agg(avg_delay=("ARR_DELAY", "mean"), pct_delayed_15=("IS_DELAYED_15", "mean"))
    .reset_index()
)
dashboard_data["yearly_trend"] = [
    {
        "year": int(r.YEAR),
        "avg_delay": round(r.avg_delay, 1),
        "pct_delayed_15": round(r.pct_delayed_15 * 100, 1),
    }
    for r in yearly_trend.itertuples()
]

json_text = json.dumps(dashboard_data, indent=2)

# Save the plain JSON (handy for inspecting/debugging the numbers directly).
OUTPUT_JSON_PATH.write_text(json_text)

# Also save it as a small JavaScript file that just assigns the same data
# to a global variable. index.html loads THIS via a normal <script> tag
# instead of fetch("data.json") -- browsers block fetch() of local files
# opened directly (file://) as a security restriction, but a <script src>
# tag loads fine either way, so this lets the dashboard work by simply
# double-clicking index.html, with no local server required.
OUTPUT_JS_PATH.write_text(f"const DASHBOARD_DATA = {json_text};")

print(f"Dashboard data saved to {OUTPUT_JSON_PATH}")
print(f"Dashboard data (script form) saved to {OUTPUT_JS_PATH}")
print(f"Years covered: {years}")

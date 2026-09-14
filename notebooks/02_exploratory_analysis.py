"""
02_exploratory_analysis.py

Step 2 of the United Airlines Flight Delay Analysis project.

What this does:
Answers the core business questions from the README using the cleaned
dataset produced by 01_data_cleaning.py:

1. Which airports have the worst average arrival delays for UA?
2. Which months/seasons are worst for UA delays?
3. What's the breakdown of delay-minutes by cause (carrier/weather/NAS/
   security/late-aircraft)?
4. Is UA's on-time performance trending better or worse over time?

Each answer is printed as a summary table AND saved as a chart in
notebooks/figures/, ready to drop into the README's "Key Findings"
section or the dashboard.

Run from the project root:
    python notebooks/02_exploratory_analysis.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "united_flights.csv"
FIG_DIR = PROJECT_ROOT / "notebooks" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

print(f"Loading cleaned data from {PROCESSED_PATH} ...")
df = pd.read_csv(PROCESSED_PATH, parse_dates=["FL_DATE"])
print(f"Loaded {len(df):,} United Airlines flights.\n")

# Non-cancelled, non-diverted flights only, for delay-minute analysis
# (cancelled/diverted flights have no meaningful arrival delay value)
flown = df[~df["CANCELLED"] & ~df["DIVERTED"]].copy()

# ---------------------------------------------------------------------------
# Q1: Which airports have the worst average arrival delays?
# ---------------------------------------------------------------------------
print("=" * 70)
print("Q1: Worst origin airports by average arrival delay (min 200 flights)")
print("=" * 70)

airport_delay = (
    flown.groupby("ORIGIN")["ARR_DELAY"]
    .agg(avg_delay="mean", flights="count")
    .query("flights >= 200")
    .sort_values("avg_delay", ascending=False)
)
top_worst_airports = airport_delay.head(10)
print(top_worst_airports)

fig, ax = plt.subplots()
sns.barplot(
    x=top_worst_airports["avg_delay"],
    y=top_worst_airports.index,
    hue=top_worst_airports.index,
    palette="Reds_r",
    legend=False,
    ax=ax,
)
ax.set_title("Top 10 Worst UA Origin Airports by Avg. Arrival Delay")
ax.set_xlabel("Average Arrival Delay (minutes)")
ax.set_ylabel("Origin Airport")
fig.tight_layout()
fig.savefig(FIG_DIR / "01_worst_airports.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Q2: Which months are worst for delays?
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Q2: Average arrival delay by month")
print("=" * 70)

monthly_delay = flown.groupby("MONTH")["ARR_DELAY"].mean().sort_index()
print(monthly_delay)

fig, ax = plt.subplots()
sns.lineplot(x=monthly_delay.index, y=monthly_delay.values, marker="o", ax=ax)
ax.set_title("UA Average Arrival Delay by Month (2019-2023 combined)")
ax.set_xlabel("Month")
ax.set_ylabel("Average Arrival Delay (minutes)")
ax.set_xticks(range(1, 13))
fig.tight_layout()
fig.savefig(FIG_DIR / "02_monthly_delay_trend.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Q3: Delay-minutes breakdown by cause
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Q3: Total delay-minutes by cause")
print("=" * 70)

delay_cause_cols = [
    "DELAY_DUE_CARRIER",
    "DELAY_DUE_WEATHER",
    "DELAY_DUE_NAS",
    "DELAY_DUE_SECURITY",
    "DELAY_DUE_LATE_AIRCRAFT",
]
cause_totals = flown[delay_cause_cols].sum().sort_values(ascending=False)
cause_totals.index = [c.replace("DELAY_DUE_", "") for c in cause_totals.index]
print(cause_totals)
print(f"\nAs % of total attributed delay minutes:")
print((cause_totals / cause_totals.sum() * 100).round(1))

fig, ax = plt.subplots()
sns.barplot(
    x=cause_totals.values,
    y=cause_totals.index,
    hue=cause_totals.index,
    palette="Blues_r",
    legend=False,
    ax=ax,
)
ax.set_title("Total UA Delay-Minutes by Cause (2019-2023)")
ax.set_xlabel("Total Delay Minutes")
ax.set_ylabel("Delay Cause")
fig.tight_layout()
fig.savefig(FIG_DIR / "03_delay_cause_breakdown.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Q4: Is on-time performance improving or worsening over time?
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Q4: On-time performance trend by year")
print("=" * 70)

yearly = flown.groupby("YEAR").agg(
    avg_delay=("ARR_DELAY", "mean"),
    pct_delayed_15=("IS_DELAYED_15", "mean"),
    flights=("ARR_DELAY", "count"),
)
yearly["pct_delayed_15"] = (yearly["pct_delayed_15"] * 100).round(1)
print(yearly)

fig, ax = plt.subplots()
sns.lineplot(x=yearly.index, y=yearly["pct_delayed_15"], marker="o", ax=ax)
ax.set_title("UA % of Flights Delayed 15+ Minutes, by Year")
ax.set_xlabel("Year")
ax.set_ylabel("% of Flights Delayed 15+ Min")
fig.tight_layout()
fig.savefig(FIG_DIR / "04_yearly_ontime_trend.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Bonus: Cancellation rate
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Bonus: Cancellation rate")
print("=" * 70)
cancel_rate = df["CANCELLED"].mean() * 100
print(f"Overall UA cancellation rate: {cancel_rate:.2f}%")

print(f"\nAll charts saved to {FIG_DIR}")

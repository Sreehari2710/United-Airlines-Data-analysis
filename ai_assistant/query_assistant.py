"""
query_assistant.py

AI Feature 2 of the United Airlines Flight Delay Analysis project:
a natural-language data assistant. A user can type a question in plain
English (e.g. "What's UA's worst month for delays at ORD?") and get a
real answer computed from the actual cleaned dataset -- no need to know
pandas or SQL.

How it works (by design, runs fully offline -- no API key required):
1. The question is matched against a set of known "intents" (question
   patterns) using simple keyword/regex matching to extract entities
   like airport codes, months, and years.
2. Each intent maps to a real pandas computation against the cleaned
   dataset (the exact same computations from notebooks/02 and sql/queries.sql).
3. The result is formatted back into a plain-English sentence.

This is a lightweight, transparent "text-to-query" system -- the kind of
rule-based NLU that's genuinely used in production before reaching for a
full LLM. It is deliberately built to run with zero external dependencies
or API keys, so it's always reproducible and free to run.

Optional upgrade path (not required to run this script):
If an OPENAI_API_KEY environment variable is present, `parse_with_llm()`
demonstrates how a real LLM call could replace the regex-based intent
matching for much more flexible, free-form question understanding --
see the "Optional AI Extension" section near the bottom.

Run interactively from the project root:
    python ai_assistant/query_assistant.py

Or import and use programmatically:
    from ai_assistant.query_assistant import answer_question
    print(answer_question("What's the worst airport for delays?"))
"""

import re
import os
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Load data once
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "united_flights.csv"

_df = None  # lazy-loaded so importing this module doesn't require the data


def _load_data() -> pd.DataFrame:
    global _df
    if _df is None:
        _df = pd.read_csv(PROCESSED_PATH, parse_dates=["FL_DATE"])
    return _df


MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

DELAY_CAUSE_COLS = [
    "DELAY_DUE_CARRIER",
    "DELAY_DUE_WEATHER",
    "DELAY_DUE_NAS",
    "DELAY_DUE_SECURITY",
    "DELAY_DUE_LATE_AIRCRAFT",
]


_valid_airports = None  # lazy-loaded set of real airport codes in the dataset


def _get_valid_airports() -> set[str]:
    global _valid_airports
    if _valid_airports is None:
        df = _load_data()
        _valid_airports = set(df["ORIGIN"].unique()) | set(df["DEST"].unique())
    return _valid_airports


def _extract_airport(question: str) -> str | None:
    """Find a real 3-letter airport code in the question, e.g. 'ORD', 'EWR'.

    A naive regex for any 3 capital letters would also match common English
    words like "THE" or "FOR" once the question is uppercased. Instead, we
    only accept a 3-letter token if it's an airport code that actually
    appears in the dataset -- avoids false positives.
    """
    valid = _get_valid_airports()
    for word in re.findall(r"\b[A-Za-z]{3}\b", question):
        if word.upper() in valid:
            return word.upper()
    return None


def _extract_month(question: str) -> int | None:
    """Find a month name in the question and return its number (1-12)."""
    q_lower = question.lower()
    for name, num in MONTH_NAMES.items():
        if name in q_lower:
            return num
    return None


def _extract_year(question: str) -> int | None:
    match = re.search(r"\b(20\d{2})\b", question)
    return int(match.group(1)) if match else None


# ---------------------------------------------------------------------------
# Intent handlers -- each one answers one category of question
# ---------------------------------------------------------------------------

def _intent_worst_airport(question: str) -> str:
    df = _load_data()
    flown = df[~df["CANCELLED"] & ~df["DIVERTED"]]
    stats = (
        flown.groupby("ORIGIN")["ARR_DELAY"]
        .agg(avg_delay="mean", flights="count")
        .query("flights >= 200")
        .sort_values("avg_delay", ascending=False)
    )
    worst = stats.iloc[0]
    return (
        f"The worst UA origin airport for delays is {stats.index[0]}, "
        f"averaging {worst['avg_delay']:.1f} minutes of arrival delay "
        f"across {int(worst['flights']):,} flights."
    )


def _intent_airport_delay(question: str, airport: str) -> str:
    df = _load_data()
    flown = df[~df["CANCELLED"] & ~df["DIVERTED"]]
    subset = flown[flown["ORIGIN"] == airport]
    if len(subset) == 0:
        return f"I couldn't find any UA flights departing from {airport} in this dataset."
    avg_delay = subset["ARR_DELAY"].mean()
    return (
        f"UA flights departing from {airport} average {avg_delay:.1f} minutes "
        f"of arrival delay, based on {len(subset):,} flights."
    )


def _intent_worst_month_at_airport(question: str, airport: str) -> str:
    df = _load_data()
    flown = df[~df["CANCELLED"] & ~df["DIVERTED"]]
    subset = flown[flown["ORIGIN"] == airport] if airport else flown
    if len(subset) == 0:
        return f"I couldn't find any UA flights departing from {airport} in this dataset."
    monthly = subset.groupby("MONTH")["ARR_DELAY"].mean().sort_values(ascending=False)
    worst_month_num = monthly.index[0]
    month_name = [k for k, v in MONTH_NAMES.items() if v == worst_month_num][0].capitalize()
    scope = f"at {airport}" if airport else "system-wide"
    return (
        f"UA's worst month for delays {scope} is {month_name}, "
        f"averaging {monthly.iloc[0]:.1f} minutes of arrival delay."
    )


def _intent_month_delay(question: str, month_num: int) -> str:
    df = _load_data()
    flown = df[~df["CANCELLED"] & ~df["DIVERTED"]]
    subset = flown[flown["MONTH"] == month_num]
    month_name = [k for k, v in MONTH_NAMES.items() if v == month_num][0].capitalize()
    avg_delay = subset["ARR_DELAY"].mean()
    return (
        f"In {month_name}, UA flights average {avg_delay:.1f} minutes of "
        f"arrival delay, based on {len(subset):,} flights across all years in the data."
    )


def _intent_delay_causes(question: str) -> str:
    df = _load_data()
    flown = df[~df["CANCELLED"] & ~df["DIVERTED"]]
    totals = flown[DELAY_CAUSE_COLS].sum().sort_values(ascending=False)
    top_cause = totals.index[0].replace("DELAY_DUE_", "").replace("_", " ").title()
    pct = totals.iloc[0] / totals.sum() * 100
    return (
        f"The biggest cause of UA delay-minutes is {top_cause}, accounting for "
        f"{pct:.1f}% of all attributed delay time."
    )


def _intent_year_trend(question: str, year: int | None) -> str:
    df = _load_data()
    flown = df[~df["CANCELLED"] & ~df["DIVERTED"]]
    if year:
        subset = flown[flown["YEAR"] == year]
        if len(subset) == 0:
            return f"I don't have UA flight data for {year}."
        pct_delayed = subset["IS_DELAYED_15"].mean() * 100
        return (
            f"In {year}, {pct_delayed:.1f}% of UA flights were delayed 15+ minutes, "
            f"based on {len(subset):,} flights."
        )
    yearly = flown.groupby("YEAR")["IS_DELAYED_15"].mean() * 100
    best_year = yearly.idxmin()
    worst_year = yearly.idxmax()
    return (
        f"UA's on-time performance varied by year: {int(best_year)} was best "
        f"({yearly.loc[best_year]:.1f}% delayed 15+ min) and {int(worst_year)} was "
        f"worst ({yearly.loc[worst_year]:.1f}% delayed 15+ min)."
    )


def _intent_cancellation_rate(question: str, airport: str | None) -> str:
    df = _load_data()
    if airport:
        subset = df[df["ORIGIN"] == airport]
        if len(subset) == 0:
            return f"I couldn't find any UA flights departing from {airport}."
        rate = subset["CANCELLED"].mean() * 100
        return f"UA's cancellation rate at {airport} is {rate:.2f}%, based on {len(subset):,} scheduled flights."
    rate = df["CANCELLED"].mean() * 100
    return f"UA's overall cancellation rate is {rate:.2f}%, based on {len(df):,} scheduled flights."


# ---------------------------------------------------------------------------
# Main routing logic: match the question to an intent
# ---------------------------------------------------------------------------

def answer_question(question: str) -> str:
    """Takes a plain-English question and returns a plain-English answer."""
    q = question.lower()
    airport = _extract_airport(question)
    month = _extract_month(question)
    year = _extract_year(question)

    if "cancel" in q:
        return _intent_cancellation_rate(question, airport)

    if "cause" in q or "why" in q:
        return _intent_delay_causes(question)

    if "worst month" in q or ("month" in q and "worst" in q):
        return _intent_worst_month_at_airport(question, airport)

    if month is not None:
        return _intent_month_delay(question, month)

    if "trend" in q or "improving" in q or "worsening" in q or "year" in q:
        return _intent_year_trend(question, year)

    if "worst airport" in q or ("worst" in q and airport is None):
        return _intent_worst_airport(question)

    if airport:
        return _intent_airport_delay(question, airport)

    return (
        "I'm not sure how to answer that yet. Try asking things like:\n"
        "  - 'What's the worst airport for delays?'\n"
        "  - 'What's the average delay at ORD?'\n"
        "  - 'What's UA's worst month for delays at ORD?'\n"
        "  - 'What causes the most delays?'\n"
        "  - 'Is UA's on-time performance improving?'\n"
        "  - 'What's the cancellation rate at EWR?'"
    )


# ---------------------------------------------------------------------------
# Optional AI Extension: swap in a real LLM to parse free-form questions
#
# This function is NOT called by default (no API key is required to run
# this script). It demonstrates how the same intent-matching system could
# be made far more flexible by using an LLM to classify the question into
# one of the same fixed set of intents, instead of regex/keyword matching --
# a common, safe pattern called "LLM as a router" that still keeps the
# actual data computation deterministic and auditable (the LLM never
# invents numbers -- it only picks which real pandas function to call).
# ---------------------------------------------------------------------------

def parse_with_llm(question: str) -> str:
    """Requires OPENAI_API_KEY to be set. Not used by default."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return (
            "No OPENAI_API_KEY found in the environment -- this is an optional "
            "extension. The rule-based answer_question() above works without it."
        )

    from openai import OpenAI  # imported here so it's only required if used

    client = OpenAI(api_key=api_key)
    prompt = f"""Classify this question about United Airlines flight delays into
exactly one of these intents: worst_airport, airport_delay, worst_month,
month_delay, delay_causes, year_trend, cancellation_rate, unknown.
Also extract any 3-letter airport code, month name, or year mentioned.
Respond with just the intent name.

Question: {question}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    intent = response.choices[0].message.content.strip()
    return f"LLM classified this as intent: '{intent}' (would then route to the matching function above)"


# ---------------------------------------------------------------------------
# Interactive CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("United Airlines Flight Delay Assistant")
    print("Ask a question in plain English, or type 'quit' to exit.")
    print("(Try: 'What's the worst airport for delays?')\n")

    # Demo run with a few example questions so this script proves itself
    # even without interactive input (e.g. when run non-interactively).
    demo_questions = [
        "What's the worst airport for delays?",
        "What's the average delay at ORD?",
        "What's UA's worst month for delays at EWR?",
        "What causes the most delays?",
        "Is UA's on-time performance improving?",
        "What's the cancellation rate at ORF?",
    ]
    print("--- Demo questions ---")
    for q in demo_questions:
        print(f"\nQ: {q}")
        print(f"A: {answer_question(q)}")

    print("\n--- Interactive mode (press Ctrl+C or type 'quit' to stop) ---")
    try:
        while True:
            user_q = input("\nYour question: ").strip()
            if user_q.lower() in ("quit", "exit", ""):
                break
            print(f"A: {answer_question(user_q)}")
    except (EOFError, KeyboardInterrupt):
        pass
    print("\nGoodbye!")

"""
load_to_sqlite.py

Loads the cleaned United Airlines dataset (data/processed/united_flights.csv)
into a local SQLite database so the project's analysis can be shown in SQL,
not just pandas.

Run from the project root:
    python sql/load_to_sqlite.py
"""

import sqlite3
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "united_flights.csv"
DB_PATH = PROJECT_ROOT / "sql" / "united_flights.db"

print(f"Loading {PROCESSED_PATH} ...")
df = pd.read_csv(PROCESSED_PATH, parse_dates=["FL_DATE"])

print(f"Writing {len(df):,} rows into SQLite database at {DB_PATH} ...")
conn = sqlite3.connect(DB_PATH)
df.to_sql("flights", conn, if_exists="replace", index=False)

# A couple of useful indexes for the queries in queries.sql
conn.execute("CREATE INDEX IF NOT EXISTS idx_origin ON flights(ORIGIN)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_year_month ON flights(YEAR, MONTH)")
conn.commit()
conn.close()

print("Done. Query it with:")
print(f'  sqlite3 "{DB_PATH}"')
print("or open sql/queries.sql against it from any SQL client / pandas.read_sql.")

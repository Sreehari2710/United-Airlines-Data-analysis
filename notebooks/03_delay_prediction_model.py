"""
03_delay_prediction_model.py

AI Feature 1 of the United Airlines Flight Delay Analysis project:
a machine learning model that predicts, before a flight departs, the
probability it will be delayed 15+ minutes on arrival.

What this does:
1. Loads the cleaned United Airlines dataset.
2. Builds features available *before departure* (route, airport, month,
   day of week, scheduled departure hour, distance) -- deliberately
   excluding anything that "leaks" the answer (e.g. actual delay minutes).
3. Trains a Random Forest classifier to predict IS_DELAYED_15.
4. Evaluates it (accuracy, precision/recall, ROC-AUC) on a held-out test set.
5. Plots and saves a feature-importance chart so predictions are
   explainable, not a black box.
6. Saves the trained model to disk for reuse.

Run from the project root:
    python notebooks/03_delay_prediction_model.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "united_flights.csv"
FIG_DIR = PROJECT_ROOT / "notebooks" / "figures"
MODEL_DIR = PROJECT_ROOT / "ai_assistant" / "models"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

print(f"Loading cleaned data from {PROCESSED_PATH} ...")
df = pd.read_csv(PROCESSED_PATH, parse_dates=["FL_DATE"])

# Only flights that actually departed and arrived can have a meaningful
# delay outcome to predict.
df = df[~df["CANCELLED"] & ~df["DIVERTED"]].copy()
print(f"Using {len(df):,} flown UA flights for modeling.\n")

# ---------------------------------------------------------------------------
# Feature engineering
#
# IMPORTANT: only use information that would be known *before* the flight
# departs. Columns like ARR_DELAY, DEP_DELAY, or the DELAY_DUE_* columns
# are the outcome (or derived from it) and must NOT be used as features --
# including them would be "data leakage" and produce a fake, useless model.
# ---------------------------------------------------------------------------

# Scheduled departure hour (0-23) from the CRS_DEP_TIME field (e.g. 1155 -> 11)
df["DEP_HOUR"] = (df["CRS_DEP_TIME"] // 100).astype(int)

# Keep only the top 30 origin/destination airports by flight volume; group
# everything else as "OTHER" so one-hot encoding doesn't explode into
# hundreds of rarely-used columns.
top_origins = df["ORIGIN"].value_counts().nlargest(30).index
top_dests = df["DEST"].value_counts().nlargest(30).index
df["ORIGIN_GROUPED"] = df["ORIGIN"].where(df["ORIGIN"].isin(top_origins), "OTHER")
df["DEST_GROUPED"] = df["DEST"].where(df["DEST"].isin(top_dests), "OTHER")

FEATURES = [
    "ORIGIN_GROUPED",
    "DEST_GROUPED",
    "MONTH",
    "DAY_OF_WEEK",
    "DEP_HOUR",
    "DISTANCE",
]
TARGET = "IS_DELAYED_15"

model_df = df[FEATURES + [TARGET]].dropna()
X = model_df[FEATURES]
y = model_df[TARGET].astype(int)

print(f"Baseline delay rate in this data: {y.mean():.1%}")
print(f"(A model needs to beat this baseline to be useful.)\n")

# ---------------------------------------------------------------------------
# Train / test split
#
# We hold out 20% of the data the model never sees during training, so we
# can honestly measure how well it performs on flights it hasn't seen.
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training on {len(X_train):,} flights, testing on {len(X_test):,} flights.\n")

# ---------------------------------------------------------------------------
# Build the model pipeline
#
# Categorical columns (airports, day of week) need to be one-hot encoded
# (turned into 0/1 columns) since the model only understands numbers.
# Numeric columns (month, hour, distance) pass through unchanged.
# ---------------------------------------------------------------------------
categorical_cols = ["ORIGIN_GROUPED", "DEST_GROUPED", "DAY_OF_WEEK"]
numeric_cols = ["MONTH", "DEP_HOUR", "DISTANCE"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ],
    remainder="passthrough",  # numeric_cols pass through as-is
)

model = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=20,
                class_weight="balanced",  # delays are the minority class
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)

print("Training Random Forest classifier ...")
model.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 70)
print("Model performance on held-out test set")
print("=" * 70)
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}  (of flights we predict delayed, how many really were)")
print(f"Recall:    {recall_score(y_test, y_pred):.3f}  (of flights that were really delayed, how many we caught)")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.3f}  (ability to rank delayed vs. on-time flights; 0.5 = random guessing, 1.0 = perfect)")

print("\nFull classification report:")
print(classification_report(y_test, y_pred, target_names=["On-time", "Delayed 15+"]))

# ---------------------------------------------------------------------------
# Feature importance -- makes the model's reasoning explainable
# ---------------------------------------------------------------------------
ohe = model.named_steps["preprocess"].named_transformers_["cat"]
cat_feature_names = ohe.get_feature_names_out(categorical_cols)
all_feature_names = list(cat_feature_names) + numeric_cols

importances = model.named_steps["classifier"].feature_importances_
importance_df = (
    pd.DataFrame({"feature": all_feature_names, "importance": importances})
    .sort_values("importance", ascending=False)
    .head(15)
)

print("\nTop 15 most important features:")
print(importance_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 7))
sns.barplot(
    x=importance_df["importance"],
    y=importance_df["feature"],
    hue=importance_df["feature"],
    palette="viridis",
    legend=False,
    ax=ax,
)
ax.set_title("Top 15 Features Driving UA Delay Predictions")
ax.set_xlabel("Feature Importance")
ax.set_ylabel("Feature")
fig.tight_layout()
fig.savefig(FIG_DIR / "05_feature_importance.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Save the trained model for reuse (e.g. by the AI assistant later)
# ---------------------------------------------------------------------------
MODEL_PATH = MODEL_DIR / "delay_risk_model.joblib"
joblib.dump(model, MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")
print(f"Feature importance chart saved to {FIG_DIR / '05_feature_importance.png'}")

# ---------------------------------------------------------------------------
# Example: predict delay risk for a few sample flights
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Example predictions")
print("=" * 70)
examples = pd.DataFrame(
    [
        {"ORIGIN_GROUPED": "ORD", "DEST_GROUPED": "EWR", "MONTH": 7, "DAY_OF_WEEK": "Friday", "DEP_HOUR": 17, "DISTANCE": 719},
        {"ORIGIN_GROUPED": "ORD", "DEST_GROUPED": "EWR", "MONTH": 10, "DAY_OF_WEEK": "Tuesday", "DEP_HOUR": 8, "DISTANCE": 719},
    ]
)
probs = model.predict_proba(examples)[:, 1]
for i, row in examples.iterrows():
    print(
        f"{row['ORIGIN_GROUPED']}->{row['DEST_GROUPED']} on a {row['DAY_OF_WEEK']} "
        f"in month {row['MONTH']} at {row['DEP_HOUR']}:00 -> "
        f"predicted delay risk: {probs[i]:.1%}"
    )

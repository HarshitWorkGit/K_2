# ============================================================
# 🌾 KISAN MULTI-MONTH ML DATASET BUILDER (FINAL)
# Builds ONE history file (Feb+March) and ONE test file (2025)
# ============================================================

import os
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

FEB_HIST = "./data/clean data/historical_final.csv"
FEB_2025 = "./data/clean data/feb_2025_final.csv"

MAR_HIST = "./data/clean data/march_all_year_final.csv"
MAR_2025 = "./data/clean data/march_2025_final.csv"

OUT_DIR = "./data/clean data/ml data"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# LOAD FILES
# ============================================================

print("\n" + "="*70)
print("📥 LOADING CLEAN MONTHLY FILES")
print("="*70)

feb_hist = pd.read_csv(FEB_HIST)
feb_2025 = pd.read_csv(FEB_2025)
mar_hist = pd.read_csv(MAR_HIST)
mar_2025 = pd.read_csv(MAR_2025)

print("Feb history :", feb_hist.shape)
print("March history:", mar_hist.shape)
print("Feb 2025    :", feb_2025.shape)
print("March 2025 :", mar_2025.shape)


# ============================================================
# TIME FEATURE ENGINE
# ============================================================

def add_time_features(df, label="DATA"):
    df = df.copy()

    print(f"\n🕒 Parsing time for: {label}")

    df["time"] = pd.to_datetime(
        df["time"],
        errors="coerce",
        infer_datetime_format=True
    )

    bad = df["time"].isna().sum()
    print(f"⏳ Total rows: {len(df)} | Bad time values: {bad}")

    if bad > 0:
        print("⚠️ Dropping rows with invalid time")
        df = df.dropna(subset=["time"])

    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["weekday"] = df["time"].dt.weekday
    df["hour"] = df["time"].dt.hour
    df["week_of_month"] = ((df["time"].dt.day - 1) // 7) + 1

    return df


# ============================================================
# PARSE TIME FIRST (CRITICAL)
# ============================================================

feb_hist = add_time_features(feb_hist, "FEB HISTORY")
mar_hist = add_time_features(mar_hist, "MARCH HISTORY")
feb_2025 = add_time_features(feb_2025, "FEB 2025")
mar_2025 = add_time_features(mar_2025, "MARCH 2025")


# ============================================================
# MERGE MONTHS
# ============================================================

print("\n🔗 Merging months...")

history_all = pd.concat([feb_hist, mar_hist], ignore_index=True)
test_all = pd.concat([feb_2025, mar_2025], ignore_index=True)

print("✅ Combined history:", history_all.shape)
print("✅ Combined test   :", test_all.shape)


# ============================================================
# FINAL ML COLUMNS
# ============================================================

FEATURE_COLS = [
    "district_id",
    "organization_id",
    "organization_type_id",
    "intotal",
    "year",
    "month",
    "week_of_month",
    "weekday",
    "hour",
    "demand_id",
    "demand_clean",
    "time"
]

history_ml = history_all[FEATURE_COLS]
test_ml = test_all[FEATURE_COLS]


# ============================================================
# DATA QUALITY CHECK
# ============================================================

print("\n" + "="*70)
print("🔍 DATA QUALITY CHECK")
print("="*70)

print("\nHistory missing values:")
print(history_ml.isna().sum())

print("\nTest missing values:")
print(test_ml.isna().sum())

print("\nHistory demand distribution:")
print(history_ml["demand_clean"].value_counts().head(10))

print("\n2025 demand distribution:")
print(test_ml["demand_clean"].value_counts().head(10))


# ============================================================
# SAVE FILES
# ============================================================

TRAIN_OUT = f"{OUT_DIR}/ml_train_feb_march.csv"
TEST_OUT  = f"{OUT_DIR}/ml_test_2025_feb_march.csv"

history_ml.to_csv(TRAIN_OUT, index=False, encoding="utf-8-sig")
test_ml.to_csv(TEST_OUT, index=False, encoding="utf-8-sig")

print("\n" + "="*70)
print("💾 FINAL ML FILES CREATED")
print("="*70)
print(TRAIN_OUT)
print(TEST_OUT)

print("\n======================================================")
print("✅ MULTI-MONTH ML DATASET READY")
print("Next:")
print("→ Retrain XGBoost on Feb+March")
print("→ Generate new weekly probability graphs")
print("→ Compare Feb-only vs Feb+March models")
print("======================================================\n")

# ============================================================
# 🌾 KISAN – ML DATASET BUILDER
# Builds final machine-learning datasets from clean files
# Output: ml_train.csv, ml_test_2025.csv
# ============================================================

import os
import pandas as pd

# ---------------- CONFIG ----------------
CLEAN_FOLDER = "./data/clean data"
OUTPUT_FOLDER = "./data/clean data/ml data"

HIST_FILE = f"{CLEAN_FOLDER}/historical_final.csv"
FEB_FILE  = f"{CLEAN_FOLDER}/feb_2025_final.csv"

TRAIN_OUT = f"{OUTPUT_FOLDER}/ml_train.csv"
TEST_OUT  = f"{OUTPUT_FOLDER}/ml_test_2025.csv"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
# ----------------------------------------


# ============================================================
# STEP 1: LOAD DATA
# ============================================================

print("\n" + "="*70)
print("📥 LOADING CLEAN DATASETS")
print("="*70)

df_hist = pd.read_csv(HIST_FILE)
df_2025 = pd.read_csv(FEB_FILE)

print("✅ Historical:", df_hist.shape)
print("✅ Feb 2025  :", df_2025.shape)


# ============================================================
# STEP 2: TIME FEATURE ENGINEERING
# ============================================================

def add_time_features(df):
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["day"] = df["time"].dt.day
    df["weekday"] = df["time"].dt.weekday
    df["hour"] = df["time"].dt.hour
    df["week_of_month"] = ((df["day"] - 1) // 7 + 1).astype(int)

    return df


print("\n🕒 Adding time-based features...")

df_hist = add_time_features(df_hist)
df_2025 = add_time_features(df_2025)


# ============================================================
# STEP 3: CORE FEATURE SELECTION
# ============================================================

FEATURES = [
    "district_id",
    "organization_id",
    "organization_type_id",
    "intotal",
    "year",
    "month",
    "week_of_month",
    "weekday",
    "hour"
]

TARGET_CLASS = "demand_id"
TARGET_NAME  = "demand_clean"

KEEP_COLS = FEATURES + [TARGET_CLASS, TARGET_NAME, "time"]

df_hist_ml = df_hist[KEEP_COLS].dropna()
df_2025_ml = df_2025[KEEP_COLS].dropna()


# ============================================================
# STEP 4: DATA VALIDATION
# ============================================================

print("\n" + "="*70)
print("🔍 DATA QUALITY CHECK")
print("="*70)

print("\nTrain shape:", df_hist_ml.shape)
print("Test shape :", df_2025_ml.shape)

print("\n🎯 Target distribution (train):")
print(df_hist_ml[TARGET_NAME].value_counts())

print("\n🎯 Target distribution (2025):")
print(df_2025_ml[TARGET_NAME].value_counts())

print("\n❓ Missing values:")
print(df_hist_ml.isna().sum())


# ============================================================
# STEP 5: SAVE FINAL ML FILES
# ============================================================

df_hist_ml.to_csv(TRAIN_OUT, index=False, encoding="utf-8-sig")
df_2025_ml.to_csv(TEST_OUT, index=False, encoding="utf-8-sig")

print("\n" + "="*70)
print("💾 FINAL ML FILES CREATED")
print("="*70)
print(TRAIN_OUT)
print(TEST_OUT)


# ============================================================
# STEP 6: FINAL SUMMARY
# ============================================================

print("\n" + "="*70)
print("✅ ML DATASET BUILD COMPLETED")
print("======================================================")
print("Next step:")
print("→ Train strong models (XGBoost / LightGBM)")
print("→ Probability-first demand forecasting")
print("→ Weekly evaluation")
print("="*70)

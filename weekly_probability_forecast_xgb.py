# ============================================================
# 🌾 WEEKLY DEMAND PROBABILITY FORECAST (XGBOOST)
# Uses trained model to estimate weekly demand risk
# ============================================================

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
TEST_FILE = "./data/clean data/ml data/ml_test_2025.csv"
MODEL_PATH = "./models/xgb_demand_model.pkl"
OUTPUT_DIR = "./outputs/weekly_probability_xgb"

os.makedirs(OUTPUT_DIR, exist_ok=True)
# ----------------------------------------


# ============================================================
# LOAD DATA & MODEL
# ============================================================

print("\n" + "="*70)
print("📥 LOADING DATA & MODEL")
print("="*70)

df = pd.read_csv(TEST_FILE)
model = joblib.load(MODEL_PATH)

print("✅ Data:", df.shape)
print("✅ Model loaded")


# ============================================================
# FEATURE SETUP
# ============================================================

feature_cols = [
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

X = df[feature_cols]
weeks = df["week_of_month"]

labels = model.classes_   # demand names in model order


# ============================================================
# PREDICT PROBABILITIES
# ============================================================

print("\n🔮 Predicting probabilities...")

probs = model.predict_proba(X)
prob_df = pd.DataFrame(probs, columns=labels)

prob_df["week_of_month"] = weeks.values


# ============================================================
# WEEKLY AGGREGATION
# ============================================================

weekly_probs = (
    prob_df
    .groupby("week_of_month")
    .mean()
    .reset_index()
)

weekly_probs.to_csv(f"{OUTPUT_DIR}/weekly_demand_probabilities.csv", index=False, encoding="utf-8-sig")

print("\n=================================================")
print("📊 WEEKLY AVERAGE DEMAND PROBABILITIES")
print("=================================================")
print(weekly_probs.round(3))


# ============================================================
# HEATMAP (CLEAN STORY VIEW)
# ============================================================

plt.figure(figsize=(14,6))
plt.imshow(weekly_probs.drop("week_of_month", axis=1).T, aspect="auto")

plt.colorbar(label="Average probability")
plt.yticks(range(len(labels)), [f"D{i}" for i in range(len(labels))])
plt.xticks(range(len(weekly_probs)), weekly_probs["week_of_month"])
plt.xlabel("Week of February 2025")
plt.ylabel("Demand ID Index")
plt.title("Weekly Demand Probability Heatmap (XGBoost)")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/weekly_probability_heatmap.png", dpi=200)
plt.close()

print("🖼 Saved: weekly_probability_heatmap.png")


# ============================================================
# TREND PLOTS (TOP 6 DEMANDS)
# ============================================================

top6 = weekly_probs.drop("week_of_month", axis=1).mean().sort_values(ascending=False).head(6).index

plt.figure(figsize=(10,6))

for d in top6:
    plt.plot(
        weekly_probs["week_of_month"],
        weekly_probs[d],
        marker="o",
        label=f"D{d}"   # clean ID-based labels
    )
    
plt.xlabel("Week of February")
plt.ylabel("Average probability")
plt.title("Top Demand Risk Trends (Feb 2025)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/weekly_probability_trends.png", dpi=200)
plt.close()

print("🖼 Saved: weekly_probability_trends.png")


# ============================================================
# THRESHOLD ALERT PLOTS (>= 0.50)
# ============================================================

THRESHOLD = 0.50

for _, row in weekly_probs.iterrows():
    week = int(row["week_of_month"])
    temp = row.drop("week_of_month")

    alerts = temp[temp >= THRESHOLD].sort_values(ascending=False)

    if len(alerts) == 0:
        continue

    plt.figure(figsize=(8,4))
    alerts.plot(kind="bar")
    plt.ylim(0,1)
    plt.title(f"Week {week} – High Risk Demands (≥ {THRESHOLD})")
    plt.ylabel("Probability")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/week_{week}_alerts.png", dpi=200)
    plt.close()

    print(f"🖼 Saved: week_{week}_alerts.png")


# ============================================================
# FINISH
# ============================================================

print("\n=================================================")
print("✅ WEEKLY PROBABILITY FORECAST COMPLETE")
print("📂 Output folder:", OUTPUT_DIR)
print("Generated:")
print(" - weekly_demand_probabilities.csv")
print(" - weekly_probability_heatmap.png")
print(" - weekly_probability_trends.png")
print(" - week_X_alerts.png")
print("=================================================\n")

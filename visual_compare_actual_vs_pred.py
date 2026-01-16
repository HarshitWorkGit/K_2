# ============================================================
# 📊 OVERALL ACTUAL vs PREDICTED (DEMAND ID ONLY) – 2025 (FEB+MARCH)
# Clean management-grade validation graph
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
EVENT_FILE   = "./outputs/visual_reports/feb_event_predictions.csv"
DEMANDS_FILE = "./data/clean data/demand_master_clean.csv"
OUT_DIR      = "./outputs/visual_reports"
# ----------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

print("\n📥 Loading files...")
df = pd.read_csv(EVENT_FILE)
demands = pd.read_csv(DEMANDS_FILE)

print("✅ Events :", df.shape)
print("✅ Demands:", demands.shape)


# ============================================================
# BUILD DEMAND NAME → ID MAP
# ============================================================

demand_map = dict(zip(demands["demand"], demands["demand_id"]))

df["actual_demand_id"] = df["actual_demand"].map(demand_map)
df["predicted_demand_id"] = df["predicted_demand"].map(demand_map)

print("⚠️ Unmapped actual demands   :", df["actual_demand_id"].isna().sum())
print("⚠️ Unmapped predicted demands:", df["predicted_demand_id"].isna().sum())

# Drop broken rows (if any)
df = df.dropna(subset=["actual_demand_id", "predicted_demand_id"])

df["actual_demand_id"] = df["actual_demand_id"].astype(int)
df["predicted_demand_id"] = df["predicted_demand_id"].astype(int)


# ============================================================
# COUNT COMPARISON
# ============================================================

actual_counts = df["actual_demand_id"].value_counts().sort_index()
pred_counts   = df["predicted_demand_id"].value_counts().sort_index()

compare = pd.DataFrame({
    "Actual": actual_counts,
    "Predicted": pred_counts
}).fillna(0).astype(int)

print("\n📊 Demand ID comparison table:\n")
print(compare)


# ============================================================
# PLOT (CLEAN ID GRAPH)
# ============================================================

plt.figure(figsize=(12,6))
compare.plot(kind="bar")

plt.title("2025 Actual vs Predicted Demand Distribution (by Demand ID)")
plt.xlabel("Demand ID")
plt.ylabel("Number of Events")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/overall_actual_vs_pred_demand_id.png", dpi=200)
plt.close()

print("\n🖼 Saved:")
print("-> overall_actual_vs_pred_demand_id.png")


# ============================================================
# FINISH
# ============================================================

print("\n=================================================")
print("✅ OVERALL VISUAL COMPARISON COMPLETE")
print("📂 Folder:", OUT_DIR)
print("=================================================")

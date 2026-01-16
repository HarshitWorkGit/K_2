# ============================================================
# 🌾 KISAN DATA CONSOLIDATION PIPELINE
# Builds final ML-ready datasets with demand_id mapping
# ============================================================

import os
import json
import ast
import pandas as pd

# ---------------- CONFIG ----------------
DEMANDS_FOLDER = "./data/Demands"
JSON_FOLDER = "./data/JSON"
CLEAN_FOLDER = "./data/clean data"

os.makedirs(CLEAN_FOLDER, exist_ok=True)
# ----------------------------------------


# ============================================================
# STEP 1: FIND FILES (EXPLICIT + SAFE)
# ============================================================

def find_file(folder, exts):
    for f in os.listdir(folder):
        if f.lower().endswith(exts):
            return os.path.join(folder, f)
    return None


print("\n" + "="*70)
print("📥 SEARCHING FILES")
print("="*70)

# --- Demands master ---
DEMANDS_PATH = find_file(DEMANDS_FOLDER, (".csv", ".xlsx"))
if not DEMANDS_PATH:
    raise FileNotFoundError("❌ No demands file found in data/Demands")

# --- JSON files (EXPLICIT, no guessing) ---
HIST_PATH = os.path.join(JSON_FOLDER, "kisan_feb_all_years.json")
FEB_PATH  = os.path.join(JSON_FOLDER, "kisan_feb_2025.json")

if not os.path.exists(HIST_PATH):
    raise FileNotFoundError("❌ Historical JSON not found: kisan_feb_all_years.json")

if not os.path.exists(FEB_PATH):
    raise FileNotFoundError("❌ Feb 2025 JSON not found: kisan_feb_2025.json")

print("✅ Demands file :", DEMANDS_PATH)
print("✅ Historical  :", HIST_PATH)
print("✅ Feb 2025    :", FEB_PATH)


# ============================================================
# STEP 2: LOAD DEMANDS MASTER
# ============================================================

print("\n" + "="*70)
print("📊 LOADING DEMANDS MASTER")
print("="*70)

if DEMANDS_PATH.endswith(".xlsx"):
    df_demands = pd.read_excel(DEMANDS_PATH)
else:
    df_demands = pd.read_csv(DEMANDS_PATH)

df_demands = df_demands.rename(columns={
    df_demands.columns[0]: "demand_id",
    "name": "demand"
})

df_demands["demand"] = df_demands["demand"].astype(str).str.strip()

print("Shape:", df_demands.shape)
print("Columns:", df_demands.columns.tolist())
print(df_demands.head(3))

df_demands.to_csv(f"{CLEAN_FOLDER}/demand_master_clean.csv", index=False, encoding="utf-8-sig")

# ============================================================
# STEP 3: BUILD SYNONYM MAP
# ============================================================

print("\n" + "="*70)
print("🧠 BUILDING DEMAND MAPPING ENGINE")
print("="*70)

synonym_map = {}

for _, row in df_demands.iterrows():
    did = row["demand_id"]
    name = row["demand"]

    synonym_map[name] = (did, name)

    try:
        syns = ast.literal_eval(row["synonyms"])
        for s in syns:
            synonym_map[str(s).strip()] = (did, name)
    except:
        pass

print("Total mapping keys:", len(synonym_map))


# ============================================================
# STEP 4: LOAD JSON DATA
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return pd.DataFrame(json.load(f))

df_hist = load_json(HIST_PATH)
df_2025 = load_json(FEB_PATH)


# ============================================================
# STEP 5: MAP DEMANDS
# ============================================================

def map_demands(df, label):
    print(f"\n📌 Mapping demands for: {label}")

    df = df.copy()
    df["demand"] = df["demand"].astype(str).str.strip()

    mapped_id = []
    mapped_name = []
    unmapped = []

    for d in df["demand"]:
        if d in synonym_map:
            did, name = synonym_map[d]
            mapped_id.append(did)
            mapped_name.append(name)
        else:
            mapped_id.append(-1)
            mapped_name.append("UNMAPPED")
            unmapped.append(d)

    df["demand_id"] = mapped_id
    df["demand_clean"] = mapped_name

    print("Total records:", len(df))
    print("Mapped:", (df["demand_id"] != -1).sum())
    print("Unmapped:", (df["demand_id"] == -1).sum())

    if unmapped:
        print("\n⚠️ Unmapped values sample:")
        print(pd.Series(unmapped).value_counts().head(10))

    return df


df_hist = map_demands(df_hist, "Historical")
df_2025 = map_demands(df_2025, "Feb 2025")


# ============================================================
# STEP 6: SAVE FINAL DATASETS
# ============================================================

hist_out = f"{CLEAN_FOLDER}/historical_final.csv"
feb_out = f"{CLEAN_FOLDER}/feb_2025_final.csv"

df_hist.to_csv(hist_out, index=False, encoding="utf-8-sig")
df_2025.to_csv(feb_out, index=False, encoding="utf-8-sig")

print("\n" + "="*70)
print("💾 FINAL FILES CREATED")
print("="*70)
print(hist_out)
print(feb_out)


# ============================================================
# STEP 7: FINAL REPORT
# ============================================================

print("\n" + "="*70)
print("📊 FINAL DATA QUALITY REPORT")
print("="*70)

print("\nHistorical demand distribution:")
print(df_hist["demand_clean"].value_counts())

print("\nFeb 2025 demand distribution:")
print(df_2025["demand_clean"].value_counts())

print("\n======================================================")
print("✅ DATA CONSOLIDATION COMPLETED SUCCESSFULLY")
print("Next step: Feature engineering + strong ML models")
print("======================================================\n")

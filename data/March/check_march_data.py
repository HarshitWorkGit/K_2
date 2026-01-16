# ============================================================
# 🌾 BUILD MARCH CLEAN DATASETS (WITH DEMAND ID MAPPING)
# Historical: March_all_year.json
# Test      : Mar_2025.json
# Output    : march_all_year_final.csv, march_2025_final.csv
# ============================================================

import os
import json
import ast
import pandas as pd

# ---------------- CONFIG ----------------
MARCH_FOLDER   = "./data/March"
DEMANDS_FOLDER = "./data/Demands"
CLEAN_FOLDER   = "./data/clean data"

os.makedirs(CLEAN_FOLDER, exist_ok=True)
# ----------------------------------------


# ============================================================
# STEP 1: FIND FILES
# ============================================================

def find_file(folder, contains=None, exts=(".json",)):
    for f in os.listdir(folder):
        if f.lower().endswith(exts):
            if contains is None or contains.lower() in f.lower():
                return os.path.join(folder, f)
    return None


print("\n" + "="*70)
print("📥 SEARCHING MARCH FILES")
print("="*70)

MARCH_HIST = find_file(MARCH_FOLDER, contains="all")
MARCH_2025 = find_file(MARCH_FOLDER, contains="2025")
DEMANDS   = find_file(DEMANDS_FOLDER, exts=(".csv", ".xlsx"))

if not all([MARCH_HIST, MARCH_2025, DEMANDS]):
    raise FileNotFoundError("❌ Missing March or Demands file")

print("✅ Demands  :", DEMANDS)
print("✅ Historical:", MARCH_HIST)
print("✅ March 2025:", MARCH_2025)


# ============================================================
# STEP 2: LOAD DEMANDS MASTER
# ============================================================

print("\n" + "="*70)
print("📊 LOADING DEMANDS MASTER")
print("="*70)

if DEMANDS.endswith(".xlsx"):
    df_demands = pd.read_excel(DEMANDS)
else:
    df_demands = pd.read_csv(DEMANDS)

df_demands = df_demands.rename(columns={
    df_demands.columns[0]: "demand_id",
    "name": "demand"
})

df_demands["demand"] = df_demands["demand"].astype(str).str.strip()

print("Shape:", df_demands.shape)
print("Columns:", df_demands.columns.tolist())
print(df_demands.head(3))


# ============================================================
# STEP 3: BUILD SYNONYM MAP
# ============================================================

print("\n" + "="*70)
print("🧠 BUILDING DEMAND MAPPING ENGINE")
print("="*70)

synonym_map = {}

for _, row in df_demands.iterrows():
    did = row["demand_id"]
    name = row["demand"].strip()

    synonym_map[name] = (did, name)

    try:
        syns = ast.literal_eval(row["synonyms"])
        for s in syns:
            synonym_map[str(s).strip()] = (did, name)
    except:
        pass

print("Total mapping keys:", len(synonym_map))


# ============================================================
# STEP 4: LOAD JSON
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return pd.DataFrame(json.load(f))

df_hist = load_json(MARCH_HIST)
df_2025 = load_json(MARCH_2025)


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

    print("Total:", len(df))
    print("Mapped:", (df["demand_id"] != -1).sum())
    print("Unmapped:", (df["demand_id"] == -1).sum())

    if unmapped:
        print("\n⚠️ Unmapped samples:")
        print(pd.Series(unmapped).value_counts().head(10))

    return df


df_hist = map_demands(df_hist, "March Historical")
df_2025 = map_demands(df_2025, "March 2025")


# ============================================================
# STEP 6: BASIC CLEANING
# ============================================================

def basic_clean(df):
    df = df.copy()

    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    text_cols = ["district", "incident", "organization", "organization_type", "leadername"]
    for c in text_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    df["organization"] = df["organization"].fillna("Unknown")
    df["organization_type"] = df["organization_type"].fillna("Unknown")
    df["organization_id"] = df["organization_id"].fillna(0)

    df = df.dropna(subset=["district_id", "time", "demand_clean"])

    return df


df_hist = basic_clean(df_hist)
df_2025 = basic_clean(df_2025)


# ============================================================
# STEP 7: SAVE FILES
# ============================================================

hist_out = f"{CLEAN_FOLDER}/march_all_year_final.csv"
test_out = f"{CLEAN_FOLDER}/march_2025_final.csv"

df_hist.to_csv(hist_out, index=False, encoding="utf-8-sig")
df_2025.to_csv(test_out, index=False, encoding="utf-8-sig")

print("\n" + "="*70)
print("💾 FINAL FILES CREATED")
print("="*70)
print(hist_out)
print(test_out)


# ============================================================
# STEP 8: FINAL REPORT
# ============================================================

print("\n" + "="*70)
print("📊 FINAL DATA QUALITY REPORT")
print("="*70)

print("\nMarch historical demand distribution:")
print(df_hist["demand_clean"].value_counts())

print("\nMarch 2025 demand distribution:")
print(df_2025["demand_clean"].value_counts())

print("\n======================================================")
print("✅ MARCH DATA CONSOLIDATION COMPLETED")
print("Next step: Merge Feb + March and rebuild ML dataset")
print("======================================================\n")

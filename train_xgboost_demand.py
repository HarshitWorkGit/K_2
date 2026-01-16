# ============================================================
# 🌾 KISAN DEMAND MODEL – XGBOOST (BALANCED, PRODUCTION GRADE)
# Trains on historical data, tests on Feb 2025
# Focus: exact demand + probability quality
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score


# ============================================================
# CONFIG
# ============================================================

TRAIN_PATH = "./data/clean data/ml data/ml_train_feb_march.csv"
TEST_PATH  = "./data/clean data/ml data/ml_test_2025_feb_march.csv"

MODEL_DIR  = "./models"

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "="*70)
print("📥 LOADING ML DATASETS")
print("="*70)

train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

print("✅ Train:", train_df.shape)
print("✅ Test :", test_df.shape)


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

target_col = "demand_clean"

X_train = train_df[feature_cols]
X_test  = test_df[feature_cols]

y_train_raw = train_df[target_col]
y_test_raw  = test_df[target_col]


# ============================================================
# LABEL ENCODING
# ============================================================

le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)
y_test  = le.transform(y_test_raw)

joblib.dump(le, f"{MODEL_DIR}/demand_label_encoder.pkl")


# ============================================================
# CLASS BALANCING (IMPORTANT)
# ============================================================

class_counts = np.bincount(y_train)
total = len(y_train)
class_weights = {i: total/(len(class_counts)*c) for i, c in enumerate(class_counts)}

sample_weights = np.array([class_weights[i] for i in y_train])


# ============================================================
# MODEL
# ============================================================

print("\n" + "="*70)
print("🚀 TRAINING BALANCED XGBOOST MODEL")
print("="*70)

model = XGBClassifier(
    n_estimators=450,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="multi:softprob",
    num_class=len(le.classes_),
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train, sample_weight=sample_weights)

joblib.dump(model, f"{MODEL_DIR}/xgboost_demand_model.pkl")

print("✅ Model trained & saved")


# ============================================================
# EVALUATION ON FEB 2025
# ============================================================

print("\n" + "="*70)
print("📊 FEB 2025 REAL-WORLD TEST")
print("="*70)

probs = model.predict_proba(X_test)
preds = np.argmax(probs, axis=1)

y_pred = le.inverse_transform(preds)

acc = accuracy_score(y_test_raw, y_pred)
print("🎯 Top-1 Accuracy:", round(acc, 3))

print("\n📄 Classification Report:\n")
print(classification_report(y_test_raw, y_pred, zero_division=0))


# ============================================================
# TOP-3 ACCURACY
# ============================================================

top3 = np.argsort(probs, axis=1)[:, -3:]
hit3 = sum(y_test[i] in top3[i] for i in range(len(y_test)))

print("\n🎯 Top-3 Accuracy:", round(hit3/len(y_test), 3))


# ============================================================
# CONFIDENCE ANALYSIS
# ============================================================

max_probs = probs.max(axis=1)

print("\n🔍 CONFIDENCE CHECK")
print("Avg confidence :", round(max_probs.mean(), 3))
print("High confidence(>0.7):", (max_probs > 0.7).sum(), "/", len(max_probs))
print("Low confidence (<0.4):", (max_probs < 0.4).sum(), "/", len(max_probs))


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print("\n🧪 SAMPLE FEB 2025 PREDICTIONS\n")

classes = le.classes_

for i in range(min(5, len(X_test))):
    print("Case", i+1)
    print("Actual   :", y_test_raw.iloc[i])
    print("Predicted:", y_pred[i])

    top5 = np.argsort(probs[i])[-5:][::-1]

    print("Top-5:")
    for idx in top5:
        print(f"   {classes[idx]}  ->  {round(probs[i][idx],3)}")
    print()


# ============================================================
# 📁 SAVE EVENT-LEVEL PREDICTIONS FOR VISUAL ANALYSIS
# ============================================================

print("\n💾 Saving event-level Feb 2025 predictions...")

OUT_DIR = "./outputs/visual_reports"
os.makedirs(OUT_DIR, exist_ok=True)

results_df = test_df.copy()

results_df["actual_demand"] = y_test_raw.values
results_df["predicted_demand"] = y_pred
results_df["confidence"] = max_probs

# Top-3 predictions
top3_idx = np.argsort(probs, axis=1)[:, -3:][:, ::-1]

results_df["top1"] = [le.classes_[i[0]] for i in top3_idx]
results_df["top1_prob"] = [probs[j][i[0]] for j, i in enumerate(top3_idx)]

results_df["top2"] = [le.classes_[i[1]] for i in top3_idx]
results_df["top2_prob"] = [probs[j][i[1]] for j, i in enumerate(top3_idx)]

results_df["top3"] = [le.classes_[i[2]] for i in top3_idx]
results_df["top3_prob"] = [probs[j][i[2]] for j, i in enumerate(top3_idx)]

SAVE_PATH = f"{OUT_DIR}/feb_event_predictions.csv"
results_df.to_csv(SAVE_PATH, index=False, encoding="utf-8-sig")

print("✅ Saved:", SAVE_PATH)


# ============================================================
# FINISH
# ============================================================

print("\n" + "="*70)
print("✅ XGBOOST DEMAND MODEL PIPELINE COMPLETE")
print("Model saved in ./models/")
print("Next: Weekly probability forecasting")
print("="*70)

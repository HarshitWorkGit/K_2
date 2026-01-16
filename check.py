import pandas as pd

df = pd.read_csv("./data/clean data/march_all_year_final.csv")

print("\nTotal rows:", len(df))

bad = df[df["time"].isna() | df["time"].astype(str).str.len().gt(40)]

print("\nShowing 15 problematic time values:\n")

for i, t in enumerate(bad["time"].astype(str).head(15)):
    print(f"{i+1}.", t)

print("------------------")

df2 = pd.read_csv("./data/clean data/feb_2025_final.csv")

print("\n--- FEB 2025 TIME SAMPLES ---\n")

for t in df2["time"].astype(str).head(10):
    print(t)

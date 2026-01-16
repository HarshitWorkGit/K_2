import pandas as pd

df = pd.read_csv("./outputs\weekly_probability_xgb\weekly_demand_probabilities.csv")
print(df.shape)
print(df.columns)
print(df.head(3))

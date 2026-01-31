import pandas as pd

df = pd.read_csv("data/aapl.csv")

# Force Price column to datetime
df["Price"] = pd.to_datetime(df["Price"], errors="coerce")

# Drop rows where Price is not a valid date (removes Date/Ticker rows)
df = df.dropna(subset=["Price"])

df.set_index("Price", inplace=True)

df = df.sort_index()

print("First 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

print("\nIndex type:")
print(type(df.index))

print("\nColumns:")
print(df.columns)

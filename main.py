import pandas as pd
import matplotlib.pyplot as plt

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
# Ensure Close is numeric
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

# Compute daily returns
df["return"] = df["Close"].pct_change()

print("\nReturns preview:")
print(df[["Close", "return"]].head(10))

# Plot closing price
plt.figure(figsize=(10, 4))
df["Close"].plot(title="AAPL Closing Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.tight_layout()
plt.show()

# Plot daily returns
plt.figure(figsize=(10, 4))
df["return"].plot(title="AAPL Daily Returns")
plt.xlabel("Date")
plt.ylabel("Daily Return")
plt.tight_layout()
plt.show()

print("\nReturn statistics:")
print(df["return"].describe())


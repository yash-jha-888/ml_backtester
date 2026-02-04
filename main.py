import pandas as pd
import matplotlib.pyplot as plt

# ---------- Data Loading ----------
def load_data():
    df = pd.read_csv("data/aapl.csv")

    df["Price"] = pd.to_datetime(
    df["Price"],
    format="%Y-%m-%d",
    errors="coerce"
)

    df = df.dropna(subset=["Price"])

    df.set_index("Price", inplace=True)
    df = df.sort_index()

    # Ensure numeric prices
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    return df


# ---------- Returns ----------
def compute_returns(df):
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["return"] = df["Close"].pct_change()
    return df


# ---------- Plots ----------
def plot_price_and_returns(df):
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

# ---------- Strategy ----------

def run_ma_strategy(df, fast=20, slow=50):
    df = df.copy()

    # Moving averages
    df["sma_fast"] = df["Close"].rolling(fast).mean()
    df["sma_slow"] = df["Close"].rolling(slow).mean()

    # Signal: 1 = long, 0 = out
    df["signal"] = (df["sma_fast"] > df["sma_slow"]).astype(int)

    # Shift signal to avoid look-ahead bias
    df["position"] = df["signal"].shift(1)

    # Strategy returns
    df["strategy_return"] = df["position"] * df["return"]

    # Cumulative returns
    df["cum_buy_hold"] = (1 + df["return"]).cumprod()
    df["cum_strategy"] = (1 + df["strategy_return"]).cumprod()

    # Plot equity curves
    plt.figure(figsize=(10, 5))
    df["cum_buy_hold"].plot(label="Buy & Hold")
    df["cum_strategy"].plot(label="MA Crossover")
    plt.title("Equity Curve: MA Strategy vs Buy & Hold")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Simple performance printout
    bh_total = df["cum_buy_hold"].iloc[-1] - 1
    strat_total = df["cum_strategy"].iloc[-1] - 1

    print(f"\nBuy & Hold Total Return: {bh_total:.2%}")
    print(f"MA Strategy Total Return: {strat_total:.2%}")

    return df


def main():
    df = load_data()
    df = compute_returns(df)

    RUN_PLOTS = False
    RUN_STRATEGY = True

    if RUN_PLOTS:
        plot_price_and_returns(df)

    if RUN_STRATEGY:
        run_ma_strategy(df)


if __name__ == "__main__":
    main()
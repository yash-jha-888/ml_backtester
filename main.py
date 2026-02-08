import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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
    bh_metrics = compute_metrics(df["return"])
    strat_metrics = compute_metrics(df["strategy_return"])

    print("\nBuy & Hold Metrics:")
    for k, v in bh_metrics.items():
        print(f"{k}: {v:.2%}")

    print("\nMA Strategy Metrics:")
    for k, v in strat_metrics.items():
        print(f"{k}: {v:.2%}")


    return df

# ---------- ML Strategy ----------

def run_ml_strategy(df, split_ratio=0.7):
    df = create_ml_features(df)

    features = ["ret_1", "ret_5", "sma_ratio"]
    X = df[features]
    y = df["target"]

    # Time-based train/test split (NO shuffling)
    split = int(len(df) * split_ratio)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # Scale features (fit ONLY on train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    # Predict signals
    df.loc[df.index[split:], "ml_signal"] = model.predict(X_test_scaled)

    # Shift to avoid look-ahead bias
    df["ml_position"] = df["ml_signal"].shift(1)

    # Strategy returns
    df["ml_return"] = df["ml_position"] * df["return"]

    # Metrics
    ml_metrics = compute_metrics(df["ml_return"])

    print("\nML Strategy Metrics:")
    for k, v in ml_metrics.items():
        print(f"{k}: {v:.2%}")

    return df

# ---------- Metrics ----------

def compute_metrics(returns):
    returns = returns.dropna()

    # Total return
    total_return = (1 + returns).prod() - 1

    # Equity curve
    equity = (1 + returns).cumprod()

    # Max drawdown
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Win rate
    # Only count days when a trade is active
    trade_returns = returns[returns != 0]

    if len(trade_returns) == 0:
        win_rate = 0.0
    else:
        win_rate = (trade_returns > 0).mean()


    return {
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
    }

def eda_returns(df):
    print("\nEDA: Returns statistics")
    print(df["return"].describe())

    plt.figure(figsize=(6, 4))
    df["return"].hist(bins=50)
    plt.title("Distribution of Daily Returns")
    plt.tight_layout()
    plt.show()

def eda_labels(df):
    labels = (df["return"] > 0).astype(int)
    print("\nEDA: Up vs Down days")
    print(labels.value_counts(normalize=True))

# ---------- ML Features ----------
def create_ml_features(df):
    df = df.copy()

    # Features (ONLY past information)
    df["ret_1"] = df["return"]
    df["ret_5"] = df["Close"].pct_change(5)
    df["sma_ratio"] = df["Close"] / df["Close"].rolling(20).mean() - 1

    # Target: next-day direction (future, allowed)
    df["target"] = (df["return"].shift(-1) > 0).astype(int)

    df = df.dropna()

    return df






def main():
    df = load_data()
    df = compute_returns(df)

    RUN_EDA = True
    RUN_PLOTS = False
    RUN_MA = False
    RUN_ML = True

    if RUN_EDA:
        eda_returns(df)
        eda_labels(df)

    if RUN_PLOTS:
        plot_price_and_returns(df)

    if RUN_MA:
        run_ma_strategy(df)

    if RUN_ML:
        run_ml_strategy(df)



if __name__ == "__main__":
    main()
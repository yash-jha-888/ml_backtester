# Rule-to-ML Trading Strategy Backtester

A lightweight Python backtesting engine that compares a traditional rule-based trading strategy with a simple machine learning classifier on historical price data.
---

## Project Overview

Financial markets are noisy and non-stationary, making predictive modeling difficult.  
This project explores that challenge by comparing:

1. A **rule-based Moving Average Crossover strategy**
2. A **Logistic Regression model** predicting next-day price movement

Both strategies are evaluated on the same historical dataset using identical performance metrics.

---

## Features

- Load historical OHLCV price data from CSV
- Robust handling of real-world financial data quirks
- Rule-based trading strategy (Moving Average Crossover)
- ML-based trading strategy (Logistic Regression classifier)
- Backtesting engine with realistic execution assumptions
- Performance comparison using common trading metrics
- Simple, reproducible project structure

---

## Data

- Asset: **AAPL (Apple Inc.)**
- Frequency: Daily
- Source: Downloaded using the `yfinance` library
- Columns: Open, High, Low, Close, Volume

The dataset is included in the repository for **reproducibility and ease of evaluation**.

---

## Strategies

### 1. Moving Average Crossover (Rule-Based)
- Uses a fast and slow simple moving average
- Long-only strategy
- Enters a position when fast MA crosses above slow MA
- Exits when the signal reverses

### 2. Logistic Regression (ML-Based)
- Supervised binary classification
- Predicts whether the next day’s closing price will increase
- Uses only **past information** as features
- Trained and tested using a chronological split to prevent data leakage

---

## Performance Metrics

Each strategy is evaluated using:
- Total return
- Maximum drawdown
- Win rate
- Equity curve visualization

All metrics are computed using **out-of-sample data only** where applicable.

---

## Key Learnings

- Correct time-series handling is more important than model complexity
- Avoiding data leakage is critical in financial ML experiments
- Simple rule-based strategies can be competitive with ML baselines
- Machine learning does not automatically outperform traditional methods in trading

---

## Project Structure

├── data/
│ └── aapl.csv
├── main.py
├── README.md
└── .gitignore


---

## How to Run

1. Clone the repository
2. Install dependencies:

   pip install pandas numpy matplotlib scikit-learn

3. Run the backtester:
   python main.py

## Disclaimer

This project is for educational purposes only.
It does not constitute financial advice, and the strategies implemented are not intended for live trading.

## Future Improvements

- Transaction costs and slippage modeling

- Walk-forward validation

- Additional technical indicators

- Support for multiple assets

- Command-line configuration options


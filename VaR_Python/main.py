"""
Runs all three VaR models on the same data and prints a comparison table.

    pip install yfinance pandas numpy
    python3 main.py

Each method can also be run on its own, e.g.  python3 var_historical.py
"""
import pandas as pd

import var_covariance
import var_historical
import var_monte_carlo
from config import HOLDING_DAYS, PORTFOLIO_VALUE, TICKERS, WEIGHTS
from data import load_returns


def main():
    returns = load_returns()  # fetch once, reuse for every method

    summary = pd.DataFrame({
        "Variance-Covariance": var_covariance.run(returns),
        "Historical": var_historical.run(returns),
        "Monte Carlo": var_monte_carlo.run(returns),
    })
    summary.index = [f"{ci:.0%}" for ci in summary.index]
    summary.index.name = "Confidence"

    print(f"\n=== SUMMARY: {HOLDING_DAYS}-day VaR on ${PORTFOLIO_VALUE:,.0f} "
          f"({', '.join(f'{t} {w:.1%}' for t, w in zip(TICKERS, WEIGHTS))}) ===")
    print(summary.map(lambda x: f"${x:,.0f}"))


if __name__ == "__main__":
    main()

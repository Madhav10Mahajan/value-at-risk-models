"""Shared settings for all three VaR models. Change values here to re-run with a different portfolio."""
import numpy as np

TICKERS = ["JPM", "MS", "BAC"]          # JPMorgan Chase, Morgan Stanley, Bank of America
WEIGHTS = np.array([1 / 3, 1 / 3, 1 / 3])  # portfolio weights (must sum to 1, same order as TICKERS)
PORTFOLIO_VALUE = 1_000_000             # $ invested
HOLDING_DAYS = 1                        # VaR horizon in trading days (scaled by sqrt(T))
CONFIDENCE_LEVELS = [0.90, 0.95, 0.99]
LOOKBACK = "2y"                         # Yahoo Finance history window
N_SIMS = 10_000                         # Monte Carlo scenarios
SEED = 42                               # fixed seed so Monte Carlo results are reproducible

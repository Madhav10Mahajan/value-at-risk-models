"""
STEP 1: Collect data   - daily adjusted close prices from Yahoo Finance.
STEP 2: Calculate returns - simple daily returns R_t = P_t / P_(t-1) - 1.
"""
import yfinance as yf

from config import LOOKBACK, TICKERS


def load_prices(period=LOOKBACK):
    # auto_adjust=True -> prices already adjusted for dividends and splits
    prices = yf.download(TICKERS, period=period, auto_adjust=True, progress=False)["Close"]
    return prices[TICKERS].dropna()


def load_returns(period=LOOKBACK):
    prices = load_prices(period)
    returns = prices.pct_change().dropna()  # first row has no previous price, so drop it
    print(f"[Step 1-2] {prices.index[0].date()} -> {prices.index[-1].date()} | "
          f"{len(prices)} prices, {len(returns)} daily returns")
    return returns

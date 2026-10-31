"""
METHOD 1: Variance-Covariance (parametric / delta-normal) VaR

Steps
  1-2. Collect prices and compute daily returns (data.py)
  3.   Mean and standard deviation of each stock's returns
  4.   Covariance matrix (how the stocks move together)
  5.   Portfolio mean  mu_p = w'mu  and std dev  sigma_p = sqrt(w' COV w)
  6.   Z-score for each confidence level, Z = N^-1(CI)
  7.   VaR = (Z * sigma_p - mu_p) * V * sqrt(T)

Assumptions
  - Returns are normally distributed
  - Mean, volatility and correlations are constant over the window/horizon
  - Positions are linear (no options); returns are i.i.d. (sqrt-of-time rule)

Pros
  - Simple, fast, transparent; diversification is explicit through the covariance matrix
  - Easy to change confidence level / horizon and to decompose risk by position

Cons
  - Normality ignores fat tails, so 99% VaR is usually understated
  - Not suitable for non-linear instruments; covariance estimates are unstable
"""
from statistics import NormalDist

import numpy as np
import pandas as pd

from config import CONFIDENCE_LEVELS, HOLDING_DAYS, PORTFOLIO_VALUE, TICKERS, WEIGHTS


def run(returns, verbose=True):
    # Step 3: per-stock statistics (ddof=1 -> sample std dev, like Excel STDEV.S)
    mu = returns.mean().values
    sd = returns.std(ddof=1).values

    # Step 4: sample covariance matrix (like Excel COVARIANCE.S)
    cov = returns.cov().values

    # Step 5: portfolio-level mean and volatility
    mu_p = WEIGHTS @ mu
    sigma_p = np.sqrt(WEIGHTS @ cov @ WEIGHTS)
    sigma_undiversified = WEIGHTS @ sd  # what risk would be if correlations were all 1

    if verbose:
        print("\n=== METHOD 1: VARIANCE-COVARIANCE ===")
        print("[Step 3] Mean / std dev of daily returns:")
        print(pd.DataFrame({"mean": mu, "std": sd}, index=TICKERS).map(lambda x: f"{x:.4%}"))
        print("[Step 4] Correlation matrix:")
        print(returns.corr().round(4))
        print(f"[Step 5] Portfolio mean {mu_p:.4%} | portfolio std dev {sigma_p:.4%} "
              f"| undiversified std dev {sigma_undiversified:.4%}")

    results = {}
    for ci in CONFIDENCE_LEVELS:
        # Step 6: Z-score (1.2816 for 90%, 1.6449 for 95%, 2.3263 for 99%)
        z = NormalDist().inv_cdf(ci)
        # Step 7: VaR as a positive $ loss, scaled to the holding period
        var = (z * sigma_p - mu_p) * np.sqrt(HOLDING_DAYS) * PORTFOLIO_VALUE
        results[ci] = var
        if verbose:
            div_benefit = z * (sigma_undiversified - sigma_p) * np.sqrt(HOLDING_DAYS) * PORTFOLIO_VALUE
            print(f"[Step 6-7] CI {ci:.0%}: Z = {z:.4f} -> VaR = ${var:,.0f} "
                  f"(diversification benefit ${div_benefit:,.0f})")
    return results


if __name__ == "__main__":
    from data import load_returns
    run(load_returns())

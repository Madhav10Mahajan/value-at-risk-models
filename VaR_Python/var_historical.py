"""
METHOD 2: Historical Simulation VaR

Steps
  1-2. Collect prices and compute daily returns (data.py)
  3.   Portfolio return for every historical day = sum(weight_i * return_i)
       (each past day is a scenario replayed on today's portfolio)
  4.   Sort portfolio returns from worst to best
  5.   VaR = -(1 - CI) percentile of those returns * V * sqrt(T)
  6.   Bonus: Expected Shortfall (CVaR) = average loss on days worse than VaR

Assumptions
  - The historical window is representative of the future
  - Every historical day is equally likely
  - No distribution is assumed (non-parametric)

Pros
  - Intuitive and easy to explain
  - Captures fat tails, skew and real correlations in the data

Cons
  - Cannot see events that are not in the window
  - Very few observations in the tail (99% VaR rests on ~5 points out of 500)
  - Slow to react to changes in volatility
"""
import numpy as np

from config import CONFIDENCE_LEVELS, HOLDING_DAYS, PORTFOLIO_VALUE, WEIGHTS


def run(returns, verbose=True):
    # Step 3: daily portfolio returns
    port = returns.values @ WEIGHTS

    # Step 4: sort worst -> best
    sorted_ret = np.sort(port)
    if verbose:
        print("\n=== METHOD 2: HISTORICAL SIMULATION ===")
        print(f"[Step 3] {len(port)} historical portfolio returns")
        print("[Step 4] 5 worst days:", ", ".join(f"{x:.2%}" for x in sorted_ret[:5]))

    results = {}
    for ci in CONFIDENCE_LEVELS:
        # Step 5: the (1-CI) percentile; numpy's default linear interpolation = Excel PERCENTILE.INC
        cutoff = np.percentile(port, (1 - ci) * 100)
        var = -cutoff * np.sqrt(HOLDING_DAYS) * PORTFOLIO_VALUE
        # Step 6: average of the returns in the tail beyond VaR
        es = -port[port <= cutoff].mean() * np.sqrt(HOLDING_DAYS) * PORTFOLIO_VALUE
        results[ci] = var
        if verbose:
            print(f"[Step 5-6] CI {ci:.0%}: {1 - ci:.0%} percentile = {cutoff:.3%} "
                  f"-> VaR = ${var:,.0f} | ES = ${es:,.0f}")
    return results


if __name__ == "__main__":
    from data import load_returns
    run(load_returns())

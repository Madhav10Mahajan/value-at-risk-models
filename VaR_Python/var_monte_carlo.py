"""
METHOD 3: Monte Carlo Simulation VaR (correlated multivariate normal)

Steps
  1-2. Collect prices and compute daily returns (data.py)
  3.   Estimate mean vector and covariance matrix from history
  4.   Cholesky decomposition COV = L L'  (turns independent shocks into correlated ones)
  5.   Draw N_SIMS x 3 independent standard normals Z; correlated returns R = mu + Z L'
  6.   Portfolio return for each scenario = sum(weight_i * R_i)
  7.   VaR = -(1 - CI) percentile of simulated returns * V * sqrt(T); also Expected Shortfall

Assumptions
  - Returns follow the chosen model (multivariate normal here) with parameters from history
  - Parameters are constant over the horizon
  - Enough simulations for the percentile to converge

Pros
  - Most flexible: any distribution (t, GARCH, jumps), non-linear products, path dependency
  - Produces a full P&L distribution (VaR, ES, any percentile)

Cons
  - Computationally heavy; results vary with the random seed (sampling error)
  - Model risk: with a normal model the answer is close to Variance-Covariance
"""
import numpy as np

from config import CONFIDENCE_LEVELS, HOLDING_DAYS, N_SIMS, PORTFOLIO_VALUE, SEED, WEIGHTS


def run(returns, n_sims=N_SIMS, seed=SEED, verbose=True):
    # Step 3: parameters estimated from history
    mu = returns.mean().values
    cov = returns.cov().values

    # Step 4: lower-triangular L with L @ L.T == cov
    L = np.linalg.cholesky(cov)

    # Step 5: independent N(0,1) shocks -> correlated returns
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_sims, len(mu)))
    sim_returns = mu + z @ L.T

    # Step 6: simulated portfolio returns
    port = sim_returns @ WEIGHTS

    if verbose:
        print("\n=== METHOD 3: MONTE CARLO SIMULATION ===")
        print("[Step 4] Cholesky L:\n", np.round(L, 6))
        # sanity check: simulated correlations should match historical ones
        print("[Step 5] Simulated correlation:\n", np.round(np.corrcoef(sim_returns.T), 4))
        print(f"[Step 6] {n_sims:,} scenarios | mean {port.mean():.4%} | std {port.std(ddof=1):.4%}")

    results = {}
    for ci in CONFIDENCE_LEVELS:
        # Step 7
        cutoff = np.percentile(port, (1 - ci) * 100)
        var = -cutoff * np.sqrt(HOLDING_DAYS) * PORTFOLIO_VALUE
        es = -port[port <= cutoff].mean() * np.sqrt(HOLDING_DAYS) * PORTFOLIO_VALUE
        results[ci] = var
        if verbose:
            print(f"[Step 7] CI {ci:.0%}: VaR = ${var:,.0f} | ES = ${es:,.0f}")
    return results


if __name__ == "__main__":
    from data import load_returns
    run(load_returns())

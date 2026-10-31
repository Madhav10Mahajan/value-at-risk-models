"""
BACKTESTING the three VaR models  (Traffic Light approach + Kupiec POF test)

Idea: a VaR model is only useful if the number of days on which the actual loss
exceeded VaR ("exceptions" / "breaches") is consistent with the confidence level.
At 99% we expect losses > VaR on ~1% of days (2.5 days out of 250).

Steps
  1. Collect prices & returns (data.py) and form daily portfolio returns.
  2. Rolling window: for each test day t, estimate VaR using ONLY the previous
     WINDOW (250) days -> genuine out-of-sample forecast (no look-ahead bias).
  3. Compare the actual P&L on day t with that day's VaR; exception = 1 if loss > VaR.
  4. Count exceptions over the test period for each method & confidence level.
  5. Traffic Light (Basel): zone from cumulative binomial probability of seeing
     <= x exceptions if the model were correct:
        Green  < 95%       Yellow 95% - 99.99%       Red >= 99.99%
     At 99% / 250 days this gives Basel's table: Green 0-4, Yellow 5-9, Red 10+.
  6. Kupiec Proportion-of-Failures (POF) likelihood-ratio test:
        H0: actual exception rate = expected rate p (= 1 - CI)
        LR = -2 ln[(1-p)^(T-x) p^x] + 2 ln[(1-x/T)^(T-x) (x/T)^x]  ~ chi-square(1)
        Reject H0 if LR > 3.841 (5% significance), i.e. p-value < 5%.

Assumptions
  - Exceptions are independent Bernoulli trials with probability p (Kupiec ignores clustering).
  - Portfolio weights are held constant over the test period.
  - The test period (250 days) is long enough - with few observations the tests have low power.

Pros
  - Traffic Light: simple, regulatory standard (Basel), maps directly to capital multiplier.
  - Kupiec: formal statistical test; penalises both too many AND too few exceptions.

Cons
  - Both only look at the COUNT of exceptions, not their timing (clustering)
    or size -> use Christoffersen's independence test / ES backtests for that.
  - Low statistical power on 250 observations, especially at 99%.
"""
import math
from statistics import NormalDist

import numpy as np
import pandas as pd

from config import CONFIDENCE_LEVELS, N_SIMS, PORTFOLIO_VALUE, SEED, TICKERS, WEIGHTS
from data import load_returns

WINDOW = 250              # estimation window (trading days) for each VaR forecast
CHI2_CRITICAL = 3.841     # chi-square(1) critical value at 5% significance
BASEL_PLUS_FACTOR = {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}  # 99% / 250-day table


# ---------------------------------------------------------------------------
# Step 2-3: rolling out-of-sample VaR forecasts and exceptions
# ---------------------------------------------------------------------------
def rolling_var(returns, window=WINDOW):
    port = returns.values @ WEIGHTS
    # Common random numbers: the same standard-normal draws are reused every day,
    # so day-to-day changes in MC VaR come from the data, not simulation noise.
    z = np.random.default_rng(SEED).standard_normal((N_SIMS, len(TICKERS)))

    rows = []
    for t in range(window, len(returns)):
        win = returns.iloc[t - window:t]          # only data known BEFORE day t
        mu, cov = win.mean().values, win.cov().values
        mu_p, sigma_p = WEIGHTS @ mu, np.sqrt(WEIGHTS @ cov @ WEIGHTS)
        win_port = port[t - window:t]
        mc_port = (mu + z @ np.linalg.cholesky(cov).T) @ WEIGHTS

        row = {"Date": returns.index[t], "Return": port[t], "PnL": port[t] * PORTFOLIO_VALUE}
        for ci in CONFIDENCE_LEVELS:
            q = (1 - ci) * 100
            row[f"VC_{ci:.0%}"] = (NormalDist().inv_cdf(ci) * sigma_p - mu_p) * PORTFOLIO_VALUE
            row[f"HS_{ci:.0%}"] = -np.percentile(win_port, q) * PORTFOLIO_VALUE
            row[f"MC_{ci:.0%}"] = -np.percentile(mc_port, q) * PORTFOLIO_VALUE
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Date")
    # exception = 1 when the actual loss is bigger than the forecast VaR
    for col in [c for c in df.columns if c[:3] in ("VC_", "HS_", "MC_")]:
        df[f"EXC_{col}"] = (-df["PnL"] > df[col]).astype(int)
    return df


# ---------------------------------------------------------------------------
# Steps 5-6: the two tests
# ---------------------------------------------------------------------------
def kupiec_pof(x, T, p):
    """Kupiec POF likelihood ratio and p-value (chi-square, 1 dof)."""
    log_h0 = (T - x) * math.log(1 - p) + x * math.log(p)
    phat = x / T
    # x*ln(x/T) -> 0 as x -> 0 (same for T-x), so skip those terms to avoid log(0)
    log_h1 = ((T - x) * math.log(1 - phat) if x < T else 0) + (x * math.log(phat) if x > 0 else 0)
    lr = -2 * (log_h0 - log_h1)
    p_value = math.erfc(math.sqrt(lr / 2))  # = 1 - CDF of chi-square(1)
    return lr, p_value


def traffic_light(x, T, p):
    """Basel zone from cumulative binomial probability P(X <= x)."""
    cum = sum(math.comb(T, k) * p ** k * (1 - p) ** (T - k) for k in range(x + 1))
    zone = "GREEN" if cum < 0.95 else ("YELLOW" if cum < 0.9999 else "RED")
    return zone, cum


def summarise(df):
    names = {"VC": "Variance-Covariance", "HS": "Historical", "MC": "Monte Carlo"}
    T = len(df)
    out = []
    for m, name in names.items():
        for ci in CONFIDENCE_LEVELS:
            p = 1 - ci
            x = int(df[f"EXC_{m}_{ci:.0%}"].sum())        # Step 4: count exceptions
            lr, pv = kupiec_pof(x, T, p)
            zone, cum = traffic_light(x, T, p)
            # Basel plus factor (capital multiplier = 3 + plus) is defined only for 99% / 250 days
            plus = BASEL_PLUS_FACTOR.get(x, 0.0 if x < 5 else 1.0) if (ci == 0.99 and T == 250) else ""
            out.append({
                "Method": name, "CI": f"{ci:.0%}", "Days": T,
                "Expected": round(p * T, 1), "Actual": x, "Rate": f"{x / T:.2%}",
                "Kupiec LR": round(lr, 3), "p-value": round(pv, 4),
                "Kupiec @5%": "Reject" if lr > CHI2_CRITICAL else "Accept",
                "Cum. binom": f"{cum:.2%}", "Traffic light": zone, "Basel plus": plus,
            })
    return pd.DataFrame(out)


def plot(df, path="backtest_99.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed - skipping chart)")
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(df.index, df["PnL"], color="grey", label="Actual daily P&L")
    for m, c in (("VC", "tab:blue"), ("HS", "tab:orange"), ("MC", "tab:green")):
        ax.plot(df.index, -df[f"{m}_99%"], color=c, label=f"-VaR 99% ({m})")
    exc = df[df["EXC_HS_99%"] == 1]
    ax.scatter(exc.index, exc["PnL"], color="red", zorder=5, label="Exception (HS 99%)")
    ax.set_title("VaR backtest: daily P&L vs 99% VaR (rolling 250-day window)")
    ax.set_ylabel("$")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"Chart saved -> {path}")


def main():
    returns = load_returns()
    if len(returns) < WINDOW + 50:
        raise SystemExit("Not enough history - increase LOOKBACK in config.py")
    df = rolling_var(returns)
    print(f"[Step 2] Rolling {WINDOW}-day window -> {len(df)} out-of-sample test days "
          f"({df.index[0].date()} to {df.index[-1].date()})")
    print("\n=== BACKTEST RESULTS ===")
    print(summarise(df).to_string(index=False))
    df.to_csv("backtest_daily.csv")
    print("\nDaily VaR forecasts & exceptions saved -> backtest_daily.csv")
    plot(df)


if __name__ == "__main__":
    main()

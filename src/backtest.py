"""
Out-of-sample backtest harness for the QAOA portfolio — Week 6.

The QAOA/QUBO solution (bitstring 00001111 -> HSBA, AZN, SHEL, ULVR) is the exact
ground state of the cardinality-constrained mean-variance QUBO, i.e. the classical
optimum under the 4-asset constraint. This module asks the only question a FinTech
reader cares about: does that optimum hold up on data it was NOT fitted to?

Construction (train window) -> evaluation (test window). All weights are fixed at
the train/test boundary and held through the test period (buy-and-hold), so we are
measuring genuine out-of-sample behaviour, not look-ahead.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def load_prices(tickers, start, end):
    """Adjusted close prices via yfinance. Run this in an env with network + yfinance."""
    import yfinance as yf
    px = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False)["Close"]
    return px[list(tickers)].dropna(how="all")


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="any")


def bitstring_to_weights(bitstring: str, n: int) -> np.ndarray:
    """Little-endian bitstring -> equal weights over selected assets (bit==1)."""
    sel = np.array([int(c) for c in bitstring[::-1]], dtype=float)   # sel[i] = asset i
    if sel.sum() == 0:
        raise ValueError("empty selection")
    return sel / sel.sum()


def _solve_long_only(objective, n):
    """Minimise `objective(w)` s.t. sum(w)=1, w>=0 (long-only, fully invested)."""
    from scipy.optimize import minimize
    res = minimize(objective, np.ones(n) / n, method="SLSQP",
                   bounds=[(0.0, 1.0)] * n,
                   constraints=({"type": "eq", "fun": lambda w: w.sum() - 1.0},),
                   options={"ftol": 1e-12, "maxiter": 500})
    return res.x


def min_variance_weights(cov: np.ndarray) -> np.ndarray:
    """Long-only minimum-variance portfolio (objective scaled for solver conditioning)."""
    s = 1.0 / np.mean(np.diag(cov))          # scale variance to O(1) so SLSQP sees gradients
    return _solve_long_only(lambda w: s * (w @ cov @ w), len(cov))


def max_sharpe_weights(mu: np.ndarray, cov: np.ndarray, rf_daily: float = 0.0) -> np.ndarray:
    """Long-only maximum-Sharpe (tangency) portfolio."""
    def neg_sharpe(w):
        v = np.sqrt(w @ cov @ w)
        return -(w @ mu - rf_daily) / v if v > 1e-12 else 0.0
    return _solve_long_only(neg_sharpe, len(mu))


def portfolio_path(test_returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    return test_returns.values @ weights


def performance_metrics(port_daily: np.ndarray, rf_annual: float = 0.0, periods: int = 252) -> dict:
    port_daily = np.asarray(port_daily)
    equity = np.cumprod(1 + port_daily)
    cum = equity[-1] - 1
    ann_ret = (1 + cum) ** (periods / len(port_daily)) - 1
    ann_vol = port_daily.std(ddof=1) * np.sqrt(periods)
    sharpe = (ann_ret - rf_annual) / ann_vol if ann_vol > 0 else np.nan
    peak = np.maximum.accumulate(equity)
    max_dd = (equity / peak - 1).min()
    return {"cum_return": cum, "ann_return": ann_ret, "ann_vol": ann_vol,
            "sharpe": sharpe, "max_drawdown": max_dd}


def run_backtest(prices: pd.DataFrame, train_end: str, selected_bitstring: str,
                 rf_annual: float = 0.0) -> dict:
    """Build weights on prices[:train_end], evaluate on prices[train_end:]."""
    rets = daily_returns(prices)
    train = rets.loc[:train_end]
    test  = rets.loc[train_end:].iloc[1:]      # drop boundary day
    n = prices.shape[1]
    mu, cov = train.mean().values, train.cov().values

    weights = {
        "QAOA / QUBO optimum (4 assets)": bitstring_to_weights(selected_bitstring, n),
        "Naive 1/N (8 assets)":           np.ones(n) / n,
        "Min-variance (8 assets)":        min_variance_weights(cov),
        "Max-Sharpe (8 assets)":          max_sharpe_weights(mu, cov, rf_annual / 252),
    }
    out = {}
    for name, w in weights.items():
        path = portfolio_path(test, w)
        out[name] = {"weights": w, "daily": path,
                     "equity": np.cumprod(1 + path),
                     **performance_metrics(path, rf_annual)}
    out["_test_index"] = test.index
    return out

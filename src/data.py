"""
Data loading and preprocessing for the Diarka QAOA portfolio project.

This module isolates everything that touches the market-data layer so the rest
of the codebase (encoding, solvers, notebooks) can stay deterministic and easy
to test. The universe is a small basket of FTSE 100 constituents chosen to
provide sector diversity without making the covariance matrix trivial.

References
----------
yfinance Yahoo Finance scraper: https://github.com/ranaroussi/yfinance
Annualisation convention: 252 trading days per year (UK & US equity standard).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
# Eight FTSE 100 constituents spanning seven sectors. VOD is included alongside
# BT-A deliberately so the covariance matrix carries a strong intra-sector
# correlation block — this gives QAOA a non-trivial optimisation to perform
# rather than a problem the optimiser can solve by inspection.
FTSE_UNIVERSE: list[str] = [
    "HSBA.L",   # Banking
    "AZN.L",    # Pharmaceuticals
    "SHEL.L",   # Energy
    "ULVR.L",   # Consumer staples
    "RIO.L",    # Mining / materials
    "BT-A.L",   # Telecoms
    "VOD.L",    # Telecoms (deliberate sector overlap)
    "DGE.L",    # Beverages / consumer discretionary
]

TRADING_DAYS_PER_YEAR: int = 252


@dataclass(frozen=True)
class PortfolioStats:
    """Annualised mean returns and covariance for a universe of assets."""

    tickers: tuple[str, ...]
    mu: np.ndarray              # shape (n,) — annualised expected log returns
    sigma: np.ndarray           # shape (n, n) — annualised covariance matrix
    start: pd.Timestamp
    end: pd.Timestamp
    n_observations: int

    @property
    def n_assets(self) -> int:
        return len(self.tickers)

    def to_npz(self, path: str | Path) -> None:
        """Persist to a single .npz file for downstream sessions."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            tickers=np.array(self.tickers),
            mu=self.mu,
            sigma=self.sigma,
            start=str(self.start.date()),
            end=str(self.end.date()),
            n_observations=self.n_observations,
        )

    @classmethod
    def from_npz(cls, path: str | Path) -> "PortfolioStats":
        data = np.load(path, allow_pickle=False)
        return cls(
            tickers=tuple(str(t) for t in data["tickers"]),
            mu=data["mu"],
            sigma=data["sigma"],
            start=pd.Timestamp(str(data["start"])),
            end=pd.Timestamp(str(data["end"])),
            n_observations=int(data["n_observations"]),
        )


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------
def fetch_prices(
    tickers: Iterable[str] = FTSE_UNIVERSE,
    *,
    years: int = 4,
    end: date | None = None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Download daily adjusted close prices from Yahoo Finance.

    Parameters
    ----------
    tickers : iterable of str
        Yahoo Finance ticker symbols (e.g. ``"HSBA.L"``).
    years : int, default 4
        Length of the historical window in calendar years.
    end : datetime.date or None
        End date (inclusive). Defaults to today.
    auto_adjust : bool, default True
        Apply Yahoo's adjustments for splits/dividends.

    Returns
    -------
    pandas.DataFrame
        One column per ticker, indexed by trading date, no missing rows.
    """
    tickers = list(tickers)
    end = end or date.today()
    start = end - timedelta(days=int(years * 365.25))

    raw = yf.download(
        tickers,
        start=start.isoformat(),
        end=end.isoformat(),
        auto_adjust=auto_adjust,
        progress=False,
        threads=False,        # Serialise requests to avoid yfinance SQLite cache locks.
        group_by="column",
    )

    # yfinance returns a MultiIndex when more than one ticker is requested.
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    # Re-order columns to match the requested universe and drop any all-NaN cols.
    prices = prices[[t for t in tickers if t in prices.columns]]

    # Forward-fill short gaps (single missing close), then drop any remaining
    # rows where at least one ticker has no observation. For an 8-asset, 4-year
    # FTSE pull this typically discards < 1% of rows.
    prices = prices.ffill(limit=1).dropna(how="any")
    prices.index = pd.to_datetime(prices.index)

    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns from a price panel.

    Log returns are used (rather than simple returns) because they are
    time-additive, which simplifies the annualisation step and is the
    convention in academic portfolio optimisation literature.
    """
    return np.log(prices / prices.shift(1)).dropna()


def annualise(
    log_returns: pd.DataFrame,
    *,
    periods: int = TRADING_DAYS_PER_YEAR,
) -> PortfolioStats:
    """Compute annualised mean vector and covariance matrix from log returns."""
    if log_returns.empty:
        raise ValueError("Log-returns frame is empty; check the data pull.")

    tickers = tuple(log_returns.columns)
    mu = log_returns.mean().to_numpy() * periods
    sigma = log_returns.cov().to_numpy() * periods

    # Symmetrise to suppress floating-point asymmetry — required by cvxpy's PSD
    # check and by qiskit-finance's PortfolioOptimization class.
    sigma = 0.5 * (sigma + sigma.T)

    return PortfolioStats(
        tickers=tickers,
        mu=mu,
        sigma=sigma,
        start=log_returns.index.min(),
        end=log_returns.index.max(),
        n_observations=len(log_returns),
    )

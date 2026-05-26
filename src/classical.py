"""
Classical baseline solvers for the QAOA portfolio project.

Three solvers are provided, each targeting a slightly different version of the
mean-variance portfolio problem. The point of having all three is to give the
QAOA results in later sessions a robust set of reference answers to be measured
against.

1. ``solve_continuous_markowitz``
   Standard convex quadratic programme: continuous weights summing to one.
   Solved with cvxpy. This is the textbook efficient-frontier solution and
   functions as a sanity check on the data and the covariance matrix.

2. ``solve_binary_brute_force``
   Discrete cardinality-constrained selection: pick exactly ``budget`` assets
   from ``n``, all equally weighted. For ``n = 8`` there are only 256 subsets,
   so brute-force enumeration is trivial and gives the *ground-truth* binary
   answer that QAOA must approximate.

3. ``solve_binary_qiskit_finance``
   Identical formulation to (2), but solved via the qiskit-finance
   ``PortfolioOptimization`` application paired with a classical
   ``NumPyMinimumEigensolver``. This is the cleanest validation that our
   QUBO encoding (Session 2) and the qiskit-finance encoding agree, before
   the same ``QuadraticProgram`` is handed to QAOA.

The qiskit-finance objective for the binary problem is

    minimise   q · xᵀ Σ x  −  μᵀ x
    subject to ∑ xᵢ = budget,   xᵢ ∈ {0, 1}

where ``q`` is the risk-aversion (``risk_factor``) parameter. Solvers (2) and
(3) use this same formulation so the resulting selections should agree exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PortfolioSolution:
    """Container for a portfolio optimisation result.

    Attributes
    ----------
    selection : np.ndarray
        Binary or continuous weight vector of length ``n``.
    expected_return : float
        ``μᵀ x`` evaluated at the solution.
    expected_risk : float
        ``√(xᵀ Σ x)`` evaluated at the solution (i.e. annualised volatility).
    objective : float
        Value of the optimiser's objective at the solution.
    solver : str
        Short identifier (``"continuous"``, ``"brute_force"``, ``"qiskit_finance"``).
    tickers : tuple[str, ...] | None
        Optional ticker labels for ergonomic printing.
    """

    selection: np.ndarray
    expected_return: float
    expected_risk: float
    objective: float
    solver: str
    tickers: tuple[str, ...] | None = None

    def pretty(self) -> str:
        lines = [
            f"Solver:           {self.solver}",
            f"Expected return:  {self.expected_return:.4f}  (annualised)",
            f"Expected risk:    {self.expected_risk:.4f}  (annualised vol)",
            f"Sharpe (rf=0):    {self.expected_return / self.expected_risk:.3f}",
            f"Objective value:  {self.objective:.6f}",
        ]
        if self.tickers is not None:
            picks = [
                t for t, w in zip(self.tickers, self.selection) if w > 0.5
            ]
            lines.append(f"Selected:         {', '.join(picks)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. Continuous Markowitz (cvxpy)
# ---------------------------------------------------------------------------
def solve_continuous_markowitz(
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    risk_aversion: float = 0.5,
    tickers: Sequence[str] | None = None,
) -> PortfolioSolution:
    """Solve the continuous, long-only, fully-invested Markowitz problem.

        minimise   q · wᵀ Σ w  −  μᵀ w
        subject to ∑ wᵢ = 1,   wᵢ ≥ 0
    """
    import cvxpy as cp  # local import keeps top-level import time small

    n = len(mu)
    w = cp.Variable(n, nonneg=True)
    objective = cp.Minimize(
        risk_aversion * cp.quad_form(w, cp.psd_wrap(sigma)) - mu @ w
    )
    constraints = [cp.sum(w) == 1]
    cp.Problem(objective, constraints).solve()

    w_opt = np.asarray(w.value).ravel()
    return PortfolioSolution(
        selection=w_opt,
        expected_return=float(mu @ w_opt),
        expected_risk=float(np.sqrt(w_opt @ sigma @ w_opt)),
        objective=float(
            risk_aversion * (w_opt @ sigma @ w_opt) - mu @ w_opt
        ),
        solver="continuous_markowitz_cvxpy",
        tickers=tuple(tickers) if tickers is not None else None,
    )


# ---------------------------------------------------------------------------
# 2. Brute-force binary k-of-n
# ---------------------------------------------------------------------------
def solve_binary_brute_force(
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    budget: int,
    risk_factor: float = 0.5,
    tickers: Sequence[str] | None = None,
) -> PortfolioSolution:
    """Enumerate all ``C(n, budget)`` subsets and return the optimum.

    The objective matches qiskit-finance's ``PortfolioOptimization``:

        minimise   q · xᵀ Σ x  −  μᵀ x       with   ∑ xᵢ = budget.
    """
    n = len(mu)
    if not 0 < budget <= n:
        raise ValueError(f"budget must satisfy 0 < budget <= {n}, got {budget}.")

    best_x = None
    best_val = np.inf
    for picks in combinations(range(n), budget):
        x = np.zeros(n)
        x[list(picks)] = 1.0
        val = risk_factor * (x @ sigma @ x) - mu @ x
        if val < best_val:
            best_val = val
            best_x = x

    assert best_x is not None
    return PortfolioSolution(
        selection=best_x,
        expected_return=float(mu @ best_x),
        expected_risk=float(np.sqrt(best_x @ sigma @ best_x)),
        objective=float(best_val),
        solver="brute_force_binary",
        tickers=tuple(tickers) if tickers is not None else None,
    )


# ---------------------------------------------------------------------------
# 3. qiskit-finance binary (NumPyMinimumEigensolver — classical)
# ---------------------------------------------------------------------------
def solve_binary_qiskit_finance(
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    budget: int,
    risk_factor: float = 0.5,
    tickers: Sequence[str] | None = None,
) -> PortfolioSolution:
    """Solve the same binary problem as (2) via the qiskit-finance stack.

    Uses ``PortfolioOptimization`` to build a ``QuadraticProgram``, converts it
    to an Ising Hamiltonian under the hood, and then solves it exactly with
    ``NumPyMinimumEigensolver``. The result should match
    ``solve_binary_brute_force`` for any well-posed input.
    """
    from qiskit_algorithms import NumPyMinimumEigensolver
    from qiskit_finance.applications.optimization import PortfolioOptimization
    from qiskit_optimization.algorithms import MinimumEigenOptimizer

    portfolio = PortfolioOptimization(
        expected_returns=mu,
        covariances=sigma,
        risk_factor=risk_factor,
        budget=budget,
    )
    qp = portfolio.to_quadratic_program()

    solver = MinimumEigenOptimizer(NumPyMinimumEigensolver())
    result = solver.solve(qp)
    x = np.asarray(result.x, dtype=float)

    return PortfolioSolution(
        selection=x,
        expected_return=float(mu @ x),
        expected_risk=float(np.sqrt(x @ sigma @ x)),
        objective=float(result.fval),
        solver="qiskit_finance_numpy_eigensolver",
        tickers=tuple(tickers) if tickers is not None else None,
    )

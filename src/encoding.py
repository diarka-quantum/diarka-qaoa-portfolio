"""
QUBO and Ising encoding for the binary portfolio optimisation problem.

This module performs by hand the encoding step that Session 1 deliberately
delegated to qiskit-finance's ``PortfolioOptimization`` class. The point isn't
to replace that class — it does exactly the same job. The point is to make
the encoding explicit so we can reason about what QAOA is actually doing in
later sessions, and so the algebra is documented and reviewable rather than
buried inside a third-party black box.

Two reductions are performed:

    Cardinality-constrained mean-variance
    ─────────────────────────────────────►  QUBO  ──►  Ising Hamiltonian

with cross-checks at every step that the encoding reproduces the brute-force
ground truth from Session 1.

References
----------
Lucas, A. (2014). "Ising formulations of many NP problems".
Frontiers in Physics 2, 5. doi:10.3389/fphy.2014.00005
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from qiskit.quantum_info import SparsePauliOp


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class QUBOFormulation:
    """A QUBO problem in symmetric-matrix form.

    Objective:
        f(x) = xᵀ Q x  +  offset,        x ∈ {0, 1}ⁿ

    The diagonal of Q absorbs the linear coefficients because x_i² = x_i for
    any binary x.
    """

    Q: np.ndarray                          # shape (n, n) symmetric
    offset: float                          # constant added to objective
    n: int                                 # number of binary variables
    tickers: tuple[str, ...] | None = None
    budget: int | None = None
    risk_factor: float | None = None
    penalty: float | None = None

    # ------------------------------------------------------------------ eval
    def evaluate(self, x: np.ndarray) -> float:
        """Compute f(x) = xᵀQx + offset for a single binary vector."""
        x = np.asarray(x, dtype=float)
        return float(x @ self.Q @ x + self.offset)

    def evaluate_all(self) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate f(x) on every x ∈ {0,1}ⁿ.

        Returns
        -------
        bitstrings : np.ndarray, shape (2ⁿ, n)
            Each row is a binary vector. Row index ``k`` corresponds to the
            integer ``k`` written in big-endian binary
            (so ``bitstrings[k, 0]`` is the most-significant bit).
        values : np.ndarray, shape (2ⁿ,)
            ``values[k] = f(bitstrings[k])``.
        """
        n = self.n
        bitstrings = np.array(
            [[(k >> (n - 1 - i)) & 1 for i in range(n)] for k in range(2 ** n)],
            dtype=float,
        )
        values = np.einsum("ki,ij,kj->k", bitstrings, self.Q, bitstrings) + self.offset
        return bitstrings, values


# ---------------------------------------------------------------------------
# QUBO construction
# ---------------------------------------------------------------------------
def choose_penalty(
    mu: np.ndarray,
    sigma: np.ndarray,
    risk_factor: float,
    safety_multiplier: float = 2.0,
) -> float:
    """Pick a penalty large enough to dominate any infeasible bonus.

    The cardinality constraint ``∑ xᵢ = B`` becomes a quadratic penalty
    ``P · (∑ xᵢ − B)²``. We want ``P`` big enough that even the
    most-attractive infeasible solution can't beat the worst feasible one.

    Heuristic: bound the largest improvement an infeasible bitstring could
    offer by ``|μ|_∞ + q · |Σ|_∞`` and multiply by a safety factor.
    """
    mu_scale = float(np.max(np.abs(mu)))
    sigma_scale = float(np.max(np.abs(sigma)))
    return safety_multiplier * (mu_scale + risk_factor * sigma_scale)


def build_qubo(
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    budget: int,
    risk_factor: float = 0.5,
    penalty: float | None = None,
    tickers: Sequence[str] | None = None,
) -> QUBOFormulation:
    """Encode the cardinality-constrained mean-variance problem as a QUBO.

    Starting from

        minimise   q · xᵀ Σ x  −  μᵀ x
        subject to ∑ xᵢ = B,   xᵢ ∈ {0, 1}

    we absorb the budget constraint as a quadratic penalty:

        minimise   q · xᵀ Σ x  −  μᵀ x  +  P · (∑ xᵢ − B)²

    Expanding (∑ xᵢ − B)² = ∑ᵢⱼ xᵢxⱼ − 2B ∑ xᵢ + B², the result is a QUBO

        f(x) = xᵀ Q x  +  offset

    with off-diagonal ``Q_ij = q Σ_ij + P`` and diagonal that absorbs both
    the unconstrained linear term ``−μ`` and the penalty's linear part
    ``−2BP``, using the identity x_i² = x_i:

        Q_ii = q Σ_ii + P − μ_i − 2 B P
        offset = P · B²

    Parameters
    ----------
    mu, sigma : np.ndarray
        Annualised expected returns and covariance.
    budget : int
        Cardinality target ``B``.
    risk_factor : float
        Risk-aversion weight ``q``.
    penalty : float, optional
        Penalty coefficient ``P``. Defaults to ``choose_penalty(...)``.
    tickers : sequence of str, optional
        Labels for printing.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n = len(mu)

    if penalty is None:
        penalty = choose_penalty(mu, sigma, risk_factor)

    # Start from the unconstrained risk term qΣ.
    Q = risk_factor * sigma.copy()

    # Add the cardinality penalty's quadratic contribution: P on every (i, j).
    Q += penalty * np.ones((n, n))

    # Fold the linear contributions (−μ from the return term and −2BP from
    # the penalty) onto the diagonal using x_i² = x_i.
    diagonal = np.diag(Q) - mu - 2.0 * budget * penalty
    np.fill_diagonal(Q, diagonal)

    # Symmetrise — already symmetric mathematically, this just damps any
    # floating-point asymmetry inherited from Σ.
    Q = 0.5 * (Q + Q.T)

    offset = penalty * (budget ** 2)

    return QUBOFormulation(
        Q=Q,
        offset=offset,
        n=n,
        tickers=tuple(tickers) if tickers is not None else None,
        budget=budget,
        risk_factor=risk_factor,
        penalty=penalty,
    )


# ---------------------------------------------------------------------------
# QUBO → Ising Hamiltonian
# ---------------------------------------------------------------------------
def qubo_to_ising(qubo: QUBOFormulation) -> tuple[SparsePauliOp, float]:
    """Convert a QUBO into its Ising Hamiltonian.

    Substituting ``x_i = (1 − z_i) / 2`` with ``z_i ∈ {−1, +1}`` into
    ``f(x) = xᵀQx + c`` (Q symmetric) gives

        f = const_ising  +  ∑ᵢ hᵢ zᵢ  +  ∑_{i<j} J_ij zᵢ z_j

    where

        hᵢ          = −(1/2) · row_sum(Q, i)
        J_ij        = Q_ij / 2                          (for i < j)
        const_ising = (1/4) · sum(Q) + (1/4) · trace(Q) + qubo.offset

    The constant term does not affect optimisation but is required to map an
    Ising expectation value back to the original QUBO objective:

        ⟨H⟩ + offset_ising = f(x)

    In the computational basis the standard Qiskit convention ``|0⟩ ↔ z = +1``
    and ``|1⟩ ↔ z = −1`` means a measurement bitstring ``b`` corresponds
    directly to the binary vector ``x = b``. So the lowest-energy
    computational-basis state of ``H`` *is* the QUBO optimum.

    Returns
    -------
    hamiltonian : SparsePauliOp
        Diagonal in the computational basis. Acts on ``qubo.n`` qubits.
    offset_ising : float
        Add to ⟨H⟩ to recover the QUBO objective.
    """
    Q = qubo.Q
    n = qubo.n

    h = -0.5 * Q.sum(axis=1)
    const_ising = 0.25 * Q.sum() + 0.25 * float(np.trace(Q)) + qubo.offset

    pauli_list: list[tuple[str, float]] = []

    # Single-qubit Z terms. Qiskit Pauli strings are read RIGHT TO LEFT:
    # position 0 (rightmost char) is qubit 0. So to place Z on qubit i,
    # set position (n - 1 - i) of the string to "Z".
    for i in range(n):
        coeff = float(h[i])
        if abs(coeff) > 1e-12:
            label = ["I"] * n
            label[n - 1 - i] = "Z"
            pauli_list.append(("".join(label), coeff))

    # Two-qubit ZZ terms.
    for i in range(n):
        for j in range(i + 1, n):
            coeff = 0.5 * float(Q[i, j])
            if abs(coeff) > 1e-12:
                label = ["I"] * n
                label[n - 1 - i] = "Z"
                label[n - 1 - j] = "Z"
                pauli_list.append(("".join(label), coeff))

    hamiltonian = SparsePauliOp.from_list(pauli_list)
    return hamiltonian, const_ising


# ---------------------------------------------------------------------------
# Ground-state utilities
# ---------------------------------------------------------------------------
def hamiltonian_eigendecomposition(
    hamiltonian: SparsePauliOp,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted eigenvalues and corresponding bitstrings.

    Because the Hamiltonian here is composed only of I and Z Paulis, it is
    diagonal in the computational basis. The k-th eigenvalue corresponds to
    the basis state ``|k⟩`` (with Qiskit's little-endian bit ordering).
    """
    matrix = hamiltonian.to_matrix(sparse=False)
    diagonal = np.real(np.diag(matrix))

    n = hamiltonian.num_qubits
    bitstrings = np.array(
        [[(k >> i) & 1 for i in range(n)] for k in range(2 ** n)],  # little-endian
        dtype=int,
    )

    order = np.argsort(diagonal)
    return diagonal[order], bitstrings[order]


def bitstring_to_selection(bitstring: str, n: int) -> np.ndarray:
    """Translate a Qiskit measurement bitstring (right→left = q0→q_{n-1})
    into a numpy selection vector indexed by qubit number.

    Example: with n=4, bitstring "0101" means
        q0 = 1, q1 = 0, q2 = 1, q3 = 0
    so selection = [1, 0, 1, 0].
    """
    if len(bitstring) != n:
        raise ValueError(f"bitstring length {len(bitstring)} ≠ n={n}.")
    return np.array([int(c) for c in reversed(bitstring)], dtype=float)

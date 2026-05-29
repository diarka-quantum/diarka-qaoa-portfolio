"""
Experiment utilities — multi-seed QAOA runs and risk-aversion sweeps.

This module is the "second pass" over the QAOA pipeline. Instead of running a
single QAOA trajectory and reporting one number, the functions here run many
trajectories (different random initialisations, different problem parameters)
and aggregate the results so we can make defensible *statistical* claims.

Two routines:

1. ``run_multiseed_qaoa`` — repeat the same QAOA(p) optimisation N times
   with random starting parameters and report distribution statistics
   (approximation ratio mean ± std, ground-state probability range, etc).

2. ``run_risk_aversion_sweep`` — for a list of risk-aversion values q,
   rebuild the QUBO, find the classical optimum, run QAOA, and report
   how the selected portfolio shifts as q varies.

Both lean on the existing ``src.qaoa`` and ``src.encoding`` modules; nothing
new is invented here, this is just orchestration plus aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from src.classical import solve_binary_brute_force
from src.encoding import (
    build_qubo, qubo_to_ising, hamiltonian_eigendecomposition,
)
from src.qaoa import (
    optimise_qaoa, sample, bitstring_energy, ground_state_probability,
)


# ---------------------------------------------------------------------------
# Multi-seed QAOA
# ---------------------------------------------------------------------------
@dataclass
class MultiseedResult:
    """Aggregated results from N QAOA runs at fixed depth ``p``."""

    p: int
    n_seeds: int
    ratios: np.ndarray              # shape (N,) — approximation ratio per seed
    energies: np.ndarray            # shape (N,) — final ⟨H⟩ per seed
    ground_probs: np.ndarray        # shape (N,) — P(ground state) in sampling
    best_sampled: np.ndarray        # shape (N,) — lowest energy sampled
    iterations: np.ndarray          # shape (N,) — COBYLA function evaluations

    def summary(self) -> dict[str, float]:
        return {
            "p": self.p,
            "n_seeds": self.n_seeds,
            "ratio_mean": float(np.mean(self.ratios)),
            "ratio_std":  float(np.std(self.ratios, ddof=1)) if self.n_seeds > 1 else 0.0,
            "ratio_min":  float(np.min(self.ratios)),
            "ratio_max":  float(np.max(self.ratios)),
            "energy_mean": float(np.mean(self.energies)),
            "energy_min":  float(np.min(self.energies)),
            "gs_prob_mean": float(np.mean(self.ground_probs)),
            "gs_prob_max":  float(np.max(self.ground_probs)),
            "best_sampled_min": float(np.min(self.best_sampled)),
        }


def run_multiseed_qaoa(
    hamiltonian: SparsePauliOp,
    p: int,
    *,
    n_seeds: int = 10,
    ground_energy: float,
    max_energy: float,
    ground_bitstring: str,
    sorted_energies: np.ndarray,
    sorted_bitstrings: np.ndarray,
    shots: int = 4096,
    seed_offset: int = 0,
    maxiter: int = 200,
) -> MultiseedResult:
    """Run ``n_seeds`` QAOA optimisations at depth ``p`` from random starts.

    Initial parameters are drawn uniformly: γ ∈ [0, π], β ∈ [0, π/2].
    Each run uses a different random seed both for the initial parameters
    *and* for the post-optimisation sampling — that way different seeds give
    statistically independent observations of "what QAOA at depth p achieves
    when initialised randomly".

    Returns
    -------
    MultiseedResult
        Distribution statistics across the N runs.
    """
    rng = np.random.default_rng(seed_offset)

    ratios       = np.empty(n_seeds)
    energies     = np.empty(n_seeds)
    ground_probs = np.empty(n_seeds)
    best_sampled = np.empty(n_seeds)
    iterations   = np.empty(n_seeds, dtype=int)

    for k in range(n_seeds):
        # Random initialisation in the natural QAOA parameter ranges.
        gammas = rng.uniform(0.0, np.pi,       size=p)
        betas  = rng.uniform(0.0, np.pi / 2.0, size=p)
        initial = np.concatenate([gammas, betas])

        result = optimise_qaoa(
            hamiltonian, p,
            initial_params=initial,
            maxiter=maxiter,
        )
        result = sample(result, shots=shots, seed=seed_offset * 1000 + k)

        ratios[k]       = (max_energy - result.optimal_energy) / (max_energy - ground_energy)
        energies[k]     = result.optimal_energy
        ground_probs[k] = ground_state_probability(result.counts, ground_bitstring, shots=shots)
        iterations[k]   = result.n_iterations

        sampled_es = [
            bitstring_energy(b, sorted_energies, sorted_bitstrings)
            for b in result.counts
        ]
        best_sampled[k] = min(sampled_es)

    return MultiseedResult(
        p=p, n_seeds=n_seeds,
        ratios=ratios, energies=energies,
        ground_probs=ground_probs, best_sampled=best_sampled,
        iterations=iterations,
    )


# ---------------------------------------------------------------------------
# Risk-aversion sweep
# ---------------------------------------------------------------------------
@dataclass
class RiskSweepRow:
    """One row of the risk-aversion sweep."""

    q: float
    classical_selection: np.ndarray
    classical_objective: float
    classical_return: float
    classical_risk: float
    qaoa_energy: float
    qaoa_ratio: float
    qaoa_selection: np.ndarray
    seeds_used: int


@dataclass
class RiskSweepResult:
    """Stacked rows for plotting and analysis."""

    rows: list[RiskSweepRow] = field(default_factory=list)

    @property
    def qs(self) -> np.ndarray:
        return np.array([r.q for r in self.rows])

    @property
    def classical_selections(self) -> np.ndarray:
        return np.array([r.classical_selection for r in self.rows], dtype=int)

    @property
    def qaoa_selections(self) -> np.ndarray:
        return np.array([r.qaoa_selection for r in self.rows], dtype=int)

    @property
    def qaoa_ratios(self) -> np.ndarray:
        return np.array([r.qaoa_ratio for r in self.rows])

    @property
    def classical_returns(self) -> np.ndarray:
        return np.array([r.classical_return for r in self.rows])

    @property
    def classical_risks(self) -> np.ndarray:
        return np.array([r.classical_risk for r in self.rows])


def run_risk_aversion_sweep(
    mu: np.ndarray,
    sigma: np.ndarray,
    qs: Sequence[float],
    *,
    budget: int,
    p: int = 1,
    n_seeds: int = 3,
    shots: int = 4096,
    maxiter: int = 200,
) -> RiskSweepResult:
    """Sweep risk-aversion q; rebuild QUBO and re-run QAOA at each value.

    For each q:
      1. Build the QUBO with that q. Convert to Ising.
      2. Find the classical optimum by brute force (8 qubits → trivial).
      3. Run QAOA n_seeds times from random initialisations, keep the best.
      4. Read off the most-frequently-sampled bitstring as the QAOA pick.

    Parameters
    ----------
    qs : sequence of float
        Risk-aversion values to sweep, e.g. ``[0.0, 0.25, 0.5, 1.0, 2.0]``.
    p : int
        QAOA depth. Use 1 unless you have time to spare — the sweep is
        already n_qs × n_seeds optimisations.
    """
    out = RiskSweepResult()

    for q in qs:
        # Classical reference.
        brute = solve_binary_brute_force(
            mu, sigma, budget=budget, risk_factor=q,
        )
        classical_return = float(mu @ brute.selection)
        classical_risk   = float(np.sqrt(brute.selection @ sigma @ brute.selection))

        # Build Ising for this q.
        qubo = build_qubo(mu, sigma, budget=budget, risk_factor=q)
        H, _ = qubo_to_ising(qubo)
        energies_sorted, bitstrings_sorted = hamiltonian_eigendecomposition(H)
        E_min, E_max = energies_sorted[0], energies_sorted[-1]
        ground_bitstring = "".join(str(b) for b in reversed(bitstrings_sorted[0]))

        # Multi-seed QAOA — keep the best.
        ms = run_multiseed_qaoa(
            H, p,
            n_seeds=n_seeds,
            ground_energy=E_min,
            max_energy=E_max,
            ground_bitstring=ground_bitstring,
            sorted_energies=energies_sorted,
            sorted_bitstrings=bitstrings_sorted,
            shots=shots,
            seed_offset=int(q * 1e4),
            maxiter=maxiter,
        )
        best_seed = int(np.argmax(ms.ratios))

        # Re-run the best seed and grab its sampling distribution to identify
        # QAOA's most-likely bitstring (mode of the empirical distribution).
        rng = np.random.default_rng(int(q * 1e4))
        # Re-roll to the chosen seed so we get the same initial params.
        for _ in range(best_seed):
            rng.uniform(0.0, np.pi, size=p)
            rng.uniform(0.0, np.pi / 2.0, size=p)
        gammas_b = rng.uniform(0.0, np.pi, size=p)
        betas_b  = rng.uniform(0.0, np.pi / 2.0, size=p)
        initial = np.concatenate([gammas_b, betas_b])
        res = optimise_qaoa(H, p, initial_params=initial, maxiter=maxiter)
        res = sample(res, shots=shots, seed=int(q * 1e4) * 1000 + best_seed)

        mode_bits, _ = max(res.counts.items(), key=lambda kv: kv[1])
        qaoa_selection = np.array(
            [int(c) for c in reversed(mode_bits)], dtype=int
        )

        out.rows.append(
            RiskSweepRow(
                q=float(q),
                classical_selection=brute.selection.astype(int),
                classical_objective=float(brute.objective),
                classical_return=classical_return,
                classical_risk=classical_risk,
                qaoa_energy=float(ms.energies[best_seed]),
                qaoa_ratio=float(ms.ratios[best_seed]),
                qaoa_selection=qaoa_selection,
                seeds_used=n_seeds,
            )
        )

    return out

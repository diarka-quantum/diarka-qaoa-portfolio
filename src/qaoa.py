"""
QAOA circuit construction, optimisation, and analysis for the Diarka portfolio project.

Three concerns are kept separate:

1. **Circuit construction.** Build the parameterised QAOA ansatz for a given
   Ising Hamiltonian and depth ``p``. Cost layers are produced by Qiskit's
   ``PauliEvolutionGate``; mixer layers are explicit ``RX`` rotations.
2. **Optimisation.** A thin wrapper around ``scipy.optimize.minimize`` that
   records the full trace of expectation values across iterations so we can
   plot convergence behaviour.
3. **Analysis.** Sampling the optimised circuit, computing the approximation
   ratio, locating the ground-state probability in the sampled distribution,
   and translating bitstrings back to portfolio selections.

The Estimator and Sampler are exact statevector simulators by default — for
n = 8 qubits this is essentially free and removes shot noise from the QAOA
optimisation curves, which makes the per-depth comparison much cleaner. Aer's
shot-based primitives are easy to swap in once we move to noise simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class QAOAResult:
    """Outcome of one QAOA run at a fixed depth ``p``."""

    p: int
    optimal_params: np.ndarray              # shape (2p,) — gammas then betas
    optimal_energy: float                   # final ⟨H⟩
    trace: np.ndarray                       # ⟨H⟩ per iteration
    n_iterations: int
    converged: bool
    ansatz: QuantumCircuit
    counts: dict[str, int] = field(default_factory=dict)   # populated by ``sample``
    shots: int = 0


# ---------------------------------------------------------------------------
# Circuit construction
# ---------------------------------------------------------------------------
def qaoa_ansatz(hamiltonian: SparsePauliOp, p: int) -> QuantumCircuit:
    """Build a parameterised QAOA circuit at depth ``p``.

    The standard QAOA prescription:

        |ψ(γ, β)⟩  =  ∏ₖ U_M(βₖ) · U_C(γₖ)  ·  H⊗ⁿ |0⟩⊗ⁿ

    where

        U_C(γ) = e^{−i γ H_cost}      (problem Hamiltonian)
        U_M(β) = e^{−i β ∑ Xᵢ}        (transverse-field mixer)

    The parameter vector has length ``2p`` arranged as
    ``[γ₀, γ₁, …, γ_{p-1}, β₀, β₁, …, β_{p-1}]``.

    Parameters
    ----------
    hamiltonian : SparsePauliOp
        Ising cost Hamiltonian on ``n`` qubits.
    p : int
        Depth (number of cost+mixer layer pairs).

    Returns
    -------
    QuantumCircuit
        With ``2p`` symbolic parameters bound to a single ``ParameterVector``
        named ``θ``.
    """
    if p < 1:
        raise ValueError("p must be >= 1.")

    n = hamiltonian.num_qubits
    theta = ParameterVector("θ", 2 * p)

    qc = QuantumCircuit(n, name=f"QAOA(p={p})")

    # Initial state: equal superposition |+⟩⊗ⁿ.
    qc.h(range(n))
    qc.barrier()

    for layer in range(p):
        gamma = theta[layer]
        beta  = theta[p + layer]

        # Cost layer: U_C(γ) = exp(−i γ H).
        qc.append(
            PauliEvolutionGate(hamiltonian, time=gamma),
            range(n),
        )
        qc.barrier()

        # Mixer layer: U_M(β) = ∏ᵢ RX(2β)ᵢ  (since exp(−iβX) = RX(2β)).
        for q in range(n):
            qc.rx(2 * beta, q)
        qc.barrier()

    return qc


def default_initial_parameters(p: int) -> np.ndarray:
    """Linear-ramp initialisation, a well-known good starting point.

    γ ramps up from a small value, β ramps down to a small value.
    This empirically outperforms random or all-zero starts on most
    Ising-style cost landscapes.
    """
    gammas = np.linspace(0.1, 0.5, p)
    betas  = np.linspace(0.5, 0.1, p)
    return np.concatenate([gammas, betas])


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------
def optimise_qaoa(
    hamiltonian: SparsePauliOp,
    p: int,
    *,
    initial_params: np.ndarray | None = None,
    method: str = "COBYLA",
    maxiter: int = 200,
    rhobeg: float = 0.3,
    tol: float = 1e-4,
    estimator=None,
    basis_gates: Sequence[str] | None = None,
    optimization_level: int = 1,
) -> QAOAResult:
    """Minimise ⟨H⟩ over the QAOA parameters at depth ``p``.

    Uses ``scipy.optimize.minimize`` and records every expectation evaluated
    by the classical optimiser in ``QAOAResult.trace``. ``StatevectorEstimator``
    gives exact (shot-noise-free) energies, which makes convergence curves
    monotone (modulo COBYLA's trust-region behaviour) and easy to compare.

    Parameters
    ----------
    hamiltonian : SparsePauliOp
        The Ising cost Hamiltonian.
    p : int
        QAOA depth.
    initial_params : np.ndarray, optional
        Starting point. Defaults to ``default_initial_parameters(p)``.
    method : str
        scipy optimiser name. COBYLA is robust and gradient-free which suits
        QAOA's noisy-looking landscapes even at the simulator level.
    maxiter, rhobeg, tol : COBYLA hyperparameters.
    estimator : BaseEstimatorV2, optional
        The primitive used to evaluate ⟨H⟩. Defaults to ``StatevectorEstimator``
        (exact, noiseless). Pass an Aer noisy ``EstimatorV2`` here to optimise
        under a noise model — this is the Week 4 entry point. The interface
        contract is the V2 PUB form ``run([(circuit, observable, params)])``
        with the energy read from ``result()[0].data.evs``; both the statevector
        and Aer primitives satisfy it.
    basis_gates : Sequence[str], optional
        If given, the ansatz is transpiled to these native gates ONCE before
        the optimisation loop. Required when using an Aer noisy estimator,
        because Aer cannot assemble the raw ``PauliEvolutionGate`` — and because
        transpiling to e.g. ``{sx, rz, cz}`` is what makes the two-qubit gate
        count (and therefore the gate-error contribution) physically faithful.
        Leave as ``None`` for the statevector path (it decomposes internally).
    optimization_level : int
        Transpiler optimisation level used only when ``basis_gates`` is set.
    """
    if initial_params is None:
        initial_params = default_initial_parameters(p)
    if len(initial_params) != 2 * p:
        raise ValueError(
            f"initial_params length {len(initial_params)} ≠ 2p = {2 * p}."
        )

    if estimator is None:
        estimator = StatevectorEstimator()

    ansatz = qaoa_ansatz(hamiltonian, p)

    # Transpile to native gates if requested (noisy-backend path). The
    # parameterised circuit keeps its free parameters through transpilation,
    # so the optimisation loop binds them exactly as before.
    run_ansatz = ansatz
    if basis_gates is not None:
        run_ansatz = transpile(
            ansatz,
            basis_gates=list(basis_gates),
            optimization_level=optimization_level,
        )

    trace: list[float] = []

    def cost(params: np.ndarray) -> float:
        job = estimator.run([(run_ansatz, hamiltonian, params)])
        energy = float(job.result()[0].data.evs)
        trace.append(energy)
        return energy

    options = {"maxiter": maxiter, "rhobeg": rhobeg, "disp": False}
    result = minimize(
        cost, x0=initial_params, method=method, tol=tol, options=options,
    )

    return QAOAResult(
        p=p,
        optimal_params=np.asarray(result.x),
        optimal_energy=float(result.fun),
        trace=np.array(trace),
        n_iterations=int(getattr(result, "nfev", len(trace))),
        converged=bool(result.success),
        ansatz=run_ansatz,
    )


# ---------------------------------------------------------------------------
# Sampling and analysis
# ---------------------------------------------------------------------------
def sample(
    result: QAOAResult,
    *,
    shots: int = 4096,
    seed: int | None = None,
    sampler=None,
) -> QAOAResult:
    """Sample bitstrings from the optimised circuit.

    Returns the same ``QAOAResult`` with ``counts`` and ``shots`` populated.

    Parameters
    ----------
    sampler : BaseSamplerV2, optional
        The primitive used to draw shots. Defaults to ``StatevectorSampler``
        (exact, noiseless). Pass an Aer noisy ``SamplerV2`` to draw shots under
        a noise model — the Week 4 entry point. The contract is the V2 PUB form
        ``run([(measured_circuit, params)], shots=shots)`` with counts read from
        ``result()[0].data.meas.get_counts()``. Note: when sampling through a
        noisy Aer sampler, ``result.ansatz`` must already be in native gates
        (it is, if ``optimise_qaoa`` was called with ``basis_gates``).
    """
    if sampler is None:
        sampler = StatevectorSampler(seed=seed) if seed is not None else StatevectorSampler()

    measured = result.ansatz.copy()
    measured.measure_all()

    job = sampler.run([(measured, result.optimal_params)], shots=shots)
    raw = job.result()[0].data.meas.get_counts()
    result.counts = dict(raw)
    result.shots = shots
    return result


def approximation_ratio(
    measured_energy: float,
    *,
    ground_energy: float,
    max_energy: float,
) -> float:
    """Approximation ratio in the range [0, 1].

        r = (E_max − E) / (E_max − E_min)

    Interpretation:

        r = 1   QAOA's ⟨H⟩ equals the ground-state energy.
        r = 0   QAOA's ⟨H⟩ equals the worst (highest) eigenvalue.
        r ∈ [0, 1] for any well-defined QAOA run on a problem with a
        non-trivial spectrum.
    """
    denom = max_energy - ground_energy
    if denom <= 0:
        raise ValueError("Spectrum has zero width — approximation ratio is undefined.")
    return (max_energy - measured_energy) / denom


def bitstring_energy(
    bitstring: str,
    sorted_energies: np.ndarray,
    sorted_bitstrings: np.ndarray,
) -> float:
    """Look up the Ising energy of a bitstring against the saved spectrum."""
    n = sorted_bitstrings.shape[1]
    if len(bitstring) != n:
        raise ValueError(f"bitstring length {len(bitstring)} ≠ n = {n}.")
    # Convert Qiskit bitstring (right-to-left = q0…q_{n-1}) to little-endian array.
    bits = np.array([int(c) for c in reversed(bitstring)], dtype=int)
    matches = np.where(np.all(sorted_bitstrings == bits, axis=1))[0]
    if len(matches) == 0:
        raise ValueError(f"Bitstring {bitstring!r} not found in spectrum.")
    return float(sorted_energies[matches[0]])


def ground_state_probability(
    counts: dict[str, int],
    ground_bitstring: str,
    shots: int | None = None,
) -> float:
    """Probability of measuring the ground-state bitstring in the sample."""
    total = shots if shots is not None else sum(counts.values())
    return counts.get(ground_bitstring, 0) / total


def top_k_bitstrings(
    counts: dict[str, int],
    k: int = 10,
) -> list[tuple[str, int]]:
    """Return the k most-frequently-sampled bitstrings as (bits, count)."""
    return sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:k]

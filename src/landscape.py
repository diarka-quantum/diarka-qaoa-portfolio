"""
QAOA parameter-landscape utilities — direct numpy statevector implementation.

For an 8-qubit Ising Hamiltonian a statevector is just 256 complex numbers,
so we can sweep ⟨H⟩ over a fine (γ, β) grid in well under a second by
working directly with arrays. This is much faster than going via Qiskit's
Estimator primitive on a per-point basis, which carries non-trivial Python
overhead.

The mathematical operations:

1. The initial state |+⟩⊗ⁿ is a uniform superposition: psi[k] = 1/√(2ⁿ) ∀ k.
2. The cost evolution e^{-iγH} on a diagonal Ising H is an element-wise
   multiply by exp(-iγ · H_diag).
3. The mixer e^{-iβ ∑ Xᵢ} factorises as a tensor product of single-qubit
   gates ⊗ e^{-iβXᵢ}. We apply it qubit-by-qubit using a reshape trick that
   costs O(n · 2ⁿ) rather than the O(2²ⁿ) of a naïve matrix product.
4. ⟨H⟩ for diagonal H is ∑ |psi[k]|² · H_diag[k] — another O(2ⁿ) sum.

The functions here are public-API compatible with the QAOA module above:
they take a `SparsePauliOp` cost Hamiltonian and return numpy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import SparsePauliOp


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LandscapeP1:
    """Energies on a regular (γ, β) grid for QAOA at p = 1."""

    gammas: np.ndarray            # shape (n_gamma,)
    betas: np.ndarray             # shape (n_beta,)
    energies: np.ndarray          # shape (n_gamma, n_beta)

    @property
    def shape(self) -> tuple[int, int]:
        return self.energies.shape

    def argmin(self) -> tuple[int, int, float, float, float]:
        idx = np.unravel_index(np.argmin(self.energies), self.energies.shape)
        i, j = int(idx[0]), int(idx[1])
        return i, j, float(self.gammas[i]), float(self.betas[j]), float(self.energies[i, j])


# ---------------------------------------------------------------------------
# Mixer application
# ---------------------------------------------------------------------------
def _apply_x_rotation(psi: np.ndarray, beta: float, qubit: int, n: int) -> np.ndarray:
    """Apply e^{-iβX} to a single qubit of a 2ⁿ-dim statevector.

    Uses a reshape trick: view psi as a length-n tensor where one axis is the
    target qubit, apply the 2×2 gate along that axis, then flatten back.

    e^{-iβX} = cos(β) · I  −  i sin(β) · X
            = [[cos β, −i sin β], [−i sin β, cos β]]
    """
    cb, sb = np.cos(beta), np.sin(beta)
    gate = np.array([[cb, -1j * sb], [-1j * sb, cb]], dtype=complex)

    # Reshape to expose the target qubit's axis.
    shape = [2] * n
    psi_t = psi.reshape(shape)

    # Qiskit-style indexing: qubit 0 is the LEAST-significant bit (rightmost
    # in the standard binary representation). With numpy's row-major reshape
    # the most-significant bit is axis 0. So qubit i lives on axis (n - 1 - i).
    axis = n - 1 - qubit
    psi_t = np.tensordot(gate, psi_t, axes=([1], [axis]))
    # tensordot puts the contracted axis first; move it back to its original
    # position so we can apply more single-qubit gates without confusion.
    psi_t = np.moveaxis(psi_t, 0, axis)

    return psi_t.reshape(-1)


def apply_mixer(psi: np.ndarray, beta: float, n: int) -> np.ndarray:
    """Apply U_M(β) = ∏ᵢ e^{-iβXᵢ} to a statevector."""
    for q in range(n):
        psi = _apply_x_rotation(psi, beta, q, n)
    return psi


# ---------------------------------------------------------------------------
# Grid sweep
# ---------------------------------------------------------------------------
def sweep_p1(
    hamiltonian: SparsePauliOp,
    *,
    gamma_range: tuple[float, float] = (0.0, np.pi),
    beta_range:  tuple[float, float] = (0.0, np.pi / 2),
    n_gamma: int = 50,
    n_beta:  int = 50,
) -> LandscapeP1:
    """Sweep ⟨H⟩(γ, β) on a regular grid for QAOA at depth 1.

    Default ranges:
        γ ∈ [0, π]      — one half of e^{-iγH}'s natural period; the landscape
                          is symmetric in γ → −γ up to mixer reflection.
        β ∈ [0, π/2]    — natural half-period of RX(2β).

    Returns
    -------
    LandscapeP1
    """
    n = hamiltonian.num_qubits

    # Diagonal of H. For an Ising Hamiltonian (only I and Z Paulis) the matrix
    # is diagonal in the computational basis, so we read it off directly.
    H_matrix = hamiltonian.to_matrix(sparse=False)
    H_diag = np.real(np.diag(H_matrix)).astype(np.float64)

    # |+⟩⊗ⁿ
    psi_init = np.ones(2 ** n, dtype=complex) / np.sqrt(2 ** n)

    gammas = np.linspace(*gamma_range, n_gamma)
    betas  = np.linspace(*beta_range,  n_beta)

    energies = np.empty((n_gamma, n_beta), dtype=float)

    for i, g in enumerate(gammas):
        # Cost evolution: e^{-iγH} is element-wise multiply for diagonal H.
        psi_after_cost = np.exp(-1j * g * H_diag) * psi_init

        for j, b in enumerate(betas):
            psi = apply_mixer(psi_after_cost.copy(), b, n)
            # ⟨H⟩ = ∑ |ψ[k]|² H_diag[k]   for diagonal H.
            energies[i, j] = float(np.sum(np.abs(psi) ** 2 * H_diag))

    return LandscapeP1(gammas=gammas, betas=betas, energies=energies)


# ---------------------------------------------------------------------------
# 1-D slices through a chosen point
# ---------------------------------------------------------------------------
def slice_along_gamma(
    hamiltonian: SparsePauliOp,
    *,
    beta_fixed: float,
    gamma_range: tuple[float, float] = (0.0, np.pi),
    n_points: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """⟨H⟩ at p=1 along the line β = β_fixed, γ varying."""
    n = hamiltonian.num_qubits
    H_diag = np.real(np.diag(hamiltonian.to_matrix(sparse=False))).astype(np.float64)
    psi_init = np.ones(2 ** n, dtype=complex) / np.sqrt(2 ** n)

    gammas = np.linspace(*gamma_range, n_points)
    energies = np.empty(n_points)

    for i, g in enumerate(gammas):
        psi = np.exp(-1j * g * H_diag) * psi_init
        psi = apply_mixer(psi, beta_fixed, n)
        energies[i] = float(np.sum(np.abs(psi) ** 2 * H_diag))

    return gammas, energies


def slice_along_beta(
    hamiltonian: SparsePauliOp,
    *,
    gamma_fixed: float,
    beta_range: tuple[float, float] = (0.0, np.pi / 2),
    n_points: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """⟨H⟩ at p=1 along the line γ = γ_fixed, β varying."""
    n = hamiltonian.num_qubits
    H_diag = np.real(np.diag(hamiltonian.to_matrix(sparse=False))).astype(np.float64)
    psi_init = np.ones(2 ** n, dtype=complex) / np.sqrt(2 ** n)

    betas = np.linspace(*beta_range, n_points)
    energies = np.empty(n_points)

    # Cost layer is independent of β — compute once.
    psi_after_cost = np.exp(-1j * gamma_fixed * H_diag) * psi_init

    for j, b in enumerate(betas):
        psi = apply_mixer(psi_after_cost.copy(), b, n)
        energies[j] = float(np.sum(np.abs(psi) ** 2 * H_diag))

    return betas, energies

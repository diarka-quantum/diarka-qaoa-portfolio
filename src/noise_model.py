"""
Noise modelling for the QAOA portfolio optimiser — Week 4.

The goal of this module is to close the loop on the Week 2/3 result:

    noiseless simulator   ->   approximation ratio 0.942
    real IBM Heron r2      ->   approximation ratio 0.910

A 3.4 percentage-point gap is small, but "small" is not an explanation.
This module lets us *reproduce* that gap in a controlled simulator so we can
attribute it to specific physical effects (two-qubit gate error, single-qubit
gate error, decoherence over the circuit duration, and readout/measurement
error) rather than hand-waving about "noise".

Design choices
--------------
* We build the noise model in two ways:
    1. ``noise_model_from_backend`` — pull the *actual* calibration data from
       a real IBM backend (most faithful; requires IBM Quantum access).
    2. ``synthetic_heron_noise_model`` — a hand-built model using published
       Heron r2 figures, so the notebook runs end-to-end with zero credentials
       and zero queue time. This is the default for development.

* We separate the *channels* so the notebook can switch them on and off one at
  a time. That on/off decomposition is the whole point of Week 4: it turns
  "noise" into a stacked bar chart of named contributions.

Nothing here invents new QAOA logic — the circuits and cost evaluation come
from your existing ``src.qaoa`` / ``src.encoding`` modules. This file only
builds NoiseModel objects and a thin ``run_noisy`` wrapper around AerSimulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    thermal_relaxation_error,
    ReadoutError,
)


# ---------------------------------------------------------------------------
# Published / representative IBM Heron r2 figures (mid-2025 generation).
# These are *order-of-magnitude correct* defaults for a synthetic model.
# Treat them as a starting point and override with real backend data when you
# want the faithful comparison. Units: seconds for times, dimensionless errors.
# ---------------------------------------------------------------------------
@dataclass
class HeronParams:
    # Single-qubit gate (sx / x / rz are virtual-Z so effectively free)
    one_qubit_gate_error: float = 2.5e-4      # ~0.025% per physical 1q gate
    one_qubit_gate_time: float = 35e-9        # 35 ns

    # Two-qubit gate (CZ on Heron). This is the dominant error source.
    two_qubit_gate_error: float = 3.0e-3      # ~0.3% per CZ
    two_qubit_gate_time: float = 70e-9        # 70 ns

    # Coherence times (median Heron r2 figures are ~150-300us)
    t1: float = 200e-6                        # 200 us
    t2: float = 150e-6                        # 150 us

    # Readout / measurement assignment error (per qubit)
    readout_error: float = 1.2e-2             # ~1.2% symmetric

    # Whether to make readout error asymmetric (P(1|0) != P(0|1)).
    # Real devices are asymmetric — relaxation biases toward reading 0.
    readout_asymmetry: float = 0.6            # P(0|1) = error*(1+a), P(1|0)=error*(1-a)


@dataclass
class NoiseChannels:
    """Switch individual noise sources on/off for attribution studies."""
    one_qubit_depolarizing: bool = True
    two_qubit_depolarizing: bool = True
    thermal_relaxation: bool = True
    readout: bool = True

    def label(self) -> str:
        on = []
        if self.one_qubit_depolarizing:
            on.append("1q-depol")
        if self.two_qubit_depolarizing:
            on.append("2q-depol")
        if self.thermal_relaxation:
            on.append("T1/T2")
        if self.readout:
            on.append("readout")
        return "+".join(on) if on else "ideal"


def synthetic_heron_noise_model(
    params: Optional[HeronParams] = None,
    channels: Optional[NoiseChannels] = None,
    n_qubits: int = 8,
) -> NoiseModel:
    """Build a hand-tuned NoiseModel approximating IBM Heron r2.

    Parameters
    ----------
    params : HeronParams
        Physical device parameters. Defaults to representative Heron r2 values.
    channels : NoiseChannels
        Which noise sources to include. Default: all on.
    n_qubits : int
        Number of qubits to attach readout error to.

    Returns
    -------
    NoiseModel
        Ready to pass to ``AerSimulator(noise_model=...)``.

    Notes
    -----
    Two-qubit gate error is modelled as depolarizing error *composed with*
    thermal relaxation on both qubits, applied to the basis two-qubit gate.
    We register it under both ``cz`` and ``ecr`` so the model is robust to
    whichever entangling gate the transpiler emits for your target.
    """
    p = params or HeronParams()
    ch = channels or NoiseChannels()
    nm = NoiseModel()

    # --- Single-qubit gate error: depolarizing (+ optional relaxation) ---
    one_q_err = None
    if ch.one_qubit_depolarizing:
        one_q_err = depolarizing_error(p.one_qubit_gate_error, 1)
    if ch.thermal_relaxation:
        relax_1q = thermal_relaxation_error(
            p.t1, p.t2, p.one_qubit_gate_time
        )
        one_q_err = relax_1q if one_q_err is None else one_q_err.compose(relax_1q)
    if one_q_err is not None:
        nm.add_all_qubit_quantum_error(one_q_err, ["sx", "x", "id"])

    # --- Two-qubit gate error: the dominant contribution ---
    two_q_err = None
    if ch.two_qubit_depolarizing:
        two_q_err = depolarizing_error(p.two_qubit_gate_error, 2)
    if ch.thermal_relaxation:
        relax_2q = thermal_relaxation_error(
            p.t1, p.t2, p.two_qubit_gate_time
        ).expand(
            thermal_relaxation_error(p.t1, p.t2, p.two_qubit_gate_time)
        )
        two_q_err = relax_2q if two_q_err is None else two_q_err.compose(relax_2q)
    if two_q_err is not None:
        nm.add_all_qubit_quantum_error(two_q_err, ["cz", "ecr", "cx"])

    # --- Readout / measurement assignment error ---
    if ch.readout:
        e = p.readout_error
        a = p.readout_asymmetry
        p0_given_1 = min(e * (1 + a), 0.5)   # relaxation: 1 decays to 0 more often
        p1_given_0 = max(e * (1 - a), 0.0)
        ro = ReadoutError(
            [
                [1 - p1_given_0, p1_given_0],   # true |0>: P(0|0), P(1|0)
                [p0_given_1, 1 - p0_given_1],   # true |1>: P(0|1), P(1|1)
            ]
        )
        for q in range(n_qubits):
            nm.add_readout_error(ro, [q])

    return nm


def noise_model_from_backend(backend) -> NoiseModel:
    """Build a faithful NoiseModel from a real IBM backend's calibration.

    Use this when you want the *actual* device behaviour rather than the
    synthetic approximation. Requires a backend object from
    ``QiskitRuntimeService().backend("ibm_...")``.
    """
    return NoiseModel.from_backend(backend)


def make_noisy_simulator(
    noise_model: NoiseModel,
    seed: Optional[int] = None,
) -> AerSimulator:
    """Return an AerSimulator configured with the given noise model."""
    return AerSimulator(noise_model=noise_model, seed_simulator=seed)


# Convenience: the standard channel-isolation suite for the attribution study.
def attribution_suite() -> dict[str, NoiseChannels]:
    """Return the set of channel configurations for the stacked decomposition.

    Each entry adds one channel on top of the previous, so plotting the
    approximation-ratio drop at each step shows how much each physical effect
    contributes to the total noiseless->hardware gap.
    """
    return {
        "ideal": NoiseChannels(False, False, False, False),
        "+1q gate": NoiseChannels(True, False, False, False),
        "+2q gate": NoiseChannels(True, True, False, False),
        "+T1/T2": NoiseChannels(True, True, True, False),
        "+readout (full)": NoiseChannels(True, True, True, True),
    }

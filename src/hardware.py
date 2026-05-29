"""
IBM Quantum hardware utilities.

This module wraps the parts of ``qiskit-ibm-runtime`` we need for a single
QAOA submission, in shapes that match the rest of the project. It assumes
credentials are already saved on this machine (e.g. via a prior
``QiskitRuntimeService.save_account(...)`` call) — we deliberately don't
deal with token entry here.

Workflow:

    service  = get_service()
    backend  = pick_backend(service, min_qubits=8)
    transp   = transpile_for_backend(ansatz, params, backend)
    job      = submit_job(transp, backend, shots=4096)
    job_id   = job.job_id()
    # ... wait (minutes to days depending on queue) ...
    result   = fetch_result(job_id, service=service)
    counts   = extract_counts(result)

The notebook does not need to be open continuously between submission and
result fetching — once the job is in IBM's queue, the local Python state
is no longer relevant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2


# ---------------------------------------------------------------------------
# Service and backend
# ---------------------------------------------------------------------------
def get_service() -> QiskitRuntimeService:
    """Open a connection to IBM Quantum using the locally-saved account.

    Raises a friendly error if credentials aren't found, with a one-line
    recipe for fixing it.
    """
    try:
        return QiskitRuntimeService()
    except Exception as exc:
        raise RuntimeError(
            "Could not connect to IBM Quantum. Most common cause: the API "
            "token isn't saved on this machine. Fix with:\n\n"
            "    from qiskit_ibm_runtime import QiskitRuntimeService\n"
            "    QiskitRuntimeService.save_account(\n"
            "        channel='ibm_quantum_platform',\n"
            "        token='YOUR_TOKEN_FROM_IBM_QUANTUM_PORTAL',\n"
            "        overwrite=True,\n"
            "    )\n\n"
            f"Original error: {exc}"
        ) from exc


def pick_backend(
    service: QiskitRuntimeService,
    *,
    min_qubits: int = 8,
    name: str | None = None,
):
    """Choose an IBM Quantum backend.

    By default returns the least-busy real (non-simulator) backend with at
    least ``min_qubits`` qubits. If ``name`` is given, returns that specific
    backend instead — useful for reproducing a previous run.
    """
    if name is not None:
        return service.backend(name)
    return service.least_busy(
        operational=True,
        simulator=False,
        min_num_qubits=min_qubits,
    )


def describe_backend(backend) -> str:
    """Compact summary of a backend for printing in notebook cells."""
    config = backend.configuration() if hasattr(backend, "configuration") else None
    status = backend.status() if hasattr(backend, "status") else None
    lines = [f"Backend:        {backend.name}"]
    if config is not None:
        lines.append(f"Qubits:         {config.n_qubits}")
        lines.append(f"Basis gates:    {', '.join(config.basis_gates)}")
    if status is not None:
        lines.append(f"Pending jobs:   {status.pending_jobs}")
        lines.append(f"Operational:    {status.operational}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Transpilation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TranspilationReport:
    """Stats about a transpiled circuit, for sanity-checking before submission."""

    depth: int
    n_2q_gates: int                  # CX + ECR + similar
    n_1q_gates: int
    n_qubits_used: int
    gate_counts: dict[str, int]
    cx_density: float                # CX per qubit
    backend_name: str


def transpile_for_backend(
    ansatz: QuantumCircuit,
    parameters: Iterable[float],
    backend,
    *,
    optimization_level: int = 3,
    seed_transpiler: int = 12345,
) -> tuple[QuantumCircuit, TranspilationReport]:
    """Bind QAOA parameters and transpile for a specific backend.

    Returns the transpiled circuit AND a small report so the notebook can
    print depth/CX counts before submitting.
    """
    # Bind the parameters first so the cost layer collapses to concrete
    # rotation angles before transpilation. Then the synthesis passes
    # produce the most compact possible decomposition.
    bound = ansatz.assign_parameters(np.asarray(parameters))

    # Add measurements last, after binding, before transpilation — the
    # pass manager will route them along with the gates.
    bound.measure_all()

    pm = generate_preset_pass_manager(
        optimization_level=optimization_level,
        backend=backend,
        seed_transpiler=seed_transpiler,
    )
    transpiled = pm.run(bound)

    gate_counts = dict(transpiled.count_ops())
    n_2q = sum(
        gate_counts.get(g, 0)
        for g in ("cx", "cz", "ecr", "rzx", "rxx", "ryy", "rzz")
    )
    n_1q = sum(
        gate_counts.get(g, 0)
        for g in ("rz", "sx", "x", "y", "z", "h", "rx", "ry", "u", "u1", "u2", "u3")
    )

    n_qubits_used = transpiled.num_qubits
    report = TranspilationReport(
        depth=transpiled.depth(),
        n_2q_gates=n_2q,
        n_1q_gates=n_1q,
        n_qubits_used=n_qubits_used,
        gate_counts=gate_counts,
        cx_density=n_2q / max(1, ansatz.num_qubits),
        backend_name=getattr(backend, "name", "unknown"),
    )
    return transpiled, report


# ---------------------------------------------------------------------------
# Submission and fetching
# ---------------------------------------------------------------------------
def submit_job(
    transpiled: QuantumCircuit,
    backend,
    *,
    shots: int = 4096,
):
    """Submit a sampler job to ``backend``. Returns the job handle.

    ``job.job_id()`` is the value to persist if the queue wait is long
    enough that you want to close your laptop and come back later.
    """
    sampler = SamplerV2(mode=backend)
    return sampler.run([transpiled], shots=shots)


def fetch_result(job_id: str, service: QiskitRuntimeService | None = None):
    """Retrieve a job's result by ID. Blocks until the job has finished."""
    if service is None:
        service = get_service()
    job = service.job(job_id)
    return job.result()


def job_status(job_id: str, service: QiskitRuntimeService | None = None) -> str:
    """Return the current status string for a job (e.g. ``'QUEUED'``,
    ``'RUNNING'``, ``'DONE'``, ``'ERROR'``)."""
    if service is None:
        service = get_service()
    job = service.job(job_id)
    status = job.status()
    return str(status) if not hasattr(status, "name") else status.name


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------
def extract_counts(result) -> dict[str, int]:
    """Pull the measurement counts dict out of a SamplerV2 result.

    Result structure (V2): ``result[0].data.<register_name>.get_counts()``.
    For circuits where ``.measure_all()`` was called, the register is named
    ``'meas'``.
    """
    pub = result[0]
    data = pub.data
    # Try the conventional names in order.
    for reg_name in ("meas", "c", "creg"):
        reg = getattr(data, reg_name, None)
        if reg is not None and hasattr(reg, "get_counts"):
            return dict(reg.get_counts())
    # Fall back: introspect the data object's attributes.
    for attr_name in dir(data):
        if attr_name.startswith("_"):
            continue
        reg = getattr(data, attr_name, None)
        if reg is not None and hasattr(reg, "get_counts"):
            return dict(reg.get_counts())
    raise RuntimeError(
        "Could not locate a BitArray with get_counts() on the result. "
        "Inspect `result[0].data` manually."
    )

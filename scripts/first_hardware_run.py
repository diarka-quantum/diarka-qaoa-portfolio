"""
First quantum hardware run — Bell state on IBM Quantum.
Submits a 2-qubit entanglement circuit to real hardware and prints the results.
"""

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

# Connect to IBM Quantum (uses the token you saved earlier)
service = QiskitRuntimeService()

# Pick the least-busy real quantum computer available to your account
backend = service.least_busy(operational=True, simulator=False)
print(f"Submitting to backend: {backend.name}")
print(f"Current queue length: {backend.status().pending_jobs} jobs")

# Build a Bell state circuit (same as the simulator hello-world)
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

# Transpile the circuit to match the physical chip layout
qc_transpiled = transpile(qc, backend=backend, optimization_level=3)

# Submit the job
sampler = Sampler(mode=backend)
job = sampler.run([qc_transpiled], shots=1024)
print(f"\nJob submitted. Job ID: {job.job_id()}")
print("Waiting for results... (this can take a few minutes depending on queue)")

# Wait for completion and get results
result = job.result()
counts = result[0].data.meas.get_counts()

print(f"\nResults from {backend.name}:")
print(counts)

print("\nExpected: roughly 50/50 split between '00' and '11'")
print("Reality: some noise leakage into '01' and '10' — that's the real-hardware fingerprint")

# QAOA Portfolio Optimiser

Cardinality-constrained mean-variance portfolio selection solved with the Quantum Approximate Optimisation Algorithm (QAOA) on real IBM quantum hardware — with a full noise-to-mitigation analysis and an out-of-sample backtest.

This project takes one financial problem end to end: encode it as a QUBO, solve it with QAOA on a simulator and on a 133-qubit IBM Heron processor, explain every percentage point of the gap between them, and then ask the question that actually matters to an investor — did the resulting portfolio perform out-of-sample?

---

## Headline results

| Configuration | Approximation ratio | Gap to ideal |
|---|---:|---:|
| Noiseless simulator | **0.942** | — |
| Synthetic Heron noise (simulator) | 0.933 | ~0.9 pp |
| IBM `ibm_fez` (Heron r2, hardware) | **0.910** | 3.2 pp |

- QAOA recovered the **exact** classical cardinality-constrained optimum — the four-asset selection `HSBA, AZN, SHEL, ULVR` — not merely a near-optimal one, and that bitstring appeared in the real hardware shot distribution.
- The 3.2 pp simulator-to-hardware gap was **fully decomposed**: readout error, gate error, SWAP-routing overhead, and an irreducible crosstalk/coherent residual.
- Out-of-sample (2025), the quantum-selected portfolio returned **23.1% at a Sharpe of 1.31**, beating naive 1/N and minimum-variance, and matching an aggressive max-Sharpe optimiser that achieved its edge only by concentrating 60% into a single stock.

---

## The problem

An eight-asset FTSE universe — `HSBA.L, AZN.L, SHEL.L, ULVR.L, RIO.L, BT-A.L, VOD.L, DGE.L` — under a mean-variance objective with a hard cardinality constraint (select exactly four names, equal weight). This is a combinatorial selection problem: with a cardinality constraint it is no longer convex, which is precisely the kind of structure QAOA is designed for.

The objective is encoded as a QUBO and mapped to an Ising Hamiltonian (8 qubits, 36 Pauli terms). The ground state of that Hamiltonian — bitstring `00001111` — is the optimal portfolio.

## The approach

- **Classical baseline** — brute-force ground state plus continuous Markowitz portfolios (minimum-variance and maximum-Sharpe) for benchmarking.
- **QAOA** — depth `p = 1` ansatz built with Qiskit's `PauliEvolutionGate`, COBYLA parameter optimisation on an exact statevector estimator, then sampled on noiseless, noisy, and hardware backends. Depths `p = 2, 3` were explored for the noise/depth crossover.
- **Hardware** — IBM `ibm_fez` (Heron r2), Open Plan free tier.

---

## Results in detail

### Where the noise comes from

A synthetic Heron noise model reproduces the simulator-to-hardware gap channel by channel, switched on one at a time:

| Channel | Contribution to the simulated gap |
|---|---:|
| Readout / measurement | 0.42 pp |
| Two-qubit (CZ) gate | 0.27 pp |
| T1 / T2 decoherence | 0.06 pp |
| Single-qubit gate | 0.04 pp |

Readout dominates the *modelled* gap.

### Error mitigation

Two complementary techniques, each targeting a different channel — M3 (matrix-free measurement mitigation) for readout, and Zero-Noise Extrapolation (ZNE) for gate error. Their behaviour depends sharply on circuit connectivity:

| Technique | All-to-all transpile (56 CZ) | Heavy-hex routed (105 CZ) |
|---|---:|---:|
| M3 (readout) | +0.40 pp | +0.43 pp |
| ZNE (gate) | ~0 pp | **+0.87 pp** |
| M3 + ZNE | +0.46 pp | +1.25 pp |

ZNE is *useless* on the all-to-all circuit — its noise-scaling curve is flat because the gate channel is too small to amplify cleanly. Route the same circuit onto a real heavy-hex coupling map, the CZ count nearly doubles, the scaling curve slopes, and ZNE comes alive.

### The full gap decomposition

The 3.2 pp simulator-to-hardware gap breaks down as:

- **~0.9 pp** — readout, gate, decoherence (synthetic channel model)
- **~0.46 pp** — SWAP-routing overhead from embedding an all-to-all QUBO on degree-3 heavy-hex connectivity (CZ count 56 → 105)
- **~1.8 pp** — residual crosstalk and coherent / calibration error that an independent-channel noise model cannot capture

The honest finding: hand-built noise models explain roughly half the hardware gap; the rest is real-device physics that only a calibration-derived model or the device itself contains.

### Out-of-sample backtest (2025)

All weights fixed at the train/test boundary and held through the test window (buy-and-hold, no look-ahead):

| Portfolio | Return | Volatility | Sharpe | Max drawdown |
|---|---:|---:|---:|---:|
| **QAOA / QUBO optimum (4 assets)** | **23.1%** | 14.2% | **1.31** | −14.6% |
| Naive 1/N (8 assets) | 18.0% | 12.9% | 1.05 | −12.0% |
| Minimum-variance (8 assets) | 9.0% | 12.7% | 0.35 | −9.4% |
| Maximum-Sharpe (8 assets) | 28.9% | 16.2% | 1.51 | −15.4% |

The quantum-selected portfolio beat both diversified benchmarks on return and risk-adjusted return. The only portfolio that edged it did so by betting 60% on a single stock — a fragility the four-asset constraint structurally avoids.

*This is a single out-of-sample window: an illustration, not a statistical claim.*

---

## Repository structure

```
diarka-qaoa-portfolio/
├── src/
│   ├── data.py            # FTSE universe, price fetch, returns, annualisation
│   ├── classical.py       # Markowitz / brute-force / qiskit-finance baselines
│   ├── encoding.py        # QUBO -> Ising Hamiltonian
│   ├── qaoa.py            # QAOA circuit construction, optimisation, analysis
│   ├── noise_model.py     # synthetic Heron r2 noise model + channel decomposition
│   ├── backtest.py        # out-of-sample portfolio backtest harness
│   └── hardware.py        # IBM Quantum runtime helpers
├── notebooks/
│   ├── 01_problem_formulation.ipynb        # 8-asset universe, classical baselines
│   ├── 02_qubo_encoding.ipynb              # QUBO -> Ising Hamiltonian
│   ├── 03_first_qaoa_run.ipynb             # QAOA p=1 on the simulator
│   ├── 04_parameter_landscape.ipynb        # gamma/beta landscape, COBYLA
│   ├── 05_hardware_run.ipynb               # first run on ibm_fez (Heron r2)
│   ├── 06_robustness_and_sensitivity.ipynb # multi-seed + risk-aversion sweep
│   ├── 07_noise_analysis.ipynb             # synthetic Heron noise, channel decomposition
│   ├── 08_error_mitigation.ipynb           # M3 + ZNE (layered)
│   ├── 09_backtest.ipynb                   # out-of-sample portfolio backtest
│   └── 10_connectivity_and_routing.ipynb   # heavy-hex routing, gap decomposition
├── data/                  # raw + processed price panels (reproducibility)
├── figures/               # publication-ready figures (dark theme)
├── artifacts/             # saved Hamiltonian + optimal angles
├── results/               # numerical summaries
└── environment.yml        # reproducible Python 3.11 environment
```

## Reproduce

```bash
conda env create -f environment.yml
conda activate diarka-q
jupyter lab
```

Run the notebooks in numerical order. The simulator and backtest notebooks run with no credentials. The hardware notebook requires an IBM Quantum account token saved via `QiskitRuntimeService.save_account(...)`.

## Environment

Python 3.11, Qiskit 2.4, Qiskit Aer, Qiskit IBM Runtime, mthree, mitiq, Qiskit Finance / Optimization, yfinance, cvxpy. Full pinned versions in `environment.yml`.

## Limitations

- Eight-asset universe, depth `p = 1`, equal weights within the selection — a deliberately tractable scope for a NISQ-era hardware study, not a production allocator.
- The backtest covers a single out-of-sample window and is illustrative.
- The synthetic noise model is calibration-representative, not a snapshot of `ibm_fez` at run time; the ~1.8 pp residual reflects that.

---

## Author

Built by Nagesh, founder of **Diarka Quantum Limited** — the trusted bridge between quantum technology and business decision-makers.

- Web: [diarkaquantum.co.uk](https://diarkaquantum.co.uk)
- Contact: nageshj@diarkaquantum.co.uk

## License

Released under the MIT License — see `LICENSE`. *(Add a `LICENSE` file if one isn't present.)*

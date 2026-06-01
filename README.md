# PySCQED: Python Superconducting Circuit Quantum Electro-Dynamics Package

Critical TODOs:
- Parameter sweeps should be separated completely from the `NumericalSystem` class. Instead `NumericalSystem` should consume a sweep object and produce a result object.
- Separate "evaluables" internally and refactor the configuration data into a proper structure.
- No longer store simulation state in `NumericalSystem`, use a dedicated datastructure for that.
- Remove unused and unfinished features (phase slip, flux discretization)
- Implement the algorithm for block diagonalization (optimal basis representation for node degrees of freedom)
- Implement system partitioning support


PySCQED is a tool for simulating arbitrary, static superconducting quantum circuits. The package is currently in heavy development, however it currently has a number of features useful for modelling CQED circuits:
- Generate symbolic and numerical Hamiltonians of arbitrary circuits.
- Modular parameter sweeping system and extraction of quantities of interest.
- Circuit-resonator coupling (capacitive coupling only currently).
- Single qubit Hamiltonian reduction (local basis method).
- Operator extraction for performing dynamics simulations with time-dependent coefficients.

Graph visualisation uses [Graphviz](https://graphviz.org/download/).

Circuit visualisation will use [SchemDraw](https://schemdraw.readthedocs.io/en/latest/).

## Installation Using `poetry`

The Graphviz executables must be installed for the python module to work, which can be done either from the [Graphviz website](https://graphviz.org/download/) or using your distributions package manager. Ensure that the executables are added to the system path. Currently this doesn't appear to work out of the box on Windows and thus the path is added in `src/circuit_graph.py`. This might need to be changed depending on how Graphviz was installed.

The PySCQED package and its dependencies can be installed locally as follows:
```Shell
poetry install
```

## Quick Start

The best way to get started quickly is to build upon the examples provided in the `notebooks/` directory. There are several use cases covered, ordered from simplest to most advanced usage of the package:
- `transmon-example.py`: Basic simulation of a transmon qubit, replicating results obtained from a more approximated model of this circuit.
- `fluxonium-example.py`: Simulation of the fluxonium qubit reported by Alibaba. The results they obtain are reproduced numerically, including the qubit spectrum, resonator dispersive shift and qubit-qubit coupling energy.
- `csfq-example.py`: Simulations of the different types of capacitively shunted flux qubits, including the RF-SQUID and 3/4JJ flux qubits, and floating variants.
- `averin-coupler.py`: Extensive simulations and example of a process for designing a tunable capacitive coupling element for quantum annealing applications. This notebooks make extensive use of parametric expressions to express circuit components in terms of fabrication parameters and to approximate tuneable Josephson junctions. The response of a dispersively coupled resonator is also analysed.
- `fluxonium-atom.py`: Simulation of strongly inductively coupled fluxonium atoms, using some optimizations for minimizing the Hilbert space size.
- `resonator-example.py`: In depth analysis of a resonator coupled to a RF-SQUID flux qubit. The resonator dispersive shift, qubit Lamb shift and cavity photon-dependent AC-Stark shift are analysed.
- `local-basis-example.py`: Here we reduce both a RF-SQUID flux qubit and a Cooper-pair box qubit to the two-level system representation using the local basis reduction technique.
- `jpsq-example.py`: Simulations of Josephson Phase-Slip Qubits are performed, demonstrating how certain circuit components impact the tunneling interference physics of this device.

## Python Support Policy

We aim to only support python versions that receive bugfix updates, however the lowest supported version will typically linger. This typically means only three minor versions of python are supported.

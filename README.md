<!-- ## PySCQED: Python Superconducting Circuit Quantum Electro-Dynamics Package -->

<h1> PySCQED: Python Superconducting Circuit Quantum Electro-Dynamics Package </h1>

To be renamed PySCQED: Python Superconducting Circuit Quantum Electro-Dynamics package.

PySCQED is a tool for simulating arbitrary, static superconducting quantum circuits. The package is currently in heavy development, however it currently has a number of features useful for modelling CQED circuits:
* Generate symbolic and numerical Hamiltonians of arbitrary circuits.
* Modular parameter sweeping system and extraction of quantities of interest.
* Circuit-resonator coupling (capacitive coupling only currently).
* Single qubit Hamiltonian reduction (local basis method).

PySCQED is built on top of the following external packages:
* [QuTiP](http://qutip.org/): QuTiP is open-source software for simulating the dynamics of open quantum systems.
* [Networkx](https://networkx.github.io/): NetworkX is a Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.

and four packages from [SciPy](https://www.scipy.org/index.html) stack

* [NumPy](https://numpy.org/): NumPy is the fundamental package for scientific computing with Python.
* [SciPy](https://www.scipy.org/index.html): The SciPy library provides many user-friendly and efficient numerical routines such as routines for numerical integration, interpolation, optimization, linear algebra and statistics.
* [SymPy](https://www.sympy.org/en/index.html): SymPy is a Python library for symbolic mathematics.
* [Matplotlib](https://matplotlib.org/): Matplotlib is a Python 2D plotting library.

Graph visualisation uses [Graphviz](https://graphviz.org/download/).

Circuit visualisation will use [SchemDraw](https://schemdraw.readthedocs.io/en/latest/).

<!-- ### Installation and Usage -->

<h2> Installation using poetry </h2>

The Graphviz executables must be installed for the python module to work, which can be done either from the [Graphviz website](https://graphviz.org/download/) or using your distributions package manager. Ensure that the executables are added to the system path. Currently this doesn't appear to work out of the box on Windows and thus the path is added in `src/circuit_graph.py`. This might need to be changed depending on how Graphviz was installed.

The PySCQED package and its dependencies can be installed locally as follows:
```Shell
poetry install
```

<h2> Quick Start </h2>

The best way to get started quickly is to build upon the examples provided in the `notebooks/` directory. There are several use cases covered, ordered from simplest to most advanced usage of the package:
* `transmon-example.py`: Basic simulation of a transmon qubit, replicating results obtained from a more approximated model of this circuit.
* `fluxonium-example.py`: Simulation of the fluxonium qubit reported by Alibaba. The results they obtain are reproduced numerically, including the qubit spectrum, resonator dispersive shift and qubit-qubit coupling energy.
* `averin-coupler.py`: Extensive simulations and example of a process for designing a tunable capacitive coupling element for quantum annealing applications. This notebooks make extensive use of parametric expressions to express circuit components in terms of fabrication parameters and to approximate tuneable Josephson junctions. The response of a dispersively coupled resonator is also analysed.
* `resonator-example.py`: In depth analysis of a resonator coupled to a RF-SQUID flux qubit. The resonator dispersive shift, qubit Lamb shift and cavity photon-dependent AC-Stark shift are analysed.
* `local-basis-example.py`: Here we reduce both a RF-SQUID flux qubit and a Cooper-pair box qubit to the two-level system representation using the local basis reduction technique.
* `jpsq-example.py`: Simulations of Josephson Phase-Slip Qubits are performed, demonstrating how certain circuit components impact the tunneling interference physics of this device.

<h2> Usage </h2>

Coming soon.



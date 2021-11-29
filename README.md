<!-- ## PyCQED: Python Circuit Quantum Electro-Dynamics Package -->

<h1> PyCQED: Python Circuit Quantum Electro-Dynamics Package </h1>

PyCQED is an attempt to combine modelling of quantum circuits and generation of data useful for performing experiments on real quantum circuits. For example, one may have designed a circuit that emulates two coupled superconducting qubits. The experiments required to test the dynamics of the circuit will require a series of control signals and procedures. Hopefully, PyCQED can be used to generate such signals based on the modelling results. Perhaps it is possible to develop a feedback procedure, where this package can be used to perform analysis that informs refinement of the model, which in turn can be used to generate new control signals and procedures. The package is currently in heavy development, however it currently has a number of features useful for modelling CQED circuits:
* Generate symbolic and numerical Hamiltonians of arbitrary circuits.
* Modular parameter sweeping system and extraction of quantities of interest.
* Circuit-resonator coupling (capacitive coupling only currently).
* Single qubit Hamiltonian reduction (local basis method).

A few additional capabilities are in development also, intended as extensions for using the results of modelling CQED circuits:
* Create arbitrary Ising Hamiltonians from graphs.
* Simulation and analysis of quantum annealing processes.
* Simulation and analysis of quantum computing processes.

PyCQED is built on top of the following external packages:
* [QuTiP](http://qutip.org/): QuTiP is open-source software for simulating the dynamics of open quantum systems.
* [Networkx](https://networkx.github.io/): NetworkX is a Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.
* [SchemDraw](https://pypi.org/project/SchemDraw/): SchemDraw is a python package for producing high-quality electrical circuit schematic diagrams.

and four packages from [SciPy](https://www.scipy.org/index.html) stack

* [NumPy](https://numpy.org/): NumPy is the fundamental package for scientific computing with Python.
* [SciPy](https://www.scipy.org/index.html): The SciPy library provides many user-friendly and efficient numerical routines such as routines for numerical integration, interpolation, optimization, linear algebra and statistics.
* [SymPy](https://www.sympy.org/en/index.html): SymPy is a Python library for symbolic mathematics.
* [Matplotlib](https://matplotlib.org/): Matplotlib is a Python 2D plotting library.

<!-- ### Installation and Basic Usage -->

<h2> Installation </h2>

The PyCQED package and its dependencies can be installed locally as follows (on Linux using python 3.x):

```Shell
python3 setup.py install --user
```

In some cases the dependencies of the required packages may not be resolved and the installation will fail. It might help to manually install the dependencies before running the setup script, as follows (on Linux using python 3.x):

```Shell
pip3 install $(cat requirements.txt)
```

<h2> Core Basic Usage </h2>

There are two main classes that are used together to generate Hamiltonians for arbitrary quantum circuits:

* The `CircuitSpec` class is used to draw a circuit and generate the constitutive equations of the system under different representations.
* The `HamilSpec` class uses an instance of the above class under a specific representation (currently 'node' or 'branch' representations) to generate operators and a Hamiltonian that can be diagonalised.
* The `HierarchicalHamilSpec` class uses an instance of a `CircuitSpec` class to define a series of subcircuits. It is then possible to approximate the total Hamiltonian by focusing on the low energy part of each subcircuit and their interactions.

The sphinx documentation is currently under construction, the homepage of which is `docs/_build/html/index.html`.

See the examples in the `Notebooks` directory for detailed usage:
* Exploring the capacitively shunted qubit (CSFQ) in `Notebooks/csfq-example.ipynb`
* Exploring single qubit reduction with the local basis technique in `Notebooks/local-basis-example.ipynb`
* Exploring the qubit-resonator interaction in `Notebooks/resonator-example.ipynb`
* Exploring the Averin coupler and advanced use of parameterisations in `Notebooks/averin-coupler.ipynb`
* Exploring the Josephson phase-slip qubit (JPSQ) in `Notebooks/jpsq-example.ipynb`
* Using mutual inductances in `Notebooks/mutual-inductance-example.ipynb`
* Checking the stoquasticiy of Hamiltonians in `Notebooks/stoquastic.ipynb`
* Exploring multi-qubit systems in `Notebooks/subcircuits-example.ipynb`

Description of the relevant source files:
* `circuitspec.py`: Circuit drawing and symbolic representation of the system equations.
* `hamilspec.py`: Generation and solving of circuit Hamiltonians.
* `dataspec.py`: Management of memory and data.
* `parameters.py`: Helper classes for manipulating model parameters.
* `units.py`: Helper library for converting between different unit systems.
* `util.py`: A collection of useful functions for processing data and performing generic operations.
* `physical_constants.py`: Physical constants in SI units.

<h2> Extensions Basic Usage </h2>

To simulate quantum annealing processes, two classes can be used in tandem:

* The `IsingGraph` class is used to initialise arbitrary Ising Hamiltonians including higher order terms.
* The `QuantumAnnealing` class is used to analyse and simulate quantum annealing processes.

The following examples are available in the `Notebooks` directory:
* Simulation and analysis of a weighted-MIS annealing problem in `Notebooks/ising-model-example.ipynb`



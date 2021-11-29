# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `CircuitSpec` function `addBranchToGraph` to create the circuit without actually drawing it.
- Sub circuits example notebook.
- Mutual inductance example notebook.
- `HierchicalHamilSpec` class for creating subcircuits out of a bigger circuit and efficiently simulating the whole system.
- `isingmodel.py` source file that implements the classes `IsingGraph` and `QuantumAnnealing`, and quantum annealing schedule functions.
- `CircuitSpec` functions to hierarchize a large circuit into subcircuits (instances of `SubCircuitSpec`).
- `SubCircuitSpec` class that inherits from `CircuitSpec` with some overloaded functions.
- `CircuitSpec` function `drawSubGraph` for drawing sub graphs.
- `TempData` class for managing temporary data. Currently only used in the `HamilSpec` class to write large sweeps to disk to avoid excessive RAM usage.
- `CircuitSpec` function `addGroundNode` that creates a new ground node from which to draw from. Useful for drawing circuit coupled only via a mutual inductance.
- `CircuitSpec` function `coupleBranchesImplicitly` to couple branches via a mutual inductance even when the self inductance of the branch is neglected.
- Utility function to check if a Hamiltonian is non-stoquastic.
- `HamilSpec` functions for extracting subsystem derived parameters.
- Utility functions for extracting Lamb and AC Stark shifts due to a coupled subsystem.
- Local basis example notebook.
- Utility function for getting Pauli coefficients using G. Consani's local basis method.
- Branch current and node voltage operator calculations.
- Unit system customizer.
- Ability to time parameter sweeps.
- `diagonalize` function in `HamilSpec`. Diagonalizer is preconfigured using `setDiagConfig` function in HamilSpec.
- `util.py` for utility functions used internally and accessible by the user.
- Resonator example notebook.
- Resonator response function.
- New sweep function in `HamilSpec` to compute user functions.
- `CircuitSpec` function to couple a linear resonator capacitively to a node.
- Copy function for cloning `CircuitSpec` instances.
- Flux bias symmetrisation function.
- Parameterisation removal function.
- Single particle energy symbolic expressions.

### Changed

- Changed the way flux bias terms are added, now terms are not omitted under certain conditions.
- Updated local-basis-example notebook to include charge qubits.
- Changed the way subsystems are treated.
- Significant refactoring of internal `HamilSpec` code.
- Updated csfq-example notebook to include new changes.
- JJ bias terms now expressed as an exponential in latex output.
- `HamilSpec` now takes an optional Unit customizer.
- Changed the way parameter sweeps are performed. Diagonalizer arguments are passed to `setDiagConfig` function in `HamilSpec` instead of paramSweep.
- Changed the way diagonalization is done. Diagonalizer needs to be preconfigured using `setDiagConfig` function in `HamilSpec`.
- Moved `_multidot` out of `HamilSpec` class into `util.py` and renamed to `mdot`.
- Stopped flux bias terms being created for non-superconducting loop closure branches.
- Changed default JJ prefix from J to I, as it is a critical current that is specified.

### Removed

- Obsolete notebooks.
- Calculation of flux bias terms across inductors.
- Unused and redundant functions.

## [0.11] - 2019-11-05

### Added

- Circuit drawing class that generates circuit graph, symbolic equations and registers the specified components.
- Hamiltonian generation class that can be used to model numerically the dependence of the energy levels on external controls.
- Feature: Parameterisation of parameters, useful for simulating in terms of fabrication parameters for example.
- Feature: Multi-dimensional parameter sweeps in a compact format.

### Changed

- First release

### Removed

- First release


[unreleased]: https://tef1.physics.tamu.edu/ucanlfr/pycqed/compare/v0.11...master
[0.11]: https://tef1.physics.tamu.edu/ucanlfr/pycqed/tags/v0.11

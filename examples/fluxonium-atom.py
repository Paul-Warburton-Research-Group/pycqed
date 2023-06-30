# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import qutip as qt
from pyscqed import *
from pyscqed import physical_constants as pc

# Here we reproduce the work from [here](https://arxiv.org/pdf/1610.01094.pdf) on the fluxonium atom.
#
# A trick is used to describe the circuit with fewer degrees of freedom. The circuit at face value has three circuit nodes. However there is an inductor T structure, that is an equivalent circuit for mutual inductance. By coupling two independent loops via a virtual mutual inductance, the number of degrees of freedom is reduced by one. Following [this](https://www.uni-ruse.bg/disciplines/TE/Lecture%20notes/Lectures%20notes%20Mutually%20coupled%20inductors.pdf), we can express the central inductor as $M$, and then since the two loop inductors will be $L_{1,2} - M$, we can ensure that all inductances in the equivalent circuit are equal by substituting $L_{1,2} = 2M$.

graph = CircuitGraph()
graph.addBranch(0, 1, "L1")
graph.addBranch(0, 1, "I1")
graph.addBranch(0, 1, "C1")
graph.addBranch(0, 2, "L2")
graph.addBranch(0, 2, "I2")
graph.addBranch(0, 2, "C2")
graph.coupleBranchesInductively("L1", "L2", "M")
graph.addFluxBias("I1", "l")
graph.addFluxBias("I2", "r")
graph.drawGraphViz()

circuit = SymbolicSystem(graph)
circuit.getQuantumHamiltonian()

# +
hamil = NumericalSystem(circuit)
hamil.configureOperator(1, 31, "oscillator")
hamil.configureOperator(2, 31, "oscillator")

# Apply the coordinate transformation
hamil.getSymbolicSystem().getSymbols("phi+", "phi-", "L")
sym = hamil.getSymbolicSystem().getSymbolList()
hamil.getSymbolicSystem().addParameterisation("phil", sym["phi+"] + sym["phi-"])
hamil.getSymbolicSystem().addParameterisation("phir", sym["phi+"] - sym["phi-"])

# Apply the inductance equivalent circuit
hamil.getSymbolicSystem().addParameterisation("M", sym["L"])
hamil.getSymbolicSystem().addParameterisation("L1", 2*sym["L"])
hamil.getSymbolicSystem().addParameterisation("L2", 2*sym["L"])
# -

hamil.getHilbertSpaceSize()

# +
# Use the energy scales of device A
Ec = 3.4e9
El = 1.2e9*0.5
Ej = 9.4e9
alpha = 0.006

# Derive the component values
C = 0.5*pc.e**2/(pc.h*Ec) * 1e15
L = 0.5*pc.phi0**2/(pc.h*El*4*np.pi**2) * 1e12
I = 2*np.pi*pc.h*Ej/(pc.phi0) * 1e6
# -

hamil.setParameterValues(
    'L', L,
    'C1', C,
    'I1', I,
    'C2', C,
    'I2', I,
    'phi+', 0.0,
    'phi-', 0.0
)

# Configure the parameter sweep
hamil.newSweep()
hamil.addSweep('phi-', 0, 1.0, 101)
hamil.setParameterValue('phi+', 0.0)
sweep = hamil.paramSweep(timesweep=True)

x,E,v = hamil.getSweep(sweep, 'phi-', {})
gaps = []
for i in range(5):
    gaps.append(E[i]-E[0])
    plt.plot(x, gaps[i])
plt.xlabel("$\\phi_+$ [$\\phi_0$]")
plt.ylabel("Potential [GHz]")

# Configure the parameter sweep
hamil.newSweep()
hamil.addSweep('phi+', 0, 1.0, 101)
hamil.setParameterValue('phi-', 0.0)
sweep = hamil.paramSweep(timesweep=True)

x,E,v = hamil.getSweep(sweep, 'phi-', {})
gaps = []
for i in range(5):
    gaps.append(E[i]-E[0])
    plt.plot(x, gaps[i])
plt.xlabel("$\\phi_+$ [$\\phi_0$]")
plt.ylabel("Potential [GHz]")



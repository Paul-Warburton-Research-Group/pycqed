# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

from pyscqed import *
from pyscqed import physical_constants as pc

# Here we simply reproduce the results from [this](http://antonpotocnik.com/?p=560257) website:

graph = CircuitGraph()
graph.addBranch(0, 1, "C")
graph.addBranch(0, 1, "I1")
graph.addBranch(0, 1, "I2")
graph.addFluxBias("I1", "Z")
graph.drawGraphViz()

circuit = SymbolicSystem(graph)
circuit.getQuantumHamiltonian()

hamil = NumericalSystem(circuit)
hamil.configureOperator(1, 40, "charge")
hamil.setParameterValues(
    "C", 64.568, # In fF
    "I1", 16.1132e-3/2, # In uA
    "I2", 16.1132e-3/2, # In uA
    "phiZ", 0
)

E = hamil.getHamiltonian().eigenenergies()
print("fge =", E[1] - E[0])
print("fef =", E[2] - E[1])
print(E[3] - E[2])

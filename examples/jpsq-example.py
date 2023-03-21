# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
import matplotlib.pyplot as plt
from pyscqed import *
from pyscqed.util import *

# # Josephson Phase-Slip Qubit Example
#
# The JPSQ is a new type of qubit that exploits interference effects between fluxon tunneling amplitudes to linearly tune the transverse field by way of a gate electrode. With this circuit it becomes possible to fully emulate a spin-half. If such a qubit can be successfully realised it is understood that quantum annealing hardware can be used to emulate spin systems beyond the simple transverse Ising model.
#
# ## Two Junction SQUID
#
# We first look at the device described by [Friedman and Averin](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.88.050403) to investigate the suppression of macroscopic tunnelling by the Aharonov-Casher effect.

graph = CircuitGraph()
graph.addBranch(0, 1, "C1")
graph.addBranch(0, 1, "I1")
graph.addBranch(1, 2, "C2")
graph.addBranch(1, 2, "I2")
graph.addBranch(0, 2, "L1")
graph.addChargeBias(1, "Cg1")
graph.drawGraphViz()

circuit = SymbolicSystem(graph)

circuit.getQuantumHamiltonian()

Ca = 60 # fF/um^2
Jc = 3  # uA/um^2
Aj = 0.3**2 # um^2
h = NumericalSystem(circuit)
h.configureOperator(1, 10, "charge")
h.configureOperator(2, 20, "oscillator")
h.setParameterValues(
    'C1', Ca*Aj,
    'C2', Ca*Aj,
    'I1', Jc*Aj,
    'I2', Jc*Aj,
    'L1', 100.0,
    'Cg1', 0.0,
    'phi21-1e', 0.0,
    'Q1e', 0.0
)

h.newSweep()
h.addSweep('phi21-1e', 0.0, 1.0, 101)
sweep = h.paramSweep(timesweep=True)

x,sweep_p,v = h.getSweep(sweep,'phi21-1e',{})
for i in range(5):
    y = sweep_p[i] - sweep_p[0]
    plt.plot(x,y)
plt.xlabel("$\\Phi_{21e}$ ($\\Phi_0$)")
plt.ylabel("$E_{g,i}$ (GHz)")

h.setParameterValues(
    'phi21-1e', 0.5,
    'Q1e', 0.0
)
h.newSweep()
h.addSweep('Q1e', 0.0, 1.0, 101)
sweep = h.paramSweep(timesweep=True)

x,sweep_q,v = h.getSweep(sweep,'Q1e',{})
for i in range(5):
    y = sweep_q[i] - sweep_q[0]
    plt.plot(x,y)
plt.xlabel("$Q_{1e}$ ($2e$)")
plt.ylabel("$E_{g,i}$ (GHz)")

# In this device, a finite gate capacitance, that is required in a real device to induce an offset charge, actually lifts the interference effect to some extent. We can observe this by sweeping the capacitance:

h.setParameterValues(
    'phi21-1e', 0.5,
    'Q1e', 0.5
)
h.newSweep()
h.addSweep('Cg1', 0.0, 10.0, 101)
sweep = h.paramSweep(timesweep=True)

x,sweep_c,v = h.getSweep(sweep,'Cg1',{})
for i in range(2):
    y = sweep_c[i] - sweep_c[0]
    plt.plot(x,y)
plt.xlabel("$C_{g1}$ (fF)")
plt.ylabel("$E_{g,i}$ (GHz)")

# ## Grounded 4-JJ JPSQ
#
# This Josephson Phase-Slip Qubit (JPSQ) circuit builds on the lessons learnt from the 3-JJ qubit, where large JJs are used as large inductors. However the smaller junction is now replaced with two junctions in series, that are possibly much larger than the other two, in between which is a very small island with a small self-capacitance. Furthermore, this design doesn't suffer from the incomplete interference caused by a finite gate capacitance.

graph = CircuitGraph()
graph.addBranch(0, 1, "C1")
graph.addBranch(0, 1, "I1")
graph.addBranch(1, 2, "C2")
graph.addBranch(1, 2, "I2")
graph.addBranch(2, 3, "C3")
graph.addBranch(2, 3, "I3")
graph.addBranch(0, 3, "C4")
graph.addBranch(0, 3, "I4")
graph.addBranch(1, 3, "Csh")
graph.addChargeBias(2, "Cg2")
graph.drawGraphViz()

circuit = SymbolicSystem(graph)

circuit.getQuantumHamiltonian()

# Now we choose an initial truncation for each degree of freedom:

h = NumericalSystem(circuit)
h.configureOperator(1, 4, "charge")
h.configureOperator(2, 12, "charge")
h.configureOperator(3, 4, "charge")
h.getHilbertSpaceSize()

# Let's now check what parameters are required for the sparse eigensolver:

Ca = 60 # fF/um^2
Jc = 3  # uA/um^2
Aj = 0.3**2 # um^2
alpha = 10.0
h.setParameterValues(
    'C1',Ca*Aj,
    'C2',Ca*Aj*alpha,
    'C3',Ca*Aj*alpha,
    'C4',Ca*Aj,
    'I1',Jc*Aj,
    'I2',Jc*Aj*alpha,
    'I3',Jc*Aj*alpha,
    'I4',Jc*Aj,
    'Cg2',5.0,
    'Csh',100.0,
    'phi32-1e',0.0,
    'Q2e',0.0
)

H = h.getHamiltonian()
print ("Hamiltonian Sparsity: %f" % h.sparsity(H))

# The sparsity is indeed very high thus a sparse eigensolver is justified, now we look at the required value of $\sigma$:

# +
h.setParameterValues('phi32-1e',0.5,'Q2e',0.0)
H = h.getHamiltonian()
E = H.eigenenergies()
print (E[0])
print (E[1]-E[0])
print()

h.setParameterValues('phi32-1e',0.0,'Q2e',0.0)
H = h.getHamiltonian()
E = H.eigenenergies()
print (E[0])
print (E[1]-E[0])
print()

h.setParameterValues('phi32-1e',0.5,'Q2e',0.5)
H = h.getHamiltonian()
E = H.eigenenergies()
print (E[0])
print (E[1]-E[0])
print()

h.setParameterValues('phi32-1e',0.0,'Q2e',0.5)
H = h.getHamiltonian()
E = H.eigenenergies()
print (E[0])
print (E[1]-E[0])
# -

# ## Flux Dependence of the Energy
#
# Let's now look at the lowest eigenvalues as a function of externally applied flux, first when there is no charge on the island:

# +
h.newSweep()
h.addSweep('phi32-1e',0.0,1.0,101)
h.setParameterValue('Q2e',0.0)

# Configure diagonalizer
opts = {"sigma":-2777, "mode":"normal", "maxiter":None, "tol":0}
h.setDiagConfig(sparse=True, sparsesolveropts=opts)

# Do the sweep
sweep = h.paramSweep(timesweep=True)
# -

x,sweep_p,v = h.getSweep(sweep,'phi32-1e',{})
for i in range(5):
    y = sweep_p[i] - sweep_p[0]
    plt.plot(x,y)
plt.xlabel("$\\Phi_{32e}$ ($\\Phi_0$)")
plt.ylabel("$E_{g,i}$ (GHz)")

# Now let's look at the spectra when there is half a Cooper-pair's worth of offset charge on the island:

h.newSweep()
h.addSweep('phi32-1e', 0.0, 1.0, 101)
h.setParameterValue('Q2e', 0.5)
sweep = h.paramSweep(timesweep=True)

x, sweep_p, v = h.getSweep(sweep, 'phi32-1e', {})
for i in range(5):
    y = sweep_p[i] - sweep_p[0]
    plt.plot(x, y)
plt.xlabel("$\\Phi_{32e}$ ($\\Phi_0$)")
plt.ylabel("$E_{g,i}$ (GHz)")

# ### Charge Dependence of the Energy
#
# Now we sweep the charge offset applied to the island:

h.newSweep()
h.addSweep('Q2e', 0.0, 1.0, 101)
h.setParameterValue('phi32-1e', 0.5)
sweep = h.paramSweep(timesweep=True)

x, sweep_q, v = h.getSweep(sweep, 'Q2e', {})
for i in range(3):
    y = sweep_q[i] - sweep_q[0]
    plt.plot(x, y)
plt.xlabel("$Q_{2e}$ ($2e$)")
plt.ylabel("$E_{g,i}$ (GHz)")

# ### Island Self-Capacitance
#
# Now we vary the capacitance of the charging island:

h.newSweep()
h.addSweep('Cg2', 0.0, 100.0, 101)
h.setParameterValues('phi32-1e', 0.5, 'Q2e', 0.0)
sweep = h.paramSweep(timesweep=True)

x,sweep_c,v = h.getSweep(sweep,'Cg2',{})
for i in range(5):
    y = sweep_c[i] - sweep_c[0]
    plt.plot(x,y)
plt.xlabel("$C_{g2}$ (pF)")
plt.ylabel("$E_{g,i}$ (GHz)")

# ### Shunt Capacitor Across Both Junctions
#
# Now we look at the dependence of the spectrum on the shunt capacitor:

h.newSweep()
h.addSweep('Csh', 0.0, 100.0, 101)
h.setParameterValues('phi32-1e', 0.5, 'Q2e', 0.0, 'Cg2', 20.0)
sweep = h.paramSweep(timesweep=True)

x,sweep_c,v = h.getSweep(sweep,'Csh',{})
for i in range(5):
    y = sweep_c[i] - sweep_c[0]
    plt.plot(x,y)
plt.xlabel("$C_{sh}$ (pF)")
plt.ylabel("$E_{g,i}$ (GHz)")



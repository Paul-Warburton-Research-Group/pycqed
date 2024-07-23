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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pyscqed import *
from pyscqed import physical_constants as pc

# ## Fluxonium Qubit Example
#
# Here we will replicate the results presented [here](https://arxiv.org/abs/2111.13504) by Alibaba Group.

graph = CircuitGraph()
graph.addBranch(0, 1, "C")
graph.addBranch(0, 1, "L")
graph.addBranch(0, 1, "I")
graph.addFluxBias("I", "Z")
graph.coupleResonatorCapacitively(1, "Cc")
graph.drawGraphViz()

circuit = SymbolicSystem(graph)
circuit.getQuantumHamiltonian()

# ### Qubit A
#
# We will first reproduce Fig 1(c). See Table S1 for the parameter values.

# Define the qubit parameters and resonator frequency
EcA = 1.398e9
ElA = 0.523e9*0.5
EjA = 2.257e9
frA = 6.696 # GHz
ZrA = 50.0

# We'll assume that Cc is approximately zero and see what we get first
C = 0.5*pc.e**2/(pc.h*EcA)
print(C*1e15, "fF")

# Inductance energy is simple
L = 0.5*pc.phi0**2/(pc.h*ElA*4*np.pi**2)
print(L*1e12, "pH") # Large inductance

# Josephson junction critical current
I = 2*np.pi*pc.h*EjA/(pc.phi0)
print(I*1e6, "uA") # Very small J energy

# Here we must use the oscillator (or flux) basis rather than the charge basis, as the inductive term will play an important role in the potential energy. We'll use a truncation of 101 states.

circuit.drawParameterisationGraph()

hamil = NumericalSystem(circuit)
hamil.configureOperator(1, 101, "oscillator")
hamil.setParameterValues(
    "C", C*1e15-0.3, # In fF
    "I", I*1e6, # In uA
    "L", L*1e12, # In pH
    "Cc", 0.3,
    "f1r", frA,
    "Z1r", ZrA,
    "phiZ", 0.5
)

# Verify what we get for the energies using these circuit parameters

hamil.getChargingEnergies(1)/4

hamil.getFluxEnergies(1)

hamil.getJosephsonEnergies((1, 0, 2))

# For the transition frequencies we'll use qutips method on the underlying Qobj:

E = hamil.getHamiltonian().eigenenergies()

print("w10/2pi =", E[1] - E[0])
print("w21/2pi =", E[2] - E[1])

# We see the parameters we get are pretty close to what they have reported. The discrepancy is likely due to the inexact value of inductance and the details of their model implementation, for example the size of their truncation could play role.

# Configure the sweep
hamil.addSweep('phiZ', 0.25, 0.6, 101)
sweep = hamil.paramSweep(timesweep=True)

# +
# Get the sweep for a high value of Ic
x, phi10e_sweep, v = hamil.getSweep(sweep, 'phiZ', {})

fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(4, 6))

# Energy spectrum
for i in range(5):
    y = phi10e_sweep[i]-phi10e_sweep[0]
    ax.plot(x,y)
ax.set_xlabel("$\Phi_{Z}$ ($\Phi_0$)")
ax.set_ylabel("$E_{g,i}$ (GHz)")
ax.set_title("Energy Spectrum")
ax.set_ylim(0, 8)
#ax.set_xlim(0.49, 0.51)
# -

# As we can see the data looks fairly consistent with their figure.
#
# ### Resonator and Dispersive Shift
#
# Now let's see what they might have observed in terms of resonator response from flux modulation:

# Looking at the qubit-resonator coupling energy,
# it is complicated, so we can adjust Cc to get what we need
circuit.getParametricExpression('g1r')

# +
# Configure the parameter sweep
hamil.newSweep()
hamil.addSweep('phiZ', -1.0, 1.0, 201)

# Configure the items to be evaluated
hamil.addEvaluation('Hamiltonian'),
hamil.addEvaluation('Resonator', cpl_node=1)

# Configure diagonaliser to give us eigenvectors
hamil.setDiagConfig(get_vectors=True, eigvalues=20)

# Do the sweep
sweep = hamil.paramSweep(timesweep=True)
# -

# Let's look at the dressed qubit spectrum first

# +
# Get the design and loaded resonator frequencies
fr = hamil.getParameterValue('f1r')
frl = hamil.getParameterValue('f1rl')

x,Erwa,v = hamil.getSweep(sweep,'phiZ',{},evaluable='Resonator')
Edressed = util.getCircuitLambShift(Erwa)
# -

# Dressed energy spectrum
for i in range(5):
    y = Edressed[i]
    plt.plot(x,y)
plt.plot([x[0], x[-1]], [fr, fr], "k--")
plt.plot([x[0], x[-1]], [frl, frl], "r--")
plt.xlabel("$\\Phi_{Z}$ ($\\Phi_0$)")
plt.ylabel("$E_{g,i}$ (GHz)")
plt.title("(Loaded) Dressed Coupler Energy Spectrum")

# We shouldn't see any avoided crossings here as the qubit energy never crosses the resonator energy. We do see however that near $\Phi_{10e} = 0.5 \Phi_0$, the resonator will be strongly coupled to the 3rd excited state of the qubit as pointed out in the text.
#
# Let's first look at the single cavity photon resonator shifts when the qubit is in the ground state:

# Get the modulated resonator frequency
Eres = util.getResonatorShift(Erwa)

plt.plot(x,Eres[0,0])
plt.plot([x[0], x[-1]], [fr, fr], "k--")
plt.plot([x[0], x[-1]], [frl, frl], "r--")
plt.xlabel("$\\Phi_{Z}$ ($\\Phi_0$)")
plt.ylabel("$\omega_{r}/2\pi$ (GHz)")
plt.title("Resonator Modulation Against $Q_{1e}$")

# And when the qubit is in the first excited state:

plt.plot(x,Eres[1, 0])
plt.plot([x[0], x[-1]], [fr, fr], "k--")
plt.plot([x[0], x[-1]], [frl, frl], "r--")
plt.xlabel("$\\Phi_{Z}$ ($\\Phi_0$)")
plt.ylabel("$\omega_{r}/2\pi$ (GHz)")
plt.title("Resonator Modulation Against $Q_{1e}$")

# The single-photon dispersive shift $\chi$ due to transitions between the ground and first excited states will then simply be:

plt.plot(x, (Eres[1, 0] - Eres[0, 0])*1e3)
plt.xlabel("$\\Phi_{Z}$ ($\\Phi_0$)")
plt.ylabel("$\chi/2\pi$ (MHz)")
plt.title("Resonator Modulation Against $\\phi_{Z}$")

# Indeed this appears to be close to the value report of 0.63 MHz.
#
# ### Coupling Energy
#
# Now we'll reproduce the avoided crossing due to coupling between qubits A and B.

graph = CircuitGraph()
graph.addBranch(0, 1, "CA")
graph.addBranch(0, 1, "LA")
graph.addBranch(0, 1, "IA")
graph.addBranch(0, 2, "CB")
graph.addBranch(0, 2, "LB")
graph.addBranch(0, 2, "IB")
graph.addBranch(1, 2, "Cc")
graph.addFluxBias("IA","ZA")
graph.addFluxBias("IB","ZB")
graph.drawGraphViz()

circuit = SymbolicSystem(graph)
circuit.getQuantumHamiltonian()

# The capacitive coupling energy between qubits A and B will be determined by the off-diagonal elements of the inverse capacitance matrix:

# +
# Define the qubit parameters and resonator frequency
EcA = 1.398e9
ElA = 0.523e9*40*0.5
EjA = 2.257e9
EcB = 1.572e9
ElB = 0.537e9*40*0.5
EjB = 2.086e9

CA = 0.5*pc.e**2/(pc.h*EcA)
CB = 0.5*pc.e**2/(pc.h*EcB)
LA = 0.5*pc.phi0**2/(pc.h*ElA)
LB = 0.5*pc.phi0**2/(pc.h*ElB)
IA = 2*np.pi*pc.h*EjA/(pc.phi0)
IB = 2*np.pi*pc.h*EjB/(pc.phi0)

hamil = NumericalSystem(circuit)
hamil.configureOperator(1, 31, "oscillator")
hamil.configureOperator(2, 31, "oscillator")
hamil.setParameterValues(
    "CA", CA*1e15, # In fF
    "CB", CB*1e15, # In fF
    "IA", IA*1e6, # In uA
    "IB", IB*1e6, # In uA
    "LA", LA*1e12, # In pH
    "LB", LB*1e12, # In pH
    "Cc", 1.0,
    "phiZA", 0.5,
    "phiZB", 0.5
)
# -

# To reproduce the data shown in Figure 1(e), we'll set $\Phi_{20e} = 0.5 \Phi_0$ and sweep only $\Phi_{10e}$:

# Configure the sweep
hamil.addSweep('phiZA', 0.4, 0.5, 101)
sweep = hamil.paramSweep(timesweep=True)

# +
# Get the sweep for a high value of Ic
x, phi10e_sweep, v = hamil.getSweep(sweep, 'phiZA', {})

fig, ax = plt.subplots(1, 2, constrained_layout=True)

# Energy spectrum
for i in range(5):
    y = phi10e_sweep[i]-phi10e_sweep[0]
    ax[0].plot(x,y)
ax[0].set_xlabel("$\Phi_{Z}$ ($\Phi_0$)")
ax[0].set_ylabel("$E_{g,i}$ (GHz)")
ax[0].set_title("Energy Spectrum")
#ax[0].set_ylim(0.5, 2)
#ax[0].set_xlim(0.49, 0.51)

# Energy spectrum
for i in range(5):
    y = phi10e_sweep[i]-phi10e_sweep[0]
    ax[1].plot(x,y)
ax[1].set_xlabel("$\Phi_{Z}$ ($\Phi_0$)")
ax[1].set_ylabel("$E_{g,i}$ (GHz)")
ax[1].set_title("Energy Spectrum")
ax[1].set_ylim(1.15, 1.3)
#ax[1].set_xlim(0.49, 0.51)
# -

# We see that we get a coupling $g$ on the order of 20 MHz in a a similar place to what Alibaba observed. We could slightly reduce the capacitor $C_c$ but this is close enough.



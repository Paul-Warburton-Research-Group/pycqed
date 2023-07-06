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
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import sympy as sy
import time
from pyscqed import *

# # Capacitively Shunted Flux Qubit Example
#
# The CSFQ circuit is the result of a series of improvements on the original flux qubit that included a large inductive loop in conjunction with a Josephson junction to create a double well potential. The large inductance was replaced with Josephson junctions to allow the loop to retain a large inductance while reducing the loop area, which decreases it's coupling to the environment and in turn reduces flux noise. To create a large tunneling amplitude, one junction is made smaller to reduce the tunnel barrier height. In doing so the capacitance of the junction can become very small, which increases the impact of charge fluctuations due to Coulomb blockade effects. This is mitigated by incorporating a large shunt capacitor across the small junction to significantly reduce the associated charging energy, and thus the impact of charge fluctuations.
#
# Different regimes of the CSFQ circuit exist, here we describe each one.
#
# ## Traditional Flux Qubit (RF-SQUID)
#
# This circuit includes a single Josephson junction and a large inductor that controls the shape of the double well potential. We first draw the circuit and define the components that belong to each branch as such:

graph = CircuitGraph()
graph.addBranch(0, 1, "L")
graph.addBranch(0, 1, "I")
graph.addBranch(0, 1, "C")
graph.addFluxBias("I", "Z")
graph.drawGraphViz()

circuit = SymbolicSystem(graph)
circuit.getQuantumHamiltonian()

circuit.getClassicalHamiltonian()

hamil = NumericalSystem(circuit)

hamil.configureOperator(1,20,"charge")

hamil.getHilbertSpaceSize()

hamil.prepareOperators()

# A full set of operators in the specified basis is generated for each node, including the charge, phase, and displacement operators. These can be accessed using the nested dictionary structure. For example here we get the charge operator associated with node 1:

hamil.circ_operators[1]['charge']

# Now we are ready to substitute parameter values. The available parameters can be obtained through the internal `ParamCollection` instance:

hamil.getParameterNames()

# This structure provides convenience methods for performing parameter sweeps in multiple dimensions. Substituting parameter values can either be done through the `ParamCollection` instance, or using convenience functions:

# +
# Fabrication parameters
Ca = 60.0 # fF/um^2
Jc = 3.0  # uA/um^2
Aj = 0.2*1.2#0.4**2 # um^2

# Parameter initial values
hamil.setParameterValues(
    "L", 570.0,
    "I", Jc*Aj,
    "C", Ca*Aj,
    "phiZ", 0.5
)
# -

# Now we can generate the Hamiltonian using the substituted values:

H = hamil.getHamiltonian()
H

# We can then diagonalise it using `qutip`'s diagonalizer for just the eigenvalues to check the value of the gap at the chosen value of `phi10e`:

E = H.eigenenergies()
E[1]-E[0]

# This value is quoted in GHz, which is the default energy unit. We will see later how to change units.
#
# Now lets perform a sweep over the flux bias and plot the five first energy gaps. Parameter sweeping can be done with multiple parameters simultaneously using the `sweepSpec` method derived from the `ParamCollection` class. Here the default configuration of the diagonaliser and parameter sweeper is used. By default a dense matrix diagonaliser is used, and the Hamiltonian is diagonalised for the first five eigenvalues only.

hamil.newSweep()
hamil.addSweep('phiZ', -1.0, 1.0, 101)
sweep = hamil.paramSweep(timesweep=True)

# The sweep is then retrieved using the `getSweep` method, which can be used for disentangling results and plotting sweeps along different dimensions of the sweep, as well as extracting results of functions other than `getHamiltonian`. Here there was only one dimensions, and we can look at the first 5 levels and take their difference with the ground level:

x,sweep_nosh,v = hamil.getSweep(sweep,'phiZ',{})
for i in range(5):
    y = sweep_nosh[i] - sweep_nosh[0]
    plt.plot(x,y)
plt.xlabel("$\\Phi_{Z}$ ($\\Phi_0$)")
plt.ylabel("$E_{g,i}$ (GHz)")

# Now say we wish to look at the flux dependence of the minimum gap for a few values of inductance, we can perform these sweeps in a compact way using the same method as above, however we now need to retrieve the results of the sweep for each value of inductance:

hamil.newSweep()
hamil.addSweep('phiZ', 0.45, 0.55, 101)
hamil.addSweep('L', 400, 600, 3)
sweep = hamil.paramSweep(timesweep=True)

x1,sweep1,v1 = hamil.getSweep(sweep,'phiZ',{"L":400.0})
x2,sweep2,v2 = hamil.getSweep(sweep,'phiZ',{"L":500.0})
x3,sweep3,v3 = hamil.getSweep(sweep,'phiZ',{"L":600.0})
plt.plot(x1,sweep1[1]-sweep1[0],label="$L=%.1f$"%v1["L"])
plt.plot(x2,sweep2[1]-sweep2[0],label="$L=%.1f$"%v2["L"])
plt.plot(x3,sweep3[1]-sweep3[0],label="$L=%.1f$"%v3["L"])
plt.xlabel("$\\Phi_{Z}$ ($\\Phi_0$)")
plt.ylabel("$E_{g,0}$ (GHz)")
plt.legend()

# We see that decreasing the inductance increases the minimum gap significantly but also reduces the rate of change of the energy near half-flux. We could now look at the dependence of the minimum gap against the inductance in more detail:

hamil.setParameterValue('phiZ',0.5)
hamil.newSweep()
hamil.addSweep('L', 400, 1000, 101)
sweep = hamil.paramSweep(timesweep=True)

x,sweep_L,v = hamil.getSweep(sweep,'L',{})
plt.plot(x,sweep_L[1]-sweep_L[0])
plt.xlabel("$L$ (pH)")
plt.ylabel("$E_{g,0}$ (GHz)")

# We see that beyond a certain inductance for the given parameters, the minimum gap closes to negligible value. To see why this happens we can analyse the potential of the system at half-flux:

# +
hamil.setParameterValue("L", 500.0)
potential, inputs = hamil.getClassicalPotentialFunction()
V1 = potential({'phi1': x, 'phiZ': 0.0})

hamil.setParameterValue("L", 1000.0)
potential, inputs = hamil.getClassicalPotentialFunction()
V2 = potential({'phi1': x, 'phiZ': 0.0})

plt.plot(x, V1, label="$L=%.1f$"%500)
plt.plot(x, V2, label="$L=%.1f$"%1000)
plt.legend()
# -

# Indeed we see that when the inductance is large the Josephson energy influences the potential much more and forms a deeper double well, reducing the tunnel rate.

# ## Modern CSFQ Circuit (Double Well Regime)
#
# Now we will replace the inductor with two Josephson junctions and analyse the circuit in a similar way as above.

graph = CircuitGraph()
graph.addBranch(0, 1, "I1")
graph.addBranch(0, 1, "C1")
graph.addBranch(1, 2, "I2")
graph.addBranch(1, 2, "C2")
graph.addBranch(1, 2, "Csh")
graph.addBranch(0, 2, "I3")
graph.addBranch(0, 2, "C3")
graph.addFluxBias("I2", "Z")
graph.drawGraphViz()

circuit = SymbolicSystem(graph)
circuit.getQuantumHamiltonian()

circuit.getCapacitanceMatrix()

circuit.getCapacitanceMatrix(False)

# Now that there are two degrees of freedom, the Hamiltonian has become more complicated and will be more difficult to simulate. We can keep simulation time reasonable by keeping the operator truncations only large enough that the low energy bands converge:

hamil = NumericalSystem(circuit)
hamil.configureOperator(1,10,"charge")
hamil.configureOperator(2,10,"charge")
hamil.getHilbertSpaceSize()

# We now have a new set of parameters:

hamil.getParameterNames()

# Lets now look at the spectrum as function of external flux:

# +
# Fabrication parameters
Ca = 60.0 # fF/um^2
Jc = 3.0  # uA/um^2
Aj=0.3**2
hamil.setParameterValues(
    'C1',Ca*Aj,
    'C2',Ca*Aj,
    'C3',Ca*Aj,
    'I1',Jc*Aj,
    'I2',Jc*Aj,
    'I3',Jc*Aj,
    'Csh',45,
    'phiZ',0.5
)

hamil.newSweep()
hamil.addSweep('phiZ', 0.0, 1.0, 101)
sweep = hamil.paramSweep(timesweep=True)
# -

x,sweep,v = hamil.getSweep(sweep, 'phiZ', {})
for i in range(5):
    y = sweep[i] - sweep[0]
    plt.plot(x,y)
plt.xlabel("$\\Phi_{Z}$ ($\\Phi_0$)")
plt.ylabel("$E_{g,i}$ (GHz)")

# We see here that the anharmonicity of the energy levels near half-flux is very high. The rate of change of the energy gap against flux is also very high in this region, which is unfavourable in the presence of flux noise. This is mitigated to a great extent by the replacement of the inductor with these Josephson junctions, which mean that the loop area of the qubit loop is much smaller and hence couples to fewer neighbouring fluctuators.
#
# We can now look at the classical potential:

circuit.getClassicalHamiltonian()

potential, inputs = hamil.getClassicalPotentialFunction()
inputs

p1 = np.linspace(-1.0, 1.0, 201)
p2 = np.linspace(-1.0, 1.0, 201)
p1g, p2g = np.meshgrid(p1, p2)
inputs['phi1'] = p1g
inputs['phi2'] = p2g

inputs['phiZ'] = 0.5
V1 = potential(inputs)
inputs['phiZ'] = 0.0
V2 = potential(inputs)

# +
fig, ax = plt.subplots(ncols=2,nrows=1,constrained_layout=True,figsize=(13,5))
ax1 = ax[0]
ax2 = ax[1]
ax1.set_title("$\\Phi_{Z} = 0$")
mesh = ax1.pcolormesh(p1g,p2g,V2,cmap='rainbow')
ax1.set_xlabel('$\\phi_1$ ($\\Phi_0$)')
ax1.set_ylabel('$\\phi_2$ ($\\Phi_0$)')

ax2.set_title("$\\Phi_{Z} = 0.5\\Phi_0$")
mesh = ax2.pcolormesh(p1g,p2g,V1,cmap='rainbow')
cbar = fig.colorbar(mesh)
cbar.set_label('$V(\\phi_1,\\phi_2)$ (GHz)',rotation=-90,labelpad=12)
ax2.set_xlabel('$\\phi_1$ ($\\Phi_0$)')
ax2.set_ylabel('$\\phi_2$ ($\\Phi_0$)')
plt.show()
# -

# ### CSFQ in Single Well Regime
#
# We can modulate the potential above to increase the tunneling rate and reduce sensitivity to flux noise by introducing a parameter $\alpha$ that scales the size of junction $I_2$:

# +
# Fabrication parameters
Ca = 60.0 # fF/um^2
Jc = 3.0  # uA/um^2
Aj=0.3**2
alpha = 0.44
hamil.setParameterValues(
    'C1',Ca*Aj,
    'C2',Ca*Aj*alpha,
    'C3',Ca*Aj,
    'I1',Jc*Aj,
    'I2',Jc*Aj*alpha,
    'I3',Jc*Aj,
    'Csh',45,
    'phiZ',0.5
)

hamil.newSweep()
hamil.addSweep('phiZ', 0.0, 1.0, 101)
sweep = hamil.paramSweep(timesweep=True)
# -

x,sweep,v = hamil.getSweep(sweep,'phiZ',{})
for i in range(5):
    y = sweep[i] - sweep[0]
    plt.plot(x,y)
plt.xlabel("$\\Phi_{Z}$ ($\\Phi_0$)")
plt.ylabel("$E_{g,i}$ (GHz)")

# Indeed we see that making one of the junctions smaller significantly increases the tunnel rate. We can compare the potential in this case:

potential, inputs = hamil.getClassicalPotentialFunction()
inputs['phiZ'] = 0.5
inputs['phi1'] = p1g
inputs['phi2'] = p2g

V1b = potential(inputs)

V1b

# +
fig, ax = plt.subplots(ncols=2,nrows=1,constrained_layout=True,figsize=(13,5))
ax1 = ax[0]
ax2 = ax[1]
ax1.set_title("$\\alpha=1$")
mesh = ax1.pcolormesh(p1g,p2g,V1,cmap='rainbow')
#cbar = fig.colorbar(mesh)
#cbar.set_label('$V(\\phi_1,\\phi_2)$ (GHz)',rotation=-90,labelpad=12)
ax1.set_xlabel('$\\phi_1$ ($\\Phi_0$)')
ax1.set_ylabel('$\\phi_2$ ($\\Phi_0$)')

ax2.set_title("$\\alpha=0.44$")
mesh = ax2.pcolormesh(p1g,p2g,V1b,cmap='rainbow')
cbar = fig.colorbar(mesh)
cbar.set_label('$V(\\phi_1,\\phi_2)$ (GHz)',rotation=-90,labelpad=12)
ax2.set_xlabel('$\\phi_1$ ($\\Phi_0$)')
ax2.set_ylabel('$\\phi_2$ ($\\Phi_0$)')
plt.show()
# -

# We see that barrier between neighbouring cells has been reduced as expected. We can look at the value of the minimum gap as a function of the alpha parameter, but to do so we should parameterise the Hamiltonian.

# ### Using New Parameters
#
# We can parametrise the circuit Hamiltonian in terms of new parameters, for instance the parameter $\alpha$ above would be useful to sweep over directly as it affects both the Josephson energy and the capacitance of the junction.

circuit.getSymbolList()

# +
# Create new symbols
hamil.getSymbolicSystem().getSymbols(
    "alpha", "C2'", "I2'",
    symbol_overrides=[sy.symbols(r"\alpha"), sy.symbols(r"C'_{2}"), sy.symbols(r"I'_{2}")]
)

# Get existing symbols
symbols = hamil.getSymbolicSystem().getSymbolList()

# Add the parameter
hamil.getSymbolicSystem().addParameterisation("C2", symbols['alpha']*symbols["C2'"])
hamil.getSymbolicSystem().addParameterisation("I2", symbols['alpha']*symbols["I2'"])

# Show the updated parameter list and inverse capacitance matrix
print(hamil.getParameterNames())
# -

hamil.getSymbolicSystem().drawParameterisationGraph()

hamil.getSymbolicSystem().getCapacitanceMatrix(False)

# +
Aj=0.3**2
hamil.setParameterValues(
    'C1',Ca*Aj,
    'C2\'',Ca*Aj,
    'C3',Ca*Aj,
    'I1',Jc*Aj,
    'I2\'',Jc*Aj,
    'I3',Jc*Aj,
    'Csh',45,
    'phiZ',0.5,
    'alpha',0.44
)

hamil.newSweep()
hamil.addSweep('alpha', 0.3, 1.0, 101)
sweep = hamil.paramSweep(timesweep=True)
# -

x,sweep,v = hamil.getSweep(sweep, 'alpha', {})
for i in range(5):
    y = sweep[i] - sweep[0]
    plt.plot(x,y)
plt.xlabel("$\\alpha$")
plt.ylabel("$E_{g,i}$ (GHz)")

# We can use this to parameterise the circuit components in terms of fabrication parameters so that the energy dependence on those can be simulated more conveniently:

# +
# Create new symbols
hamil.getSymbolicSystem().getSymbols("Jc", "Ca", "lse", "l1", "l2", "l3")
sym = hamil.getSymbolicSystem().getSymbolList()
lse = sym['lse']
l1 = sym['l1']
l2 = sym['l2']
l3 = sym['l3']
alpha = sym['alpha']
Ca = sym['Ca']
Jc = sym['Jc']

# Replace capacitors
hamil.getSymbolicSystem().addParameterisation("C1", lse*l1*Ca)
hamil.getSymbolicSystem().addParameterisation("C2", alpha*lse*l2*Ca)
hamil.getSymbolicSystem().addParameterisation("C3", lse*l3*Ca)

# Replace JJs
hamil.getSymbolicSystem().addParameterisation("I1", lse*l1*Jc)
hamil.getSymbolicSystem().addParameterisation("I2", alpha*lse*l2*Jc)
hamil.getSymbolicSystem().addParameterisation("I3", lse*l3*Jc)

# Show the parameter graph
hamil.getSymbolicSystem().drawParameterisationGraph()
# -

# For example now we can simulate the dependence of the minimum gap on the critical current density $J_c$:

# +
hamil.setParameterValues(
    'l1', 0.3,
    'l2', 0.3,
    'l3', 0.3,
    'lse', 0.3,
    'Ca', 60,
    'Jc', 3,
    'Csh', 45,
    'phiZ', 0.5,
    'alpha', 0.44
)

hamil.newSweep()
hamil.addSweep('Jc', 1, 4, 101)
sweep = hamil.paramSweep(timesweep=True)
# -

x,sweep,v = hamil.getSweep(sweep, 'Jc', {})
for i in range(5):
    y = sweep[i] - sweep[0]
    plt.plot(x,y)
plt.xlabel("$J_c$ ($\\mathrm{\mu A.\mu m^{-2}}$)")
plt.ylabel("$E_{g,i}$ (GHz)")

# ## Floating CSFQ Variants
#
# So far we have looked at CSFQ circuits where the main qubit loop is grounded. Now we add a degree of freedom by placing capacitors between ground and the qubit circuit.

graph = CircuitGraph()
graph.addBranch(0, 1, "Cgnd1")
graph.addBranch(0, 2, "Cgnd2")
graph.addBranch(0, 3, "Cgnd3")
graph.addBranch(1, 2, "C1")
graph.addBranch(1, 2, "I1")
graph.addBranch(2, 3, "C2")
graph.addBranch(2, 3, "I2")
graph.addBranch(2, 3, "Csh")
graph.addBranch(1, 3, "C3")
graph.addBranch(1, 3, "I3")
graph.addFluxBias("I1", "Z")
graph.drawGraphViz()

circuit = SymbolicSystem(graph)
circuit.getQuantumHamiltonian()

hamil = NumericalSystem(circuit)
hamil.configureOperator(1, 7, "charge")
hamil.configureOperator(2, 7, "charge")
hamil.configureOperator(3, 7, "charge")
hamil.prepareOperators()
hamil.getHilbertSpaceSize()

# We see now that the Hilbert space is becoming very large. To potentially improve the diagonalisation time we will make use of sparse matrices. First lets set some parameters:

hamil.getParameterNames()

# Now we can look at the sparsity of the Hamiltonian and check what order of magnitude the eigenvectors come out so that we can configure the sparse matrix diagonaliser correctly:

# Fabrication parameters
Ca = 60.0 # fF/um^2
Jc = 3.0  # uA/um^2
Aj = 0.3**2
alpha = 0.44
hamil.setParameterValues(
    'C1', Ca*Aj,
    'C2', Ca*Aj*alpha,
    'C3', Ca*Aj,
    'I1', Jc*Aj,
    'I2', Jc*Aj*alpha,
    'I3', Jc*Aj,
    'Csh', 45,
    'Cgnd1', 2,
    'Cgnd2', 2,
    'Cgnd3', 2,
    'phiZ', 0.5
)
H = hamil.getHamiltonian()
print ("Hamiltonian Sparsity: %f" % hamil.sparsity(H))

# Clearly the Hamiltonian is very sparse in this case, so we can expect better scaling in a larger Hilbert space. Now let's look at some of the eigenvalues:

# At half-flux
H.eigenenergies()[0]

# At zero-flux, where we expect to find the other extremum
hamil.setParameterValues('phiZ', 0.0)
H = hamil.getHamiltonian()
H.eigenenergies()[0]

# So with this, we know we should set the shift value of the sparse solver $\sigma$ on the order of $-300$ to ensure the lowest eigenvalues are found. Let's compare the time to solution for both the dense and sparse solvers for this large Hilbert space:

hamil.newSweep()
hamil.addSweep('phiZ', 0.45, 0.55, 21)
sweep = hamil.paramSweep(timesweep=True)

x,sweep1,v = hamil.getSweep(sweep,'phiZ',{})
for i in range(5):
    y = sweep1[i] - sweep1[0]
    plt.plot(x,y)
plt.xlabel("$\\Phi_{Z}$ ($\\Phi_0$)")
plt.ylabel("$E_{g,i}$ (GHz)")

# Indeed the solving takes quite a long time, let's see if sparse matrices improve this speed. To configure the sparse solver we use the `setDiagConfig` function:

# +
hamil.newSweep()
hamil.addSweep('phiZ', 0.0, 1.0, 101)

# Configure diagonalizer
opts = {
    "sigma":-300,
    "mode":"normal",
    "maxiter":None,
    "tol":0
}
hamil.setDiagConfig(sparse=True, sparsesolveropts=opts)

sweep = hamil.paramSweep(timesweep=True)
# -

x,sweep1,v = hamil.getSweep(sweep,'phiZ',{})
for i in range(5):
    y = sweep1[i] - sweep1[0]
    plt.plot(x,y)
plt.xlabel("$\\Phi_{Z}$ ($\\Phi_0$)")
plt.ylabel("$E_{g,i}$ (GHz)")

# We see here that using a sparse matrix solver drastically improved the time to solution for this specific model. For best performance a $\sigma$ value very close to the lowest eigenvalue should be used. If the spectra have a predictable parameter dependence, such as monotonically increasing or decreasing, it should be relatively straight-forward to adjust the $\sigma$ value for every parameter value to get the best performance possible.
#
# Now let's increase the Hilbert space size and see if the original truncation was adequate:

hamil.configureOperator(1, 10, "charge")
hamil.configureOperator(2, 10, "charge")
hamil.configureOperator(3, 10, "charge")
hamil.getHilbertSpaceSize()

# +
hamil.newSweep()
hamil.addSweep('phiZ', 0.0, 1.0, 101)

# Configure diagonalizer
opts = {
    "sigma":-300,
    "mode":"normal",
    "maxiter":None,
    "tol":0
}
hamil.setDiagConfig(sparse=True, sparsesolveropts=opts)

sweep = hamil.paramSweep(timesweep=True)
# -

x,sweep1,v = hamil.getSweep(sweep,'phiZ',{})
for i in range(5):
    y = sweep1[i] - sweep1[0]
    plt.plot(x,y)
plt.xlabel("$\\Phi_{Z}$ ($\\Phi_0$)")
plt.ylabel("$E_{g,i}$ (GHz)")

# We see that the lowest energy levels have indeed converged, thus we can use a lower truncation.

hamil.setDiagConfig(sparse=False)

# ## Tunable CSFQ
#
# We now look at making one of the junctions of the CSFQ tunable by turning it into a DC-SQUID. If the inductance of the SQUID loop is small and the two Josephson junction energies are equal, we can use a parameterisation instead of including the extra junctions in the model.
#
# ### Tuneable Traditional CSFQ with Parameterisation

graph = CircuitGraph()
graph.addBranch(0, 1, "L")
graph.addBranch(0, 1, "I")
graph.addBranch(0, 1, "C")
graph.addBranch(0, 1, "Csh")
graph.addFluxBias("I", "Z")
graph.drawGraphViz()

circuit = SymbolicSystem(graph)
circuit.getQuantumHamiltonian()

hamil = NumericalSystem(circuit)
hamil.configureOperator(1, 20, "charge")
hamil.getParameterNames()

# +
# Create new symbols
hamil.getSymbolicSystem().getSymbols("phiX", "I'", symbol_overrides=[sy.symbols(r"\phi_x"), sy.symbols(r"I'")])

# Add the parameter
symbols = hamil.getSymbolicSystem().getSymbolList()
hamil.getSymbolicSystem().addParameterisation("I", sy.cos(sy.pi*symbols['phiX']) * symbols["I'"])

# +
# Fabrication parameters
Ca = 60.0 # fF/um^2
Jc = 3.0  # uA/um^2
Aj = 0.2*1.2#0.4**2 # um^2
hamil.setParameterValues(
    'C', 2*Ca*Aj,
    "I'", 2*Jc*Aj,
    'Csh', 45,
    'L', 570.0,
    'phiZ', 0.5,
    'phiX', 0.0
)

hamil.newSweep()
hamil.addSweep('phiX', -1.0, 1.0, 101)
sweep = hamil.paramSweep(timesweep=True)
# -

x,sweep1,v = hamil.getSweep(sweep,'phiX',{})
y = sweep1[1] - sweep1[0]
plt.plot(x,y)
plt.xlabel("$\\Phi_{X}$ ($\\Phi_0$)")
plt.ylabel("$E_{g,i}$ (GHz)")

# Now let's look at the energy gap landscape:

hamil.newSweep()
hamil.addSweep('phiX', -1.0, 1.0, 101)
hamil.addSweep('phiZ', 0, 1.0, 101)
sweep = hamil.paramSweep(timesweep=True)

# Here we get the full two dimensional sweep:

xy, sweep2d, s = hamil.getSweep(sweep, ['phiX','phiZ'], {})

# Take the difference between the first excited state and ground state at all flux values:

gap = sweep2d[1,:,:] - sweep2d[0,:,:]

# +
fig, ax1 = plt.subplots(ncols=1,nrows=1,constrained_layout=True,figsize=(7,6))
ax1.set_title("Energy Gap $E_g$")
mesh = ax1.pcolormesh(xy[0], xy[1], gap, cmap='rainbow')
ax1.set_xlabel('$\\phi_X$ ($\\Phi_0$)')
ax1.set_ylabel('$\\phi_{Z}$ ($\\Phi_0$)')

cbar = fig.colorbar(mesh)
cbar.set_label('$E_g(\\phi_X,\\phi_{Z})$ (GHz)',rotation=-90,labelpad=20)
plt.show()
# -

# ### Tuneable 3JJ CSFQ using Parameterisation

graph = CircuitGraph()
graph.addBranch(0, 1, "C1")
graph.addBranch(0, 1, "I1")
graph.addBranch(1, 2, "C2")
graph.addBranch(1, 2, "I2")
graph.addBranch(1, 2, "Csh")
graph.addBranch(0, 2, "C3")
graph.addBranch(0, 2, "I3")
graph.addFluxBias("I1", "Z")
graph.drawGraphViz()

circuit = SymbolicSystem(graph)
circuit.getQuantumHamiltonian()

hamil = NumericalSystem(circuit)
hamil.configureOperator(1, 10, "charge")
hamil.configureOperator(2, 10, "charge")
hamil.getHilbertSpaceSize()

# First we assume that the junctions are equal as for the traditional CSFQ circuit, and we add the $\alpha$ parameter. This is valid for equal junctions due to the fact that 
#
# $$\hat{D}_a^\dagger \hat{D}_b = \hat{D}_b \hat{D}_a^\dagger$$
#
# which leads to the $e^{\pm i \phi_e}$ terms simplifying to $\cos\left( \phi_e \right)$.

# +
# Create new symbols
hamil.getSymbolicSystem().getSymbols(
    "phiX", "I2'", "C2'", "a",
    symbol_overrides=[
        sy.symbols(r"\phi_X"), sy.symbols(r"I'_2"), sy.symbols(r"C'_2"), sy.symbols(r"\alpha")
    ]
)

# Add the parameter
symbols = hamil.getSymbolicSystem().getSymbolList()
hamil.getSymbolicSystem().addParameterisation("C2", symbols["a"] * symbols["C2'"])
hamil.getSymbolicSystem().addParameterisation("I2", symbols["a"] * sy.cos(sy.pi * symbols["phiX"]) * symbols["I2'"])

# +
Ca = 60.0 # fF/um^2
Jc = 3.0  # uA/um^2
Aj = 0.3**2
hamil.setParameterValues(
    'C1', Ca*Aj,
    "C2'", Ca*Aj,
    'C3', Ca*Aj,
    'I1', Jc*Aj,
    "I2'", Jc*Aj,
    'I3', Jc*Aj,
    'Csh', 45,
    'phiZ', 0.5,
    'a', 1.0,
    'phiX', 0.0
)

hamil.newSweep()
hamil.addSweep('a', 0.4, 1.0, 7),
hamil.addSweep('phiX', -1.0, 1.0, 101)
sweep = hamil.paramSweep(timesweep=True)
# -

x, sweep1, v1 = hamil.getSweep(sweep, 'phiX', {'a': 0.7})
x, sweep2, v2 = hamil.getSweep(sweep, 'phiX', {'a': 1.0})
y1 = sweep1[1] - sweep1[0]
y2 = sweep2[1] - sweep2[0]
plt.plot(x,y1,label="$\\alpha$ = $%.1f$" % (v1['a']))
plt.plot(x,y2,label="$\\alpha$ = $%.1f$" % (v2['a']))
plt.xlabel("$\\Phi_{x}$ ($\\Phi_0$)")
plt.ylabel("$E_{g,i}$ (GHz)")
plt.legend()

# ### Tuneable 3JJ CSFQ using parallel JJs
#
# Now we look at adding a branch with an additional JJ to define a second superconducting loop. This way it is possible to account for asymmetry in the junctions, as in this case it is no longer possible to simplify the bias terms.

graph = CircuitGraph()
graph.addBranch(0, 1, "C1")
graph.addBranch(0, 1, "I1")
graph.addBranch(1, 2, "C2a")
graph.addBranch(1, 2, "I2a")
graph.addBranch(1, 2, "C2b")
graph.addBranch(1, 2, "I2b")
graph.addBranch(1, 2, "Csh")
graph.addBranch(0, 2, "C3")
graph.addBranch(0, 2, "I3")
graph.addFluxBias("I3", "Z")
graph.addFluxBias("I2a", "Xa")
graph.addFluxBias("I2b", "Xb")
graph.drawGraphViz()

circuit = SymbolicSystem(graph)
circuit.getQuantumHamiltonian()

hamil = NumericalSystem(circuit)
hamil.configureOperator(1, 10, "charge")
hamil.configureOperator(2, 10, "charge")
hamil.getHilbertSpaceSize()

# Lets first look at the symmetric JJ case and compare with the parameterisation method:

# Fabrication parameters
Ca = 60.0 # fF/um^2
Jc = 3.0  # uA/um^2
Aj=0.3**2
hamil.setParameterValues(
    'C1', Ca*Aj,
    'C2a', 0.5*Ca*Aj,
    'C2b', 0.5*Ca*Aj,
    'C3', Ca*Aj,
    'I1', Jc*Aj,
    'I2a', 0.5*Jc*Aj,
    'I2b', 0.5*Jc*Aj,
    'I3', Jc*Aj,
    'Csh', 45,
    'phiZ', 0.5,
    'phiXa', 0.0,
    'phiXb', 0.0
)

# +
# Create new symbols
hamil.getSymbolicSystem().getSymbols(
    "phiX",
    symbol_overrides=[
        sy.symbols(r"\phi_X")
    ]
)

# Add the parameter
symbols = hamil.getSymbolicSystem().getSymbolList()
hamil.getSymbolicSystem().addParameterisation("phiXa", 0.5 * symbols["phiX"])
hamil.getSymbolicSystem().addParameterisation("phiXb", -0.5 * symbols["phiX"])
# -

hamil.setParameterValues(
    'C1', Ca*Aj,
    'C2a', 0.5*Ca*Aj,
    'C2b', 0.5*Ca*Aj,
    'C3', Ca*Aj,
    'I1', Jc*Aj,
    'I2a', 0.5*Jc*Aj,
    'I2b', 0.5*Jc*Aj,
    'I3', Jc*Aj,
    'Csh', 45,
    'phiZ', 0.5,
    'phiX', 0.0
)

hamil.newSweep()
hamil.addSweep('phiX', -1.0, 1.0, 101)
sweep = hamil.paramSweep(timesweep=True)

x,sweep1,v = hamil.getSweep(sweep, 'phiX', {})
y = sweep1[1] - sweep1[0]
plt.plot(x, y)
plt.plot(x, y2, "x")
plt.xlabel("$\\Phi_{x}$ ($\\Phi_0$)")
plt.ylabel("$E_{g,i}$ (GHz)")

# We see that the results are almost identical. Now we look at making the junctions asymmetric using parameterisations:

# +
# Introduce the asymmetry parameter
hamil.getSymbolicSystem().getSymbols(
    "d", "I2a'", "C2a'", "I2b'", "C2b'",
    symbol_overrides=[
        sy.symbols(r"d"),
        sy.symbols(r"I'_{a2}"),
        sy.symbols(r"C'_{a2}"),
        sy.symbols(r"I'_{b2}"),
        sy.symbols(r"C'_{b2}")
    ]
)

# Add the parameter
symbols = hamil.getSymbolicSystem().getSymbolList()
d = symbols["d"]
hamil.getSymbolicSystem().addParameterisation("C2a", (1+d) * symbols["C2a'"])
hamil.getSymbolicSystem().addParameterisation("I2a", (1+d) * symbols["I2a'"])
hamil.getSymbolicSystem().addParameterisation("C2b", (1-d) * symbols["C2b'"])
hamil.getSymbolicSystem().addParameterisation("I2b", (1-d) * symbols["I2b'"])
# -

hamil.setParameterValues(
    'C1', Ca*Aj,
    "C2a'", 0.5*Ca*Aj,
    "C2b'", 0.5*Ca*Aj,
    'C3', Ca*Aj,
    'I1', Jc*Aj,
    "I2a'", 0.5*Jc*Aj,
    "I2b'", 0.5*Jc*Aj,
    'I3', Jc*Aj,
    'Csh', 45,
    'phiZ', 0.5,
    'phiX', 0.0,
    "d", 0.0
)

hamil.newSweep()
hamil.addSweep('d', 0.0, 0.1, 3)
hamil.addSweep('phiX', -1.0, 1.0, 101)
sweep = hamil.paramSweep(timesweep=True)

x,sweep1,v1 = hamil.getSweep(sweep, 'phiX', {'d': 0.0})
x,sweep2,v2 = hamil.getSweep(sweep, 'phiX', {'d': 0.04})
x,sweep3,v3 = hamil.getSweep(sweep, 'phiX', {'d': 0.1})
y1 = sweep1[1] - sweep1[0]
y2 = sweep2[1] - sweep2[0]
y3 = sweep3[1] - sweep3[0]
plt.plot(x,y1,label="$d$ = $%.2f$" % (v1['d']))
plt.plot(x,y2,label="$d$ = $%.2f$" % (v2['d']))
plt.plot(x,y3,label="$d$ = $%.2f$" % (v3['d']))
plt.xlabel("$\\Phi_{x}$ ($\\Phi_0$)")
plt.ylabel("$E_{g,i}$ (GHz)")
plt.legend()

# We can also observe that the asymmetry causes a shift in the location of the minimum gap, away from half flux:

hamil.setParameterValue('phiX',0.4)
hamil.newSweep()
hamil.addSweep('d', 0.0, 0.1, 3)
hamil.addSweep('phiZ', 0.4, 0.6, 21)
sweep = hamil.paramSweep(timesweep=True)

x,sweep1,v1 = hamil.getSweep(sweep, 'phiZ', {'d': 0.0})
x,sweep2,v2 = hamil.getSweep(sweep, 'phiZ', {'d': 0.04})
x,sweep3,v3 = hamil.getSweep(sweep, 'phiZ', {'d': 0.1})
y1 = sweep1[1] - sweep1[0]
y2 = sweep2[1] - sweep2[0]
y3 = sweep3[1] - sweep3[0]
plt.plot(x,y1,label="$d$ = $%.2f$" % (v1['d']))
plt.plot(x,y2,label="$d$ = $%.2f$" % (v2['d']))
plt.plot(x,y3,label="$d$ = $%.2f$" % (v3['d']))
plt.xlabel("$\\Phi_{z}$ ($\\Phi_0$)")
plt.ylabel("$E_{g,i}$ (GHz)")
plt.legend()



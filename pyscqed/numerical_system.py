import qutip as qt
import numpy as np
import sympy as sy
import sympy.utilities as syu
import scipy as sc
import networkx as nx
import progress.bar
import time

from .dataspec import TempData
from .symbolic_system import SymbolicSystem
from .sweeping import SweepConfig, SweepResult, SweepResultFromDisk, SweepResultFromMemory
from .evaluation_graph import EvaluationGraph
from .result import EigenvalueResult, EigenvectorResult, FunctionResult
from .units import Units
from . import physical_constants as pc
from . import util
from . import units

# Local override for the Qobj tolerance. The global settings appear to not work
_qobj_atol = 1e-12


class HamiltonianSpectrum(EvaluationGraph):
    def __init__(self, nsys: "NumericalSystem"):
        super().__init__()
        self.addNode("Hamiltonian", fn=nsys.getHamiltonian, outputs=["qobj"])
        self.addNode("Spectrum", fn=nsys.diagonalize, outputs=["E"])
        self.addDependency("Hamiltonian", "Spectrum")


class SingleResonatorInteraction(EvaluationGraph):
    def __init__(self, nsys: "NumericalSystem"):
        super().__init__()
        self.addNode("Hamiltonian", fn=nsys.getHamiltonian, outputs=["qobj"])
        self.addNode("Spectrum", fn=nsys.diagonalize, outputs=["energies", "vectors"])
        self.addNode("Resonator", fn=nsys.getResonatorResponse, outputs=["rwa_energies"])
        self.addDependency("Hamiltonian", "Spectrum")
        self.addDependency("Spectrum", "Resonator")


class NumericalSystem(TempData):
    
    ## Mode types
    __mode_types = [
        "osc", # DoFs have capacitive and inductive parts only
        "jos", # DoFs have capacitive, inductive and Josephson parts
        "isl", # DoFs have capacitive parts only
        "leg"  # DoFs have inductive parts only
    ]
    
    ## Initialise a Hamiltonian using a circuit specification
    def __init__(self, symbolic_system: SymbolicSystem, unit=Units("CQED1")):
        
        # Initialise the temporary data manager
        super().__init__()
        self.newSession(id(self))
        self.__use_temp = False
        
        # Assign the circuit
        self.SS = symbolic_system
        
        # Nested dictionary for DoF operators, keyed by the relevant mode
        self.circ_operators = {}
        self.operator_data = {}
        
        # Set the unit system
        self.units = unit
        self._set_parameter_units()
        
        # Load default diagonaliser configuration
        self.setDiagConfig()
        
        # Load the symbolic expressions
        self.getSymbolicExpressions()
    
    def getNodeList(self):
        return self.SS.nodes
    
    def getNodeIndex(self, node):
        return self.SS.nodes.index(node)
    
    def getEdgeList(self):
        return self.SS.edges
    
    def getEdgeIndex(self, edge):
        return self.SS.edges.index(edge)
    
    def getCircuitGraph(self):
        return self.SS.CG
    
    def getSymbolicSystem(self):
        return self.SS
    
    ## Get Hilbert space size
    def getHilbertSpaceSize(self):
        """ Returns the Hilbert space size considering all currently defined operator truncations.
        
        :return: The Hilbert space size
        :rtype: int
        """
        ret = 1
        for k, v in self.operator_data.items():
            trunc = v["truncation"]
            basis = v["basis"]
            if basis == "charge":
                ret *= (2*trunc+1)
            elif basis == "oscillator":
                ret *= (trunc)
        return ret
    
    def sparsity(self, op):
        return 1 - op.to("CSR").data.as_scipy().nnz/self.getHilbertSpaceSize()**2
    
    ###################################################################################################################
    #       Operator Generation and Functions
    ###################################################################################################################
        
    def getCommutator(self, Q, P, basis="charge"):
        if basis == "charge": # Need to figure out the correction for this case
            dims = 2*pc.e*pc.phi0/pc.hbar
            return qt.commutator(P, Q)*dims
        elif basis == "oscillator":
            
            # Get highest number operator eigenvalue and associated eigenstate
            enum = Q.shape[0]
            vnum = qt.basis(enum, enum-1)
            
            # Create the correction matrix
            mat = 1 - (enum)*vnum*vnum.dag()
            
            # Invert it
            corrmat = qt.Qobj(np.linalg.inv(mat.data.to_array()))
            
            # Get the correct commutator
            return qt.commutator(P, Q)*corrmat
    
    ## Available basis representations
    __basis_repr = [
        "charge", 
        "flux", 
        "oscillator", 
        "custom"
    ]
    
    def configureOperator(self, node, trunc, basis, fmax=4.0):
        if node not in self.getNodeList():
            raise Exception("Node '%i' is not a valid circuit node." % node)
        
        # Get the operator circuit-dependent parameters
        impedance = None
        frequency = None
        flux_max = None
        if basis == "oscillator":
            index = self.getNodeIndex(node)
            frequency = sy.sqrt(self.Linv[index, index]*self.Cinv[index, index])
            freq = "fosc%i" % node
            self.SS.addParameter(freq)
            self.SS.addParameterisation(freq, frequency)
            impedance = sy.sqrt(self.Cinv[index, index]/self.Linv[index, index])
            impe = "Zosc%i" % node
            self.SS.addParameter(impe)
            self.SS.addParameterisation(impe, impedance)
        elif basis == "discretized_flux":
            flux_max = fmax
        
        self.operator_data[node] = {
            "truncation": trunc, 
            "basis": basis, 
            "impedance": impedance, 
            "frequency": frequency,
            "flux_max": flux_max
        }
    
    def getOperatorList(self, node):
        Q = None
        P = None
        D = None
        Ddag = None
        S = None
        Sdag = None
        basis = self.operator_data[node]["basis"]
        # FIXME: Determine if we need to generate all the operators for this node
        
        if basis == "charge":
            return self._get_charge_basis(node)
        elif basis == "oscillator":
            return self._get_oscillator_basis(node)
        elif basis == "flux":
            return self._get_flux_basis(node)
        elif basis == "discretized_flux":
            # FIXME: This doesn't work properly yet
            return self._get_discretized_flux_basis(node)
        else:
            raise Exception("Unrecognized basis representation '%s'." % repr(basis))
    
    ## Expand operator Hilbert spaces and update mapping to associated symbols
    def getExpandedOperatorsMap(self, nodes=None):
        """ Creates all the operators associated with each node in the currently defined circuit. The operators are expanded into the total Hamiltonian Hilbert space.
        
        :return: None
        """
        
        # Get the pos list for indexing the DoFs
        node_list = self.getNodeList()
        
        # Generate the Hilbert space expanders
        Ilist = np.empty([len(node_list)], dtype=np.dtype(qt.Qobj))
        for i, node in enumerate(node_list):
            trunc = self.operator_data[node]["truncation"]
            basis = self.operator_data[node]["basis"]
            if basis == "oscillator":
                Ilist[i] = qt.qeye(trunc)
            elif basis == "charge":
                Ilist[i] = qt.qeye(2*trunc + 1)
        
        # Create mode operators
        Olist = np.empty([len(node_list)], dtype=np.dtype(qt.Qobj))
        op_dict = {}
        for i, node in enumerate(node_list):
            # Ignore nodes that are not in the list, if provided
            if nodes is not None:
                if node not in nodes:
                    continue
            
            op_dict = {}
            trunc = self.operator_data[node]["truncation"]
            basis = self.operator_data[node]["basis"]
            Q, P, D, Ddag, S, Sdag = self.getOperatorList(node)
            
            # Indices minus the current index
            indices = list(range(len(node_list)))
            indices.remove(i)
            
            # Current index is the operator
            Olist[i] = Q
            for j in indices:
                Olist[j] = Ilist[j]
            op_dict["charge"] = qt.tensor(Olist)
            
            Olist[i] = P
            for j in indices:
                Olist[j] = Ilist[j]
            op_dict["flux"] = qt.tensor(Olist)
            
            Olist[i] = D
            for j in indices:
                Olist[j] = Ilist[j]
            op_dict["disp"] = qt.tensor(Olist)
            
            Olist[i] = Ddag
            for j in indices:
                Olist[j] = Ilist[j]
            op_dict["disp_adj"] = qt.tensor(Olist)
            
            Olist[i] = S
            for j in indices:
                Olist[j] = Ilist[j]
            op_dict["pdisp"] = qt.tensor(Olist)
            
            Olist[i] = Sdag
            for j in indices:
                Olist[j] = Ilist[j]
            op_dict["pdisp_adj"] = qt.tensor(Olist)
            
            self.circ_operators[node] = op_dict.copy()
    
    ###################################################################################################################
    #       Hamiltonian Building Functions
    ###################################################################################################################
    
    def getChargeOpVector(self):
        pos_ops = {k: v["charge"] for k, v in self.circ_operators.items()}
        arr = []
        for node in self.getNodeList():
            arr.append(pos_ops[node])
        self.Qnp = self._init_qobj_vector(arr, dtype=object)
    
    def getFluxOpVector(self):
        pos_ops = {k: v["flux"] for k, v in self.circ_operators.items()}
        arr = []
        for node in self.getNodeList():
            arr.append(pos_ops[node])
        self.Pnp = self._init_qobj_vector(arr, dtype=object)
    
    def getBasisPrefactors(self):
        Zpref = sy.eye(self.SS.Nn)
        for i, node in enumerate(self.SS.nodes):
            if self.operator_data[node]["basis"] == "oscillator":
                Zpref[i, i] = self.operator_data[node]["impedance"]
        return Zpref
    
    def getSymbolicExpressions(self):
        
        # Generate final symbolic expressions
        self.Cinv = self.SS.getInverseCapacitanceMatrix()
        self.Linv = self.SS.getInverseInductanceMatrix()

        # Get branch inverse inductance matrix for branch current calculations
        self.Linv_b = self.SS.getInverseInductanceMatrix(mode='branch')
        self.Cinv_n = self.SS.getInverseCapacitanceMatrix()
        
        # Symbolic expressions independent of a coupled subsystem
        self.Jvec = self.SS.getJosephsonVector()
        self.Pvec = self.SS.getPhaseSlipVector()
        self.Qb = self.SS.getChargeBiasVector()
        self.Qbt = self.SS.Rnb*self.SS.getChargeBiasVector()
        self.Pb = self.SS.getFluxBiasVector(mode="branch")
        self.Pbm = self.SS.getFluxBiasMatrix(mode="branch")
        self.Pbi = self.SS.getFluxBiasVectorInd()
        
        # Basis representation prefactors
        #self.Zpref = self.getBasisPrefactors()
    
    def prepareOperators(self):
        self.getExpandedOperatorsMap()
        self.getChargeOpVector()
        self.getFluxOpVector()

    ###################################################################################################################
    #       Analysis Support
    ###################################################################################################################
    def getClassicalPotentialFunction(self):
        """Returns a function that takes the flux circuit degrees of freedom and the flux bias terms as scalars and
        returns the potential energy at those coordinates. It can be used with numpy arrays too."""
        builder = ClassicalPotentialBuilder(self)
        return builder.getPotentialFunction(), builder.getDefaultInputs()

    ###################################################################################################################
    #       Numerical Hamiltonian Generation
    ###################################################################################################################
    def substitute(self):
        
        subs = self.SS.getSymbolValuesDict()
        
        # Substitute circuit parameters
        self.Cinvnp = np.asmatrix(self.Cinv.subs(subs), dtype=np.float64)
        self.Linvnp = np.asmatrix(self.Linv.subs(subs), dtype=np.float64)
        self.Jvecnp = np.asarray(self.Jvec.subs(subs), dtype=np.float64)[:, 0]
        self.Pvecnp = np.asarray(self.Pvec.subs(subs), dtype=np.float64)[:, 0]
        
        # Substitute external biases
        self.Qbnp = np.asmatrix(self.Qb.subs(subs), dtype=np.float64) # x 2e
        self.Qbtnp = np.asmatrix(self.Qbt.subs(subs), dtype=np.float64) # x 2e
        self.Pbsm = np.asmatrix(self.Pbm.subs(subs), dtype=np.float64)
        self.Pbnp = np.asmatrix(self.Pb.subs(subs), dtype=np.float64) # x Phi0
        self.Pbinp = np.asmatrix(self.Pbi.subs(subs), dtype=np.float64)
        
        # Generate exponentiated flux biases
        Pexp1 = []
        Pexp2 = []
        for i in range(self.Pbsm.shape[0]):
            Pexp1.append(np.exp(2j*np.pi*self.Pbsm[i, i]))
            Pexp2.append(np.exp(-2j*np.pi*self.Pbsm[i, i]))
        self.Pexp_pnp = Pexp1
        self.Pexp_mnp = Pexp2
        
        # Generate exponentiated charge biases
        Qexp1 = []
        Qexp2 = []
        for i in range(self.Qbtnp.shape[0]):
            Qexp1.append(np.exp(2j*np.pi*self.Qbtnp[i, 0]))
            Qexp2.append(np.exp(-2j*np.pi*self.Qbtnp[i, 0]))
        self.Qexp_pnp = Qexp1
        self.Qexp_mnp = Qexp2
        
        # Get branch inverse inductance matrix for branch current calculations
        self.Linvnp_b = np.asmatrix(self.Linv_b.subs(subs), dtype=np.float64)
    
    def getLinearPart(self):
        # Get charging energy
        Hq = self.units.getPrefactor("Ec")*0.5*\
        util.mdot((self.Qnp + self.Qbnp).T, self.Cinvnp, self.Qnp + self.Qbnp)[0, 0]
        
        # Get flux energy
        Hf = self.units.getPrefactor("El")*0.5*\
        util.mdot((self.Pnp + self.Pbinp).T, self.Linvnp, self.Pnp + self.Pbinp)[0, 0]
        
        return Hq + Hf
    
    def getStaticJosephsonPart(self):
        # Need the branch DoFs in the possibly transformed representation
        Pp = self.SS.Rnb*self.SS.Rinv*self.SS.node_vector
        
        # Get the Josephson energy
        Hj_l = []
        Hj_r = []
        for i, edge in enumerate(self.SS.edges):
            if self.Jvecnp[i] == 0.0:
                continue
            prod1 = 0.0
            prod2 = 0.0
            if len(Pp[i].atoms()) > 2: # Case where there is sum of elements
                prod1 = 1.0
                prod2 = 1.0
                # Left
                for arg in Pp[i].args:
                    node = self.SS.node_map_rev[arg.args[1]]
                    if arg.args[0] > 0:
                        prod1 *= self.circ_operators[node]["disp"]
                    else:
                        prod1 *= self.circ_operators[node]["disp_adj"]
                
                # Right
                for arg in Pp[i].args:
                    node = self.SS.node_map_rev[arg.args[1]]
                    if arg.args[0] < 0:
                        prod2 *= self.circ_operators[node]["disp"]
                    else:
                        prod2 *= self.circ_operators[node]["disp_adj"]
            else:
                prod1 = 1.0
                prod2 = 1.0
                node = self.SS.node_map_rev[Pp[i].args[1]]
                if Pp[i].args[0] > 0:
                    prod1 *= self.circ_operators[node]["disp"]
                    prod2 *= self.circ_operators[node]["disp_adj"]
                else:
                    prod1 *= self.circ_operators[node]["disp_adj"]
                    prod2 *= self.circ_operators[node]["disp"]
        
            Hj_l.append(-0.5*self.units.getPrefactor("Ej")*self.Jvecnp[i]*prod1)
            Hj_r.append(-0.5*self.units.getPrefactor("Ej")*self.Jvecnp[i]*prod2)
        return Hj_l, Hj_r
    
    ###################################################################################################################
    #       Evaluables
    ###################################################################################################################
    
    def getHamiltonian(self) -> qt.Qobj:
        # Get charging energy
        Hq = self.units.getPrefactor("Ec")*0.5*\
        util.mdot((self.Qnp + self.Qbnp).T, self.Cinvnp, self.Qnp + self.Qbnp)[0, 0]
        
        # Get flux energy
        Hf = self.units.getPrefactor("El")*0.5*\
        util.mdot((self.Pnp + self.Pbinp).T, self.Linvnp, self.Pnp + self.Pbinp)[0, 0]
        
        # Need the branch DoFs in the possibly transformed representation
        Pp = self.SS.Rnb*self.SS.Rinv*self.SS.node_vector
        
        # Get the Josephson energy
        Hj = 0
        for i, edge in enumerate(self.SS.edges):
            if self.Jvecnp[i] == 0.0:
                continue
            
            prod1 = self.Pexp_pnp[i]
            prod2 = self.Pexp_mnp[i]
            if len(Pp[i].atoms()) > 2: # Case where there is sum of elements
                # Left
                for arg in Pp[i].args:
                    node = self.SS.node_map_rev[arg.args[1]]
                    if arg.args[0] > 0:
                        prod1 *= self.circ_operators[node]["disp"]
                    else:
                        prod1 *= self.circ_operators[node]["disp_adj"]
                
                # Right
                for arg in Pp[i].args:
                    node = self.SS.node_map_rev[arg.args[1]]
                    if arg.args[0] < 0:
                        prod2 *= self.circ_operators[node]["disp"]
                    else:
                        prod2 *= self.circ_operators[node]["disp_adj"]
            else:
                node = self.SS.node_map_rev[Pp[i].args[1]]
                if Pp[i].args[0] > 0:
                    prod1 *= self.circ_operators[node]["disp"]
                    prod2 *= self.circ_operators[node]["disp_adj"]
                else:
                    prod1 *= self.circ_operators[node]["disp_adj"]
                    prod2 *= self.circ_operators[node]["disp"]
        
            Hj += -0.5*self.Jvecnp[i]*(prod1 + prod2)
        Hj *= self.units.getPrefactor("Ej")
        
        # Get the Phaseslip energy
        Hp = 0
        for i, edge in enumerate(self.SS.edges):
            if self.Pvecnp[i] == 0.0:
                continue
            
            prod1 = self.Qexp_pnp[i]
            prod2 = self.Qexp_mnp[i]
            if len(Pp[i].atoms()) > 2: # Case where there is sum of elements
                # Left
                for arg in Pp[i].args:
                    node = self.SS.node_map_rev[arg.args[1]]
                    if arg.args[0] > 0:
                        prod1 *= self.circ_operators[node]["pdisp"]
                    else:
                        prod1 *= self.circ_operators[node]["pdisp_adj"]
                
                # Right
                for arg in Pp[i].args:
                    node = self.SS.node_map_rev[arg.args[1]]
                    if arg.args[0] < 0:
                        prod2 *= self.circ_operators[node]["pdisp"]
                    else:
                        prod2 *= self.circ_operators[node]["pdisp_adj"]
            else:
                node = self.SS.node_map_rev[Pp[i].args[1]]
                if Pp[i].args[0] > 0:
                    prod1 *= self.circ_operators[node]["pdisp"]
                    prod2 *= self.circ_operators[node]["pdisp_adj"]
                else:
                    prod1 *= self.circ_operators[node]["pdisp_adj"]
                    prod2 *= self.circ_operators[node]["pdisp"]
        
            Hp += -0.5*self.Pvecnp[i]*(prod1 + prod2)
        Hp *= self.units.getPrefactor("Ep")
        
        # Total Hamiltonian
        return (Hq + Hf + Hj + Hp).tidyup(_qobj_atol)
    
    def getCurrentOperator(self, edge=None) -> qt.Qobj:
        # Check edge
        if edge is None:
            raise Exception("No edge specified for branch current operator.")
        if edge not in self.getEdgeList():
            raise Exception("Edge %s not in the circuit. Check that it is in the right direction." % repr(edge))
        
        # Create the transformed branch vector and get the branch
        Pp = self.SS.Rnb*self.SS.Rinv*self.SS.node_vector
        i = self.getEdgeIndex(edge)
        
        # Inductive edge type should generally be favoured
        if self.getCircuitGraph().isInductiveEdge(edge):
            # Construct the branch flux operators from node representations
            if len(Pp[i].atoms()) > 2:
                sum1 = 0
                for arg in Pp[i].args:
                    node = self.SS.node_map_rev[arg.args[1]]
                    sum1 += float(arg.args[0]) * self.circ_operators[node]["flux"]
            else:
                node = self.SS.node_map_rev[Pp[i].args[1]]
                sum1 = float(Pp[i].args[0]) * self.circ_operators[node]["flux"]
            
            # Take difference of node fluxes of corresponding branch and use *branch* inverse inductance matrix
            return self.units.getPrefactor("IopL") * sum1 * self.Linvnp_b[i, i]
        
        elif self.getCircuitGraph().isJosephsonEdge(edge):
            
            # Get the Josephson operators
            prod1 = self.Pexp_pnp[i]
            prod2 = self.Pexp_mnp[i]
            if len(Pp[i].atoms()) > 2: # Case where there is sum of elements
                # Left
                for arg in Pp[i].args:
                    node = self.SS.node_map_rev[arg.args[1]]
                    if arg.args[0] > 0:
                        prod1 *= self.circ_operators[node]["disp"]
                    else:
                        prod1 *= self.circ_operators[node]["disp_adj"]
                
                # Right
                for arg in Pp[i].args:
                    node = self.SS.node_map_rev[arg.args[1]]
                    if arg.args[0] < 0:
                        prod2 *= self.circ_operators[node]["disp"]
                    else:
                        prod2 *= self.circ_operators[node]["disp_adj"]
            else:
                node = self.SS.node_map_rev[Pp[i].args[1]]
                if Pp[i].args[0] > 0:
                    prod1 *= self.circ_operators[node]["disp"]
                    prod2 *= self.circ_operators[node]["disp_adj"]
                else:
                    prod1 *= self.circ_operators[node]["disp_adj"]
                    prod2 *= self.circ_operators[node]["disp"]
        
            return 0.5j * self.units.getPrefactor("IopJ") * self.Jvecnp[i] * (prod1 - prod2)
        else:
            raise Exception("Edge %s is not current-carrying" % repr(edge))
    
    def getCurrentMatrixElement(self, energies: EigenvalueResult, vectors: EigenvectorResult, edge=None, elements=None) -> float:
        # Get the relevant operator
        Iop = self.getCurrentOperator(edge=edge)
        
        # Check the elements
        if elements is None:
            raise Exception("No elements specified for current matrix elements.")
        result = np.zeros(len(elements), dtype=np.float64)
        for i, indices in enumerate(elements):
            i1, i2 = indices
            result[i] = Iop.matrix_element(vectors.data[i1], vectors.data[i2]).real
        return result
    
    def getVoltageOperator(self, node=None):
        # Check node
        if node is None:
            raise Exception("No node specified for node voltage operator.")
        if node not in self.getNodeList():
            raise Exception("Node %s is not in the circuit." % repr(node))
        i = self.getNodeIndex(node)
        
        # Get charge superoperator including charge offsets
        Q = self.Qnp + self.Qbnp
        
        # Use the inverse capacitance matrix
        return self.units.getPrefactor("Vop") * Q[i, 0] * self.Cinvnp[i, i]
    
    def getVoltageMatrixElement(self, energies: EigenvalueResult, vectors: EigenvectorResult, node=None, elements=None):
        # Get the relevant operator
        Vop = self.getVoltageOperator(node=node)
        
        # Check the elements
        if elements is None:
            raise Exception("No elements specified for voltage matrix elements.")
        result = np.zeros(len(elements), dtype=np.float64)
        for i, indices in enumerate(elements):
            i1, i2 = indices
            result[i] = Vop.matrix_element(vectors.data[i1], vectors.data[i2]).real
        return result
    
    def getChargingEnergies(self, node=None):
        if node is None:
            ret = {}
            for i, pos in enumerate(self.getNodeList()):
                ret[pos] = 0.5 * self.Cinvnp[i, i] * self.units.getPrefactor("Ec")
            return ret
        else:
            i = self.getNodeList().index(node)
            return 0.5 * self.Cinvnp[i, i] * self.units.getPrefactor("Ec")
    
    def getFluxEnergies(self, node=None):
        if node is None:
            ret = {}
            for i, pos in enumerate(self.getNodeList()):
                ret[pos] = 0.5 * self.Linvnp[i, i] * self.units.getPrefactor("El")
            return ret
        else:
            i = self.getNodeList().index(node)
            return 0.5 * self.Linvnp[i, i] * self.units.getPrefactor("El")
    
    def getJosephsonEnergies(self, edge=None):
        if edge is None:
            ret = {}
            for i, edge in enumerate(self.SS.edges):
                ret[edge] = self.Jvecnp[i] * self.units.getPrefactor("Ej")
            return ret
        else:
            i = self.SS.edges.index(edge)
            return self.Jvecnp[i] * self.units.getPrefactor("Ej")
    
    def getPhaseSlipEnergies(self, edge=None):
        if edge is None:
            ret = {}
            for i, edge in enumerate(self.SS.edges):
                ret[edge] = self.Pvecnp[i] * self.units.getPrefactor("Ep")
            return ret
        else:
            i = self.SS.edges.index(edge)
            return self.Pvecnp[i] * self.units.getPrefactor("Ep")
    
    def getResonatorResponse(
        self,
        energies: EigenvalueResult,
        vectors: EigenvectorResult,
        nmax=100,
        cpl_node=None
    ) -> np.array:
        # Save the derived parameter values for each sweep value
        if cpl_node is None:
            resonator_nodes = [node for node, value in self.SS.CG.resonators_cap.items() if value is not None]
            if len(resonator_nodes) == 1:
                cpl_node = resonator_nodes[0]
            else:
                raise RuntimeError("There are either no coupled resonators or too many to guess the node.")
        
        # Get the model parameters
        gC = self.SS.getParameterValue('g%ir' % cpl_node)
        wrl = self.SS.getParameterValue('f%irl' % cpl_node)
        
        # Get the operator associated with selected node
        index = self.getNodeList().index(cpl_node)
        Op = self.Qnp[index, 0] + self.Qbnp[index, 0]
        
        # Get coupling terms
        E = energies.data - energies.data[0]
        V = vectors.data
        tmax = len(E)
        nmax = nmax + 1 + tmax
        norm = Op.matrix_element(V[0], V[1])
        g_list = []
        for i in range(tmax-1):
            g_list.append(Op.matrix_element(V[i], V[i+1]))
        
        # Apply the circuit derived prefactor
        g_list = gC * np.abs(np.array(g_list)/norm)
        
        # Diagonalise RWA strips
        order = np.zeros(tmax, dtype=int)
        diag_bare = np.array([-i*wrl + E[i] for i in range(tmax)])
        eigensolver_order = np.linalg.eigvalsh(np.diag(diag_bare))
        for i in range(tmax):
            index, = np.where(eigensolver_order == diag_bare[i])
            order[i] = index[0]
        Erwa = [0.0] * nmax
        diagonal_elements = None
        offdiagonal_elements = None
        strip_H = None
        e = None
        for n in range(nmax):
            diagonal_elements = np.array([(n-i)*wrl + E[i] for i in range(tmax)])
            offdiagonal_elements = np.array([g_list[i]*np.sqrt((n-i)*(n-i>0)) for i in range(tmax-1)])
            strip_H = (np.diag(diagonal_elements) + np.diag(offdiagonal_elements, 1) +
                       np.diag(offdiagonal_elements, -1))
            e = sc.linalg.eigvalsh(strip_H)
            Erwa[n] = np.array([e[i].real for i in order])
        
        return np.array(Erwa)
    
    ###################################################################################################################
    #       Diagonaliser Configuration
    ###################################################################################################################
    
    def setDiagConfig(
        self,
        eigvalues=5,
        get_vectors=False,
        sparse=False,
        sparsesolveropts={"sigma":None, "mode":"normal", "maxiter":None, "tol":1e-3, "which":"SA"}
    ):
        self.diagonalizer_config = {
            'kwargs':{
                'eigvalues':eigvalues, 
                'get_vectors':get_vectors, 
                'sparsesolveropts':sparsesolveropts
            }, 
            'sparse':sparse
        }
        
        # Choose the diagonalizer function and matrix conversion operation
        if sparse:
            self.diagonalizer_config['func'] = util.diagSparseH
        else:
            self.diagonalizer_config['func'] = util.diagDenseH
    
    def getDiagConfig(self):
        return self.diagonalizer_config
    
    def diagonalize(self, qobj: qt.Qobj) -> tuple[EigenvalueResult, EigenvectorResult | None]:
        result = self.diagonalizer_config['func'](qobj, **self.diagonalizer_config['kwargs'])
        if self.diagonalizer_config["kwargs"]["get_vectors"]:
            return (
                EigenvalueResult(data=result[0], source="diagonalize"),
                EigenvectorResult(data=result[1], source="diagonalize")
            )
        return EigenvalueResult(data=result, source="diagonalize"), None
    
    ###################################################################################################################
    #       Parameter Collection Wrapper Functions and Extended Functions
    ###################################################################################################################
    
    ## Set the value of a parameter.
    def setParameterValue(self, name, value):
        self.SS.setParameterValue(name, value)
        params = self.getParameterValuesDict()
        assert all(value is not None for value in params.values()), \
               "Some parameters do not have valid values. All parameters should be set first with the " \
               "setParameterValues function in one go."
        self.substitute()
        self.prepareOperators()
    
    ## Get the value of a parameter.
    def getParameterValue(self, name):
        return self.SS.getParameterValue(name)
    
    ## Set many parameter values.
    def setParameterValues(self, *name_value_pairs):
        self.SS.setParameterValues(*name_value_pairs)
        params = self.getParameterValuesDict()
        assert all(value is not None for value in params.values()), "Not all parameters were set in this call. " \
               f"The missing parameters are {repr([name for name, value in params.items() if value is None])}"
        self.substitute()
        self.prepareOperators()
    
    ## Get many parameter values.
    def getParameterValues(self, *names):
        return self.SS.getParameterValues(*names)
    
    ## Gets all parameters
    def getParameterValuesDict(self):
        return self.SS.getParameterValuesDict()
    
    def getParameterSweep(self, name):
        return self.SS.getParameterSweep(name)
    
    def getParameterNames(self):
        return self.SS.getParameterNamesList()
    
    def getPrefactor(self, name):
        return self.units.getPrefactor(name)
    
    ###################################################################################################################
    #       Parameter Sweep Functions
    ###################################################################################################################
    def newSweepConfig(self) -> SweepConfig:
        config = SweepConfig(self.SS)
        # Default evaluation graph
        config.setEvaluationGraph(HamiltonianSpectrum(self))
        return config
    
    def runSweep(self, sweep_config: SweepConfig, use_disk: bool = True) -> SweepResult:
        # FIXME: Determine if we should be saving the data to temp files rather than in RAM:
        # Use the diagonaliser configuration, the requested evaluation functions, and the total number of sweep setpoints that will be used.
        self._perform_pre_sweep_substitutions(sweep_config)

        evaluation_graph = sweep_config.getEvaluationGraph()
        if evaluation_graph is None:
            raise RuntimeError("An evaluation graph must be set.")

        results = []
        point_gen = sweep_config.getGenerator()
        with progress.bar.Bar('Solving', check_tty=False, max=sweep_config.getTotalCount()) as bar:
            for sweep_point in point_gen:
                # Set the parameter values, this updates the state for the next call to work
                self.SS.setParameterValues(sweep_point)
                self._perform_sweep_point_substitutions(sweep_config)

                # Compute evaluation graph
                default_inputs = evaluation_graph.getDefaultInputs()
                result = evaluation_graph.evaluate(default_inputs)

                if use_disk:
                    # Write to temp file
                    f = self.writePart(result)
                    results.append(f)
                else:
                    results.append(result)
                bar.next()
            bar.finish()

        if use_disk:
            return SweepResultFromDisk(sweep_config, results)
        else:
            return SweepResultFromMemory(sweep_config, results)
    
    def _perform_pre_sweep_substitutions(self, sweep_config: SweepConfig):
        # Get the static parameters
        substitutions = sweep_config.getStaticSymbols()

        # Generate final symbolic expressions
        self.Cinv_pre = self.SS.getInverseCapacitanceMatrix().subs(substitutions)
        self.Linv_pre = self.SS.getInverseInductanceMatrix().subs(substitutions)

        # Get branch inverse inductance matrix for branch current calculations
        self.Linv_b_pre = self.SS.getInverseInductanceMatrix(mode='branch').subs(substitutions)
        
        self.Jvec_pre = self.SS.getJosephsonVector().subs(substitutions)
        self.Pvec_pre = self.SS.getPhaseSlipVector().subs(substitutions)
        self.Qb_pre = self.SS.getChargeBiasVector().subs(substitutions)
        self.Qbt_pre = self.SS.Rnb*self.SS.getChargeBiasVector().subs(substitutions)
        self.Pbm_pre = self.SS.getFluxBiasMatrix(mode="branch").subs(substitutions)
        self.Pbi_pre = self.SS.getFluxBiasVectorInd().subs(substitutions)
        
        # Find which operators will need to be regenerated for each sweep
        self._get_dynamic_node_dofs(sweep_config)
    
    def _get_dynamic_node_dofs(self, sweep_config: SweepConfig):
        # FIXME: Could identify the precise parameter and where they occur in the sweep to only regenerate when necessary rather than every iteration. This would require _postsub to know which iteration we are at.
        
        # Get the parameters that are being swept
        sweep_syms = set(sweep_config.getSweptSymbols())
        
        # For each node:
        self.regen_nodes = []
        for node, data in self.operator_data.items():
            if data["basis"] != "oscillator":
                continue
            
            # Check if those parameters are oscillator impedance parameters
            if not data["impedance"].free_symbols.isdisjoint(sweep_syms):
                self.regen_nodes.append(node)
    
    def _perform_sweep_point_substitutions(self, sweep_config: SweepConfig):
        # Get the subs
        substitutions = sweep_config.getSweptSymbols()
        
        # Substitute circuit parameters
        self.Cinvnp = np.asmatrix(self.Cinv_pre.subs(substitutions), dtype=np.float64)
        self.Linvnp = np.asmatrix(self.Linv_pre.subs(substitutions), dtype=np.float64)
        self.Jvecnp = np.asarray(self.Jvec_pre.subs(substitutions), dtype=np.float64)[:, 0]
        self.Pvecnp = np.asarray(self.Pvec_pre.subs(substitutions), dtype=np.float64)[:, 0]
        self.Linvnp_b = np.asmatrix(self.Linv_b_pre.subs(substitutions), dtype=np.float64)
        
        # Substitute external biases
        self.Qbnp = np.asmatrix(self.Qb_pre.subs(substitutions), dtype=np.float64) # x 2e
        self.Qbtnp = np.asmatrix(self.Qbt_pre.subs(substitutions), dtype=np.float64) # x 2e
        self.Pbsm = np.asmatrix(self.Pbm_pre.subs(substitutions), dtype=np.float64)
        #self.Pbnp = np.asmatrix(self.Pb_pre.subs(subs), dtype=np.float64) # x Phi0
        self.Pbinp = np.asmatrix(self.Pbi_pre.subs(substitutions), dtype=np.float64)
        
        # Generate exponentiated flux biases
        Pexp1 = []
        Pexp2 = []
        for i in range(self.Pbsm.shape[0]):
            Pexp1.append(np.exp(2j*np.pi*self.Pbsm[i, i]))
            Pexp2.append(np.exp(-2j*np.pi*self.Pbsm[i, i]))
        self.Pexp_pnp = Pexp1
        self.Pexp_mnp = Pexp2
        
        # Generate exponentiated charge biases
        Qexp1 = []
        Qexp2 = []
        for i in range(self.Qbtnp.shape[0]):
            Qexp1.append(np.exp(2j*np.pi*self.Qbtnp[i, 0]))
            Qexp2.append(np.exp(-2j*np.pi*self.Qbtnp[i, 0]))
        self.Qexp_pnp = Qexp1
        self.Qexp_mnp = Qexp2
        
        # Regenerate operators if required
        if self.regen_nodes != []:
            self.getExpandedOperatorsMap(self.regen_nodes)

    ###################################################################################################################
    #       Internal Functions
    ###################################################################################################################
    # Replaces np asmatrix
    def _init_qobj_vector(self, obj_list, dtype=None):
        obj = np.empty((len(obj_list), 1) , dtype=dtype)
        for i, op in enumerate(obj_list):
            obj[i, 0] = op
        return obj
    
    # Operator generators
    def _get_oscillator_basis(self, node):
        trunc = self.operator_data[node]["truncation"]
        
        # Get the impedance of the mode
        i = self.getNodeIndex(node)
        osc_impedance = self.getParameterValue("Zosc%i" % node)*self.units.getPrefactor("Impe")
        
        # Get the prefactors that results from transformation (the charge and fluxon increment prefactors)
        a = self.SS.cooper_disp[node]
        b = self.SS.fluxon_disp[node]
        
        # Using oscillator basis
        Q = 1j*np.sqrt(1/(2*osc_impedance))*(qt.create(trunc) - qt.destroy(trunc))*self.units.getPrefactor("ChgOsc")
        P = np.sqrt(osc_impedance/2)*(qt.create(trunc) + qt.destroy(trunc))*self.units.getPrefactor("FlxOsc")
        Pp = a*2*np.pi/pc.phi0*np.sqrt(pc.hbar)*P/self.units.getPrefactor("FlxOsc")
        Qp = b*np.pi/pc.e*np.sqrt(pc.hbar)*Q/self.units.getPrefactor("ChgOsc")
        
        # Generate Josephson displacement operators by first diagonalising the flux operator
        E, V = np.linalg.eigh(Pp.data.to_array())
        
        # Create transformation matrices
        U = qt.Qobj(V)
        Uinv = qt.Qobj(np.linalg.inv(V))
        
        # Exponentiate the diagonal matrix
        D = U*qt.Qobj(np.diag(np.exp(1j*E)))*Uinv
        Ddag = D.dag()
        
        # Generate Phase Slip displacement operators by first diagonalising the charge operator
        E, V = np.linalg.eigh(Qp.data.to_array())
        
        # Create transformation matrices
        U = qt.Qobj(V)
        Uinv = qt.Qobj(np.linalg.inv(V))
        
        # Exponentiate the diagonal matrix
        S = U*qt.Qobj(np.diag(np.exp(1j*E)))*Uinv
        Sdag = S.dag()
        return (
            Q.to("CSR").tidyup(_qobj_atol),
            P.to("CSR").tidyup(_qobj_atol),
            D.to("CSR").tidyup(_qobj_atol),
            Ddag.to("CSR").tidyup(_qobj_atol),
            S.to("CSR").tidyup(_qobj_atol),
            Sdag.to("CSR").tidyup(_qobj_atol)
        )
    
    def _get_charge_basis(self, node):
        trunc = self.operator_data[node]["truncation"]
        
        # For IJ modes the charge states are the eigenvectors of the following matrix
        q, s = (-qt.num(2*trunc + 1, trunc)).eigenstates()
        
        # Construct the flux states
        phik_list = []
        phik = None
        qm = [float(i)-trunc for i in range(2*trunc + 1)]
        for k in qm:
            phik = qt.basis(2*trunc + 1, 0) * 0
            for j, qi in enumerate(qm):
                phik += np.sqrt(1/(2*trunc + 1)) * np.exp(2j*np.pi*k*qi/(2*trunc + 1))*s[j]
            phik_list.append(phik)
        
        # From this build the flux operator
        phik_eigvals = [(float(i)-trunc)/(2*trunc + 1) for i in range(2*trunc + 1)]
        P = qt.qeye(2*trunc + 1) - qt.qeye(2*trunc + 1)
        for i, phik in enumerate(phik_list):
            P += phik_eigvals[i]*phik*phik.dag()
        
        # Get a simple charge number operator
        Q = -qt.num(2*trunc + 1)+float(trunc)
        
        # Generate Josephson displacement operators by first diagonalising the flux operator
        E, V = np.linalg.eigh(P.data.to_array())
        
        # Create transformation matrices
        U = qt.Qobj(V)
        Uinv = qt.Qobj(np.linalg.inv(V))
        
        # Exponentiate the diagonal matrix
        D = U*qt.Qobj(np.diag(np.exp(-2j*np.pi*E)))*Uinv - qt.basis(2*trunc+1, 2*trunc)*qt.basis(2*trunc+1, 0).dag()
        Ddag = D.dag()
        
        # Generate Phase Slip displacement operators by just exponentiating the charge operator, which is diagonal already in this case
        S = qt.Qobj(np.diag(np.exp(-2j*np.pi*q)))
        Sdag = S.dag()
        return (
            Q.to("CSR").tidyup(_qobj_atol),
            P.to("CSR").tidyup(_qobj_atol),
            D.to("CSR").tidyup(_qobj_atol),
            Ddag.to("CSR").tidyup(_qobj_atol),
            S.to("CSR").tidyup(_qobj_atol),
            Sdag.to("CSR").tidyup(_qobj_atol)
        )
    
    def _get_flux_basis(self, node):
        trunc = self.operator_data[node]["truncation"]
        
        # Flux operator counts the fluxon occupation
        P = -qt.num(2*trunc + 1)+float(trunc)
        q, s = P.eigenstates()
        
        # Construct the flux states
        phik_list = []
        phik = None
        qm = [float(i)-trunc for i in range(2*trunc + 1)]
        for k in qm:
            phik = qt.basis(2*trunc + 1, 0) * 0
            for j, qi in enumerate(qm):
                phik += np.sqrt(1/(2*trunc + 1)) * np.exp(2j*np.pi*k*qi/(2*trunc + 1))*s[j]
            phik_list.append(phik)

        # From this build the flux operator
        phik_eigvals = [(float(i)-trunc)/(2*trunc + 1) for i in range(2*trunc + 1)]
        Q = qt.qeye(2*trunc + 1) - qt.qeye(2*trunc + 1)
        for i, phik in enumerate(phik_list):
            Q += phik_eigvals[i]*phik*phik.dag()
        
        # Generate Josephson displacement operators by first diagonalising the flux operator
        E, V = np.linalg.eigh(P.data.to_array())
        
        # Create transformation matrices
        U = qt.Qobj(V)
        Uinv = qt.Qobj(np.linalg.inv(V))
        
        # Exponentiate the diagonal matrix
        D = U*qt.Qobj(np.diag(np.exp(-2j*np.pi*E)))*Uinv - qt.basis(2*trunc+1, 2*trunc)*qt.basis(2*trunc+1, 0).dag()
        Ddag = D.dag()
        
        # Generate Phase displacement operators by first diagonalising the charge operator
        E, V = np.linalg.eigh(Q.data.to_array())
        
        # Create transformation matrices
        U = qt.Qobj(V)
        Uinv = qt.Qobj(np.linalg.inv(V))
        
        # Exponentiate the diagonal matrix
        S = U*qt.Qobj(np.diag(np.exp(-2j*np.pi*E)))*Uinv - qt.basis(2*trunc+1, 2*trunc)*qt.basis(2*trunc+1, 0).dag()
        Sdag = S.dag()
        return (
            Q.to("CSR").tidyup(_qobj_atol),
            P.to("CSR").tidyup(_qobj_atol),
            D.to("CSR").tidyup(_qobj_atol),
            Ddag.to("CSR").tidyup(_qobj_atol),
            S.to("CSR").tidyup(_qobj_atol),
            Sdag.to("CSR").tidyup(_qobj_atol)
        )
    
    def _get_discretized_flux_basis(self, node):
        trunc = self.operator_data[node]["truncation"]
        pmax = self.operator_data[node]["flux_max"]
        
        # Flux operator counts the fluxon occupation
        grid = np.linspace(-pmax, pmax, 2*trunc+1)
        P = qt.Qobj(np.diag(grid))
        
        q, s = P.eigenstates()
        
        # Construct the flux states
        phik_list = []
        phik = None
        qm = [float(i)-trunc for i in range(2*trunc + 1)]
        for k in qm:
            phik = qt.basis(2*trunc + 1, 0) * 0
            for j, qi in enumerate(qm):
                phik += np.sqrt(1/(2*trunc + 1)) * np.exp(2j*np.pi*k*qi/(2*trunc + 1))*s[j]
            phik_list.append(phik)

        # From this build the flux operator
        phik_eigvals = [(float(i)-trunc)/(2*trunc + 1) for i in range(2*trunc + 1)]
        Q = qt.qeye(2*trunc + 1) - qt.qeye(2*trunc + 1)
        for i, phik in enumerate(phik_list):
            Q += phik_eigvals[i]*phik*phik.dag()
        
        Q = Q/(grid[1]-grid[0])
        
        # Generate Josephson displacement operators by first diagonalising the flux operator
        E, V = np.linalg.eigh(P.data.to_array())
        
        # Create transformation matrices
        U = qt.Qobj(V)
        Uinv = qt.Qobj(np.linalg.inv(V))
        
        # Exponentiate the diagonal matrix
        D = U*qt.Qobj(np.diag(np.exp(-2j*np.pi*E)))*Uinv - qt.basis(2*trunc+1, 2*trunc)*qt.basis(2*trunc+1, 0).dag()
        Ddag = D.dag()
        
        # Generate Phase displacement operators by first diagonalising the charge operator
        E, V = np.linalg.eigh(Q.data.to_array())
        
        # Create transformation matrices
        U = qt.Qobj(V)
        Uinv = qt.Qobj(np.linalg.inv(V))
        
        # Exponentiate the diagonal matrix
        S = U*qt.Qobj(np.diag(np.exp(-2j*np.pi*E)))*Uinv - qt.basis(2*trunc+1, 2*trunc)*qt.basis(2*trunc+1, 0).dag()
        Sdag = S.dag()
        return (
            Q.to("CSR").tidyup(_qobj_atol),
            P.to("CSR").tidyup(_qobj_atol),
            D.to("CSR").tidyup(_qobj_atol),
            Ddag.to("CSR").tidyup(_qobj_atol),
            S.to("CSR").tidyup(_qobj_atol),
            Sdag.to("CSR").tidyup(_qobj_atol)
        )
    
    # FIXME: This causes issues when regenerating code
    def _set_parameter_units(self):
        
        # Get the unit prefactors
        Uf = self.units.getUnitPrefactor('Hz')
        Uo = self.units.getUnitPrefactor('Ohm')
        Uc = self.units.getUnitPrefactor('F')
        Ul = self.units.getUnitPrefactor('H')
        
        # Resonators
        if self.SS._has_resonators():
            for node, resonator in self.SS.CG.resonators_cap.items():
                if resonator is not None:
                    self.SS.addParameterisationPrefactor(resonator["Cr"], 1/(Uf*Uo*Uc))
                    self.SS.addParameterisationPrefactor(resonator["Lr"], Uo/(Uf*Ul))
                    self.SS.addParameterisationPrefactor(resonator["gC"], self.units.getPrefactor('ChgOscCpl'))
                    self.SS.addParameterisationPrefactor(resonator["frl"], self.units.getPrefactor('Freq'))
                    self.SS.addParameterisationPrefactor(resonator["Zrl"], self.units.getPrefactor('Impe'))


class ClassicalPotentialBuilder:
    """Used internally to build a numerical classical potential energy getter as a function of all the parameters.
    """
    def __init__(self, hamil):
        self._hamil = hamil
        self._dof_map = {}
        self._critical_currents = self._hamil.getPrefactor('Ej') * self._get_critical_currents()
        self._inverse_inductance_matrix = 0.5 * hamil.getPrefactor('El') * self._hamil.Linvnp
        self._dof_symbol_vector = self._hamil.SS.getFluxVector(mode="branch") + \
                                 self._hamil.SS.getFluxBiasVector(mode="branch")
        self._get_input_format()
        self._symbolic_inductive_energy = self._get_symbolic_inductive_energy()

        # Create numerical functions
        self.jj_func = syu.lambdify(list(self._dof_map.values()), self._dof_symbol_vector)
        self.ind_func = syu.lambdify(list(self._dof_map.values()), self._symbolic_inductive_energy)

    def _get_critical_currents(self):
        subs = self._hamil.SS.getSymbolValuesDict()
        assert all(val is not None for val in subs.values()), "All parameters need to be initialized"
        Jvec = self._hamil.SS.getJosephsonVector().transpose().subs(subs)
        return np.array(Jvec).astype(np.float64)

    def _get_symbolic_inductive_energy(self):
        P = self._hamil.SS.getFluxVector()
        Pe = self._hamil.SS.getFluxBiasVectorInd()
        expr = (P + Pe).transpose() * sy.Matrix(self._inverse_inductance_matrix) * (P + Pe)
        return expr[0, 0]

    def _get_input_format(self):
        # Get the DoF symbol-node map
        self._dof_map = {"phi%i" % n: sym[0] for n, sym in self._hamil.SS.node_dofs.items() if n > 0}

        # Get the bias symbols
        bias_terms = self._hamil.SS.getFluxBiasVector(mode="branch").free_symbols
        for sym in bias_terms:
            name = self._hamil.SS.getParameterFromSymbol(sym)
            self._dof_map[name] = sym

    def getDefaultInputs(self):
        return {name: 0.0 for name in self._dof_map}

    def getPotentialFunction(self):
        def potential(inputs):
            # Check inputs
            assert isinstance(inputs, dict), "inputs should be a dict instance"
            for name in self._dof_map:
                assert name in inputs, f"expected attribute {name} in inputs"
                assert isinstance(inputs[name], (float, int, np.float64, np.ndarray))

            # Order the arguments correctly
            args = [inputs[name] for name in self._dof_map]

            # Turn the JJ vector into numerical function and get the values
            result = 0.0
            for i, I in enumerate(self._critical_currents[0]):
                result += I*np.cos(2*np.pi*self.jj_func(*args)[i, 0])

            # Calculate the inductive terms
            result += self.ind_func(*args)
            return result

        return potential

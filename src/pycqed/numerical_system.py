import qutip as qt
import numpy as np
import sympy as sy
import scipy as sc
import networkx as nx
import time

from . import dataspec as ds
from . import symbolic_system as cs
from . import parameters as pa
from . import units as un
from . import physical_constants as pc
from . import util
from . import units

class NumericalSystem(ds.TempData):
    
    ## Mode types
    __mode_types = [
        "osc", # DoFs have capacitive and inductive parts only
        "jos", # DoFs have capacitive, inductive and Josephson parts
        "isl", # DoFs have capacitive parts only
        "leg"  # DoFs have inductive parts only
    ]
    
    ## Initialise a Hamiltonian using a circuit specification
    def __init__(self, symbolic_system, unit=units.Units("CQED1")):
        
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
        
        # Init the sweeper data
        self._init_sweep_data()
        
        # Load the symbolic expressions
        self.getSymbolicExpressions()
    
    # Called when deleting
    def __del__(self):
        self.clearSessionData()
    
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
        return 1 - op.data.nnz/self.getHilbertSpaceSize()**2
    
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
            corrmat = qt.Qobj(np.linalg.inv(mat.data.todense()))
            
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
        elif basis == "flux":
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
        trunc = self.operator_data[node]["truncation"]
        basis = self.operator_data[node]["basis"]
        #impedance = self.operator_data[node]["impedance"]
        
        if basis == "charge":
            # For IJ modes the charge states are the eigenvectors of the following matrix
            q, s = (-qt.num(2*trunc + 1, trunc)).eigenstates()
            
            # Construct the flux states
            phik_list = []
            phik = None
            qm = [float(i)-trunc for i in range(2*trunc + 1)]
            for k in qm:
                phik = qt.basis(2*trunc + 1, 0) - 1
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
            
            # Generate displacement operators by first diagonalising the flux operator
            E, V = np.linalg.eigh(P.data.todense())
            
            # Create transformation matrices
            U = qt.Qobj(V)
            Uinv = qt.Qobj(np.linalg.inv(V))
            
            # Exponentiate the diagonal matrix
            D = U*qt.Qobj(np.diag(np.exp(-2j*np.pi*E)))*Uinv - qt.basis(2*trunc+1, 2*trunc)*qt.basis(2*trunc+1, 0).dag()
            Ddag = D.dag()
            
        elif basis == "oscillator":
            
            # Get the impedance of the mode
            #osc_impedance = None
            #if impedance is None:
            #    raise Exception("Basis for node %i is oscillator, but impedance expression is not set.")
            #subs = self.SS.getSymbolValuesDict()
            #num = impedance.subs(subs)
            #osc_impedance = float(num)*self.units.getPrefactor("Impe")
            #if osc_impedance != osc_impedance or osc_impedance == 0.0:
            #    raise Exception("Parameters not set for oscillator mode '%i'." % node)
            i = self.getNodeIndex(node)
            osc_impedance = self.getParameterValue("Zosc%i" % node)*self.units.getPrefactor("Impe")
            
            # Get the prefactor that results from transformation (the charge increment prefactor)
            a = self.SS.cooper_disp[node]
            
            # Using oscillator basis
            Q = 1j*np.sqrt(1/(2*osc_impedance))*(qt.create(trunc) - qt.destroy(trunc))*self.units.getPrefactor("ChgOsc")
            P = np.sqrt(osc_impedance/2)*(qt.create(trunc) + qt.destroy(trunc))*self.units.getPrefactor("FlxOsc")
            Pp = a*2*np.pi/pc.phi0*np.sqrt(pc.hbar)*P/self.units.getPrefactor("FlxOsc")
            
            # Generate displacement operators by first diagonalising the flux operator
            E, V = np.linalg.eigh(Pp.data.todense())
            
            # Create transformation matrices
            U = qt.Qobj(V)
            Uinv = qt.Qobj(np.linalg.inv(V))
            
            # Exponentiate the diagonal matrix
            D = U*qt.Qobj(np.diag(np.exp(1j*E)))*Uinv
            Ddag = D.dag()
        elif basis == "flux":
            # Flux operator is based on a flux grid
            pmax = self.operator_data[node]["flux_max"]
            grid = np.linspace(-pmax, pmax, 2*trunc+1)
            P = qt.Qobj(np.diag(grid))
            
            q, s = P.eigenstates()
            
            # Construct the flux states
            phik_list = []
            phik = None
            qm = [float(i)-trunc for i in range(2*trunc + 1)]
            for k in qm:
                phik = qt.basis(2*trunc + 1, 0) - 1
                for j, qi in enumerate(qm):
                    phik += np.sqrt(1/(2*trunc + 1)) * np.exp(2j*np.pi*k*qi/(2*trunc + 1))*s[j]
                phik_list.append(phik)

            # From this build the flux operator
            phik_eigvals = [(float(i)-trunc)/(2*trunc + 1) for i in range(2*trunc + 1)]
            Q = qt.qeye(2*trunc + 1) - qt.qeye(2*trunc + 1)
            for i, phik in enumerate(phik_list):
                Q += phik_eigvals[i]*phik*phik.dag()
            
            Q = Q/(grid[1]-grid[0])
            
            # Generate displacement operators by first diagonalising the flux operator
            E, V = np.linalg.eigh(P.data.todense())
            
            # Create transformation matrices
            U = qt.Qobj(V)
            Uinv = qt.Qobj(np.linalg.inv(V))
            
            # Exponentiate the diagonal matrix
            D = U*qt.Qobj(np.diag(np.exp(-2j*np.pi*E)))*Uinv - qt.basis(2*trunc+1, 2*trunc)*qt.basis(2*trunc+1, 0).dag()
            Ddag = D.dag()
            
        return Q, P, D, Ddag
    
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
            Q, P, D, Ddag = self.getOperatorList(node)
            
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
        self.Qb = self.SS.getChargeBiasVector()
        self.Pb = self.SS.getFluxBiasVector(mode="branch")
        self.Pbm = self.SS.getFluxBiasMatrix(mode="branch")
        
        # Basis representation prefactors
        #self.Zpref = self.getBasisPrefactors()
    
    def prepareOperators(self):
        self.getExpandedOperatorsMap()
        self.getChargeOpVector()
        self.getFluxOpVector()
    
    ###################################################################################################################
    #       Numerical Hamiltonian Generation
    ###################################################################################################################
    
    def substitute(self):
        
        subs = self.SS.getSymbolValuesDict()
        
        # Substitute circuit parameters
        self.Cinvnp = np.asmatrix(self.Cinv.subs(subs), dtype=np.float64)
        self.Linvnp = np.asmatrix(self.Linv.subs(subs), dtype=np.float64)
        self.Jvecnp = np.asarray(self.Jvec.subs(subs), dtype=np.float64)[:, 0]
        
        # Substitute external biases
        self.Qbnp = np.asmatrix(self.Qb.subs(subs), dtype=np.float64) # x 2e
        self.Pbsm = np.asmatrix(self.Pbm.subs(subs), dtype=np.float64)
        self.Pbnp = np.asmatrix(self.Pb.subs(subs), dtype=np.float64) # x Phi0
        
        # Generate exponentiated biases
        Pexp1 = []
        Pexp2 = []
        for i in range(self.Pbsm.shape[0]):
            Pexp1.append(np.exp(2j*np.pi*self.Pbsm[i, i]))
            Pexp2.append(np.exp(-2j*np.pi*self.Pbsm[i, i]))
        self.Pexpbnp = np.asmatrix(np.diag(Pexp1), dtype=np.complex64)
        self.Pexpbnpc = np.asmatrix(np.diag(Pexp2), dtype=np.complex64)
        self.Pexp_pnp = Pexp1
        self.Pexp_mnp = Pexp2
        
        # Generate basis representation prefactors
        #self.Zprefnp = np.asmatrix(self.Zpref.subs(subs), dtype=np.float64)
        
        # Get branch inverse inductance matrix for branch current calculations
        self.Linvnp_b = np.asmatrix(self.Linv_b.subs(subs), dtype=np.float64)
    
    def _presub(self):
        # Set the parameters that are not being swept
        subs = self.SS.getNonSweepParametersDict()
        
        # Generate final symbolic expressions
        self.Cinv_pre = self.SS.getInverseCapacitanceMatrix().subs(subs)
        self.Linv_pre = self.SS.getInverseInductanceMatrix().subs(subs)

        # Get branch inverse inductance matrix for branch current calculations
        self.Linv_b_pre = self.SS.getInverseInductanceMatrix(mode='branch').subs(subs)
        
        self.Jvec_pre = self.SS.getJosephsonVector().subs(subs)
        self.Qb_pre = self.SS.getChargeBiasVector().subs(subs)
        self.Pbm_pre = self.SS.getFluxBiasMatrix(mode="branch").subs(subs)
        
        # Generate basis representation prefactors
        #self.Zpref_pre = self.Zpref.subs(subs)
        
        # Find which operators will need to be regenerated for each sweep
        self._get_regen_coordinate_nodes()
    
    def _get_regen_coordinate_nodes(self):
        
        # FIXME: Could identify the precise parameter and where they occur in the sweep to only regenerate when necessary rather than every iteration. This would require _postsub to know which iteration we are at.
        
        # Get the parameters that are being swept
        sweep_syms = set(self.SS.getSweepParametersDict().keys())
        
        # For each node:
        self.regen_nodes = []
        for node, data in self.operator_data.items():
            if data["basis"] != "oscillator":
                continue
            
            # Check if those parameters are oscillator impedance parameters
            if not data["impedance"].free_symbols.isdisjoint(sweep_syms):
                self.regen_nodes.append(node)
    
    def _postsub(self, params):
    
        # Set the parameter values
        self.SS.setParameterValues(params)
        
        # Get the subs
        subs = self.SS.getSweepParametersDict()
        
        # Substitute circuit parameters
        self.Cinvnp = np.asmatrix(self.Cinv_pre.subs(subs), dtype=np.float64)
        self.Linvnp = np.asmatrix(self.Linv_pre.subs(subs), dtype=np.float64)
        self.Jvecnp = np.asarray(self.Jvec_pre.subs(subs), dtype=np.float64)[:, 0]
        self.Linvnp_b = np.asmatrix(self.Linv_b_pre.subs(subs), dtype=np.float64)
        
        # Substitute external biases
        self.Qbnp = np.asmatrix(self.Qb_pre.subs(subs), dtype=np.float64) # x 2e
        self.Pbsm = np.asmatrix(self.Pbm_pre.subs(subs), dtype=np.float64)
        #self.Pbnp = np.asmatrix(self.Pb_pre.subs(subs), dtype=np.float64) # x Phi0
        
        # Generate exponentiated biases
        Pexp1 = []
        Pexp2 = []
        for i in range(self.Pbsm.shape[0]):
            Pexp1.append(np.exp(2j*np.pi*self.Pbsm[i, i]))
            Pexp2.append(np.exp(-2j*np.pi*self.Pbsm[i, i]))
        self.Pexpbnp = np.asmatrix(np.diag(Pexp1), dtype=np.complex64)
        self.Pexpbnpc = np.asmatrix(np.diag(Pexp2), dtype=np.complex64)
        self.Pexp_pnp = Pexp1
        self.Pexp_mnp = Pexp2
        
        # Generate basis representation prefactors
        #self.Zprefnp = np.asmatrix(self.Zpref_pre.subs(subs), dtype=np.float64)
        
        # Regenerate operators if required
        if self.regen_nodes != []:
            self.getExpandedOperatorsMap(self.regen_nodes)
    
    ###################################################################################################################
    #       Evaluables
    ###################################################################################################################
    
    __eval_spec = {
        "Hamiltonian":        {'eval': 'getHamiltonian', 'diag': True, 'depends': None, 'kwargs': {}},
        "Resonator":          {'eval': 'getResonatorResponse', 'diag': False, 'depends': 'getHamiltonian', 'kwargs': {}},
        "Current":            {'eval': 'getCurrentMatrixElement', 'diag': False, 'depends': 'getHamiltonian', 'kwargs': {}},
        "Voltage":            {'eval': 'getVoltageMatrixElement', 'diag': False, 'depends': 'getHamiltonian', 'kwargs': {}},
        "ChargingEnergy":     {'eval': 'getChargingEnergies', 'diag': False, 'depends': None, 'kwargs': {}},
        "FluxEnergy":         {'eval': 'getFluxEnergies', 'diag': False, 'depends': None, 'kwargs': {}},
        "JosephsonEnergy":    {'eval': 'getJosephsonEnergies', 'diag': False, 'depends': None, 'kwargs': {}}
    }
    
    def getHamiltonian(self):
        # Get charging energy
        self.Hq = self.units.getPrefactor("Ec")*0.5*\
        util.mdot((self.Qnp + self.Qbnp).T, self.Cinvnp, self.Qnp + self.Qbnp)[0, 0]
        
        # Get flux energy
        self.Hp = self.units.getPrefactor("El")*0.5*\
        util.mdot(self.Pnp.T, self.Linvnp, self.Pnp)[0, 0]
        
        # Get Josephson energy
        #self.Hj = -self.units.getPrefactor("Ej")*0.5*\
        #(util.mdot(self.Jvecnp, self.Pexpbnpc, self.Dl_adj, self.Dr_adj) + \
        # util.mdot(self.Jvecnp, self.Pexpbnp, self.Dl, self.Dr))
        
        # Create the transformed branch vector
        Pp = self.SS.Rnb*self.SS.Rinv*self.SS.node_vector
        
        self.Hj = 0
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
        
            self.Hj += -0.5*self.Jvecnp[i]*(prod1 + prod2)
        self.Hj *= self.units.getPrefactor("Ej")
        
        # Total Hamiltonian
        self.Ht = (self.Hq + self.Hp + self.Hj)
        return self.Ht
    
    def getCurrentOperator(self, edge=None):
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
    
    def getCurrentMatrixElement(self, E, V, edge=None, elements=None):
        # Get the relevant operator
        Iop = self.getCurrentOperator(edge=edge)
        
        # Check the elements
        if elements is None:
            raise Exception("No elements specified for current matrix elements.")
        result = np.zeros(len(elements), dtype=np.float64)
        for i, indices in enumerate(elements):
            i1, i2 = indices
            result[i] = Iop.matrix_element(V[i1], V[i2]).real
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
    
    def getVoltageMatrixElement(self, E, V, node=None, elements=None):
        # Get the relevant operator
        Vop = self.getVoltageOperator(node=node)
        
        # Check the elements
        if elements is None:
            raise Exception("No elements specified for voltage matrix elements.")
        result = np.zeros(len(elements), dtype=np.float64)
        for i, indices in enumerate(elements):
            i1, i2 = indices
            result[i] = Vop.matrix_element(V[i1], V[i2]).real
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
    
    def getResonatorResponse(self, E, V, nmax=100, cpl_node=None):
        # Save the derived parameter values for each sweep value
        if cpl_node is None:
            #for k in self.SS.coupled_subsys[self.subsystem]['derived_parameters'].keys():
            #    self.subsys_dparams[self.subsystem][k].append(self.dpnp[k])
            return None
        
        # Get the model parameters
        gC = self.SS.getParameterValue('g%ir' % cpl_node)
        wrl = self.SS.getParameterValue('f%irl' % cpl_node)
        
        # Get the operator associated with selected node
        index = self.getNodeList().index(cpl_node)
        Op = self.Qnp[index, 0] + self.Qbnp[index, 0]
        
        # Get coupling terms
        E = E - E[0]
        tmax = len(E)
        nmax = nmax + 1 + tmax
        norm = Op.matrix_element(V[0], V[1])#(V[0].dag()*Op*V[1])[0][0][0]
        g_list = []
        for i in range(tmax-1):
            #g_list.append((V[i].dag()*Op*V[i+1])[0][0][0])
            g_list.append(Op.matrix_element(V[i], V[i+1]))
        
        # Apply the circuit derived prefactor
        g_list = gC * np.abs(np.array(g_list)/norm)
        
        # Diagonalise RWA strips
        order = np.zeros(tmax, dtype=np.int)
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
    
    def setDiagConfig(self, eigvalues=5, get_vectors=False, sparse=False, sparsesolveropts={"sigma":None, "mode":"normal", "maxiter":None, "tol":1e-3, "which":"SA"}):
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
    
    def diagonalize(self, M):
        return self.diagonalizer_config['func'](M, **self.diagonalizer_config['kwargs'])
    
    ###################################################################################################################
    #       Parameter Collection Wrapper Functions and Extended Functions
    ###################################################################################################################
    
    ## Set the value of a parameter.
    def setParameterValue(self, name, value):
        self.SS.setParameterValue(name, value)
        self.substitute()
        self.prepareOperators()
    
    ## Get the value of a parameter.
    def getParameterValue(self, name):
        return self.SS.getParameterValue(name)
    
    ## Set many parameter values.
    def setParameterValues(self, *name_value_pairs):
        self.SS.setParameterValues(*name_value_pairs)
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
    
    def newSweep(self):
        self._init_sweep_data()
    
    ## Create a sweep specification for a single parameter.
    def addSweep(self, *args, **kwargs):
        self.sweep_specs.append(self.SS.paramSweepSpec(*args, **kwargs))
    
    ## Create an evaluation specification for a single function.
    def addEvaluation(self, evaluable, **kwargs):
        if evaluable not in self.__eval_spec.keys():
            raise Exception("Evaluable '%s' not valid." % evaluable)
        
        # Set up the evaluable data
        data = self.__eval_spec[evaluable].copy()
        data['kwargs'] = kwargs
        self.evaluations.append(data)
        # FIXME: Can implement kwargs checking here
        # FIXME: Can implement proper order of evaluations here
    
    def getSweep(self, data, ind_var, static_vars, evaluable="Hamiltonian"):
        if evaluable not in self.__eval_spec.keys():
            raise Exception("Evaluable '%s' not valid." % evaluable)
        key = self.__eval_spec[evaluable]['eval']
        return self.SS.getSweepResult(ind_var, static_vars, data=data, key=key)
    
    def paramSweep(self, timesweep=False):
        
        # Time initialisation
        if timesweep:
            init_time = time.time()
        
        # FIXME: Automatically configure diagonaliser here, based on data on the evaluables
        
        # Check if using default evaluables
        if len(self.evaluations) == 0:
            self.evaluations = [self.__eval_spec["Hamiltonian"]]
        
        # Generate sweep grid
        self.SS.ndSweep(self.sweep_specs)
        
        # FIXME: Determine if we should be saving the data to temp files rather than in RAM:
        # Use the diagonaliser configuration, the requested evaluation functions, and the total number of sweep setpoints that will be used.
        self.__use_temp = True
        if self.__use_temp:
            tmp_results = []
        
        # Do pre-substitutions to avoid repeating un-necessary substitutions in loops
        self._presub()
        
        # FIXME: Check that all symbolic variables have an associated value at this point
        
        # Do the requested evaluations
        if len(self.evaluations) > 1:
            
            # Prepare the results structure
            results = {}
            for entry in self.evaluations:
                results[entry['eval']] = []
            
            # Time loop
            if timesweep:
                loop_time = time.time()
            for i in range(self.SS.sweep_grid_npts):
                # Do the post-substitutions
                self._postsub({k: v[i] for k, v in self.SS.sweep_grid_c.items()})
                
                # Get requested evaluables
                E = None
                V = None
                for entry in self.evaluations:
                    
                    # Check if this evaluable depends on another
                    if entry['depends'] is not None:
                        try:
                            if self.__use_temp:
                                dep = results[entry['depends']]
                            else:
                                dep = results[entry['depends']][i]
                        except:
                            raise Exception("eval spec with 'depends':'%s' entry should be specified after the one it depends on ('%s'), or Possibly invalid 'depends' value." % (entry['depends'], entry['eval'])) # FIXME
                        
                        # In almost every case the depends will be on the eigenvalues and eigenvectors of the independent eval spec
                        try:
                            E, V = dep
                        except:
                            raise Exception("need eigenvectors for 'depends'")
                    
                    # Check if evaluation depends on eigenvalues and eigenvectors and run it
                    if V is not None:
                        M = getattr(self, entry['eval'])(E, V, **entry['kwargs'])
                    else:
                        M = getattr(self, entry['eval'])(**entry['kwargs'])
                    
                    # Check if diagonalisation is required
                    if self.__use_temp:
                        if entry['diag']:
                            results[entry['eval']] = self.diagonalize(M)
                        else:
                            results[entry['eval']] = M
                    else:
                        if entry['diag']:
                            results[entry['eval']].append(self.diagonalize(M))
                        else:
                            results[entry['eval']].append(M)
                    E = None
                    V = None
                
                if self.__use_temp:
                    # Write to temp file
                    f = self.writePart(results)
                    tmp_results.append(f)
            
            # Convert the result entries to ndarray
            if not self.__use_temp:
                for entry in self.evaluations:
                    results[entry['eval']] = np.array(results[entry['eval']])
        
        # Do the single requested evaluation
        else:
            results = []
            entry = self.evaluations[0]
            
            # Time loop
            if timesweep:
                loop_time = time.time()
            for i in range(self.SS.sweep_grid_npts):
                # Do the post-substitutions
                self._postsub(dict([(k, v[i]) for k, v in self.SS.sweep_grid_c.items()]))
                
                # Get requested evaluable
                M = getattr(self, entry['eval'])(**entry['kwargs'])
                if self.__use_temp:
                    if entry['diag']:
                        results = self.diagonalize(M)
                    else:
                        results = M
                else:
                    if entry['diag']:
                        results.append(self.diagonalize(M))
                    else:
                        results.append(M)
                
                if self.__use_temp:
                    # Write to temp file
                    f = self.writePart(results)
                    tmp_results.append(f)
            
            # Convert results to ndarray
            if not self.__use_temp:
                results = np.array(results)
        
        # Reset the evaluables
        self._init_sweep_data()
        
        # Report timings
        if timesweep:
            end_time = time.time()
            print ("Parameter Sweep Duration:")
            print ("  Initialization:\t%.3f s" % (loop_time-init_time))
            print ("  Loop duration:\t%.3f s" % (end_time-loop_time))
            print ("  Avg iteration:\t%.3f s" % ((end_time-loop_time)/self.SS.sweep_grid_npts))
        if self.__use_temp:
            return tmp_results
        else:
            return results
    
    def paramSweepFunc(self, sweep_spec, ufcn, ufcn_args={}, get_vectors=False, sparse=False, sparselevels=6, sparsesolveropts={"sigma":None, "mode":"normal", "maxiter":None, "tol":1e-3, "which":"SA"}):
        self.params.ndSweep(sweep_spec)
        #self.eval_dp = True # Signal we want to calculate derived parameters
        
        # FIXME: Need a way to check if the Hamiltonian will need to be updated, this complicates things.
        
        self._presub()
        result = []
        func_result = []
        params = None
        E = None
        V = None
        if not get_vectors:
            if not sparse:
                for i in range(self.params.sweep_grid_npts):
                    params = dict([(k, v[i]) for k, v in self.params.sweep_grid_c.items()])
                    self._postsub(params)
                    E = self.getHamiltonian().eigenenergies()
                    result.append(E)
                    func_result.append(ufcn(self, params, E, **ufcn_args))
            else:
                H = None
                for i in range(self.params.sweep_grid_npts):
                    params = dict([(k, v[i]) for k, v in self.params.sweep_grid_c.items()])
                    self._postsub(params)
                    
                    # Solve
                    E = sc.sparse.linalg.eigsh(self.getHamiltonian().data, k=sparselevels, return_eigenvectors=False, **sparsesolveropts)
                    
                    # Sort the results
                    E.sort()
                    result.append(E)
                    func_result.append(ufcn(self, params, E, **ufcn_args))
        else:
            if not sparse:
                for i in range(self.params.sweep_grid_npts):
                    params = dict([(k, v[i]) for k, v in self.params.sweep_grid_c.items()])
                    self._postsub(params)
                    E, V = self.getHamiltonian().eigenstates()
                    result.append([E, V])
                    func_result.append(ufcn(self, params, E, V, **ufcn_args))
            else:
                H = None
                for i in range(self.params.sweep_grid_npts):
                    params = dict([(k, v[i]) for k, v in self.params.sweep_grid_c.items()])
                    self._postsub(params)
                    
                    # Solve
                    E, V = sc.sparse.linalg.eigsh(self.getHamiltonian().data, k=sparselevels, return_eigenvectors=True, **sparsesolveropts)
                    
                    # FIXME: Sort the results and convert eigenvectors to qutip 
                    result.append((E, V.T))
                    func_result.append(ufcn(self, params, E, V.T, **ufcn_args))
                    
        return np.array(result), np.array(func_result)
    
    ###################################################################################################################
    #       Internal Functions
    ###################################################################################################################
    
    # Initialises the sweep data
    def _init_sweep_data(self):
        self.sweep_specs = []
        self.evaluations = []
    
    # Replaces np asmatrix
    def _init_qobj_vector(self, obj_list, dtype=None):
        obj = np.empty((len(obj_list), 1) , dtype=dtype)
        for i, op in enumerate(obj_list):
            obj[i, 0] = op
        return obj
    
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













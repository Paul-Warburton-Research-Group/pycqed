import networkx as nx
import sympy as sy
import numpy as np
import copy
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from .parameters import ParamCollection

class SymbolicSystem(ParamCollection):
    
    # Mapping of the DoF structures
    __dof_map = {'flux':0,'charge':1,'disp':2,'disp_adj':3}
    
    def __init__(self, graph, dof_prefixes=["\\Phi", "\\phi", "Q", "q"], mode_transform=False, quiet=False):
        """
        """
        
        # Init the base class
        super().__init__([])
        
        # The CircuitGraph instance
        self.CG = graph
        if 0 not in self.CG.circuit_graph.nodes:
            raise Exception("CircuitGraph instance must include the ground node as 0.")
        
        # Transform the mode coordinates?
        self.use_transform = mode_transform
        
        # Get useful data before hand for efficiency
        self.Nn = len(self.CG.circuit_graph.nodes)-1 # Don't count the ground node
        self.Nb = len(self.CG.sc_spanning_tree_wc.edges)
        self.nodes = list(self.CG.circuit_graph.nodes)
        self.nodes.remove(0)
        # Note we use the spanning tree graph here as it's directional
        self.edges = list(self.CG.sc_spanning_tree_wc.edges)
        
        # Define physical constant symbols
        self.phi0 = sy.symbols("\\Phi_0") # Flux quantum
        self.Rq0 = sy.symbols("R_Q")      # Resistance quantum
        self.qcp = 2*sy.symbols("e")      # Cooper pair charge
        self.pi = sy.pi                   # Pi
        
        # Resonator structures
        self.resonator_symbols_expr = {}
        self.resonator_symbols_cap = {} # Capacitive, keyed by node
        self.resonator_symbols_ind = {} # Inductive, keyed by edge
        
        # Assign degree of freedom prefixes
        self.flux_prefix = dof_prefixes[0]
        self.redflux_prefix = dof_prefixes[1]
        self.charge_prefix = dof_prefixes[2]
        self.redcharge_prefix = dof_prefixes[3]
        
        # Flux and Charge bias terms
        self.flux_bias = {}
        self.flux_bias_prefactor = {}
        self.red_flux_bias = {}
        self.exp_flux_bias = {}
        self.charge_bias = {}
        self.red_charge_bias = {}
        
        # Flux and Charge bias names (name -> edge)
        self.flux_bias_names = {}
        self.charge_bias_names = {}
        
        # Create the circuit parameters
        self._create_circuit_symbols()
        self._create_flux_bias_symbols()
        self._create_charge_bias_symbols()
        
        # Get the resonator parameters
        if self._has_resonators():
            self._create_resonator_terms()
            self._create_loaded_resonator_parameters()
        
        # Get representation transformation matrices
        self._get_topology_matrices()
        
        # Get coordinate transformation matrices
        self._create_coordinate_transforms()
        if not quiet:
            print("Optimal basis representations for the circuit coordinates:")
            print(self.coordinate_modes)
        
        # Populate the degrees of freedom
        self._create_node_dofs()
        self._add_branch_dofs()
    
    #
    # TRANSFORM
    #
    def setTransform(self, V):
        # We require the transform for the flux coordinates in np.matrix format
        self.R = sy.Matrix(V)
        self.RT = self.R.T
        self.Rinv = self.R**(-1)
        self.RinvT = self.Rinv.T
        
        # Get the raw Linv
        self.use_transform = False
        Linv = self.getInverseInductanceMatrix()
        self.use_transform = True
        
        # Update the coordinate modes
        M = self.RinvT*Linv*self.Rinv
        for i in range(M.shape[0]):
            if M[i, i] != 0:
                self.coordinate_modes[self.nodes[i]] = "oscillator"
            else:
                self.coordinate_modes[self.nodes[i]] = "charge"
        print("Optimal basis representations for the circuit coordinates:")
        print(self.coordinate_modes)
        
        # Update the node DoFs
        self._create_node_dofs()
    
    #
    # CHARGE
    #
    def getChargeVector(self):
        return sy.Matrix([self.node_dofs[node][1] for node in self.nodes])
    
    def getVoltageVector(self):
        return sy.Matrix([sy.symbols("V_{%i}" % node) for node in self.nodes])
    
    def getChargeBiasVector(self, form="charge"):
        bias_vec = list(np.zeros(self.Nn))
        if form == "charge":
            for i, node in enumerate(self.nodes):
                bias_vec[i] = self.charge_bias[node]
        elif form == "phase":
            for i, node in enumerate(self.nodes):
                bias_vec[i] = self.red_charge_bias[node]
        return sy.Matrix(bias_vec)

    def getCapacitanceMatrix(self, parameterise=True):
        # WARN: Should be called only once in the parent class
        # First construct the matrix in the 'circuit basis'
        M = sy.eye(self.Nn) - sy.eye(self.Nn)

        # Diagonal
        for i, node in enumerate(self.nodes):
            sym = 0.0
            for edge in self.CG.circuit_graph.edges(node, keys=True):
                cstr = self.CG.components_map[edge]
                if cstr[0] == self.CG._element_prefixes[0]:
                    M[i, i] += self.circuit_params[cstr]

            # Add the gate capacitances
            if node in self.CG.charge_bias_nodes:
                if self.CG.charge_bias_nodes[node] is not None:
                    cstr = self.CG.charge_bias_nodes[node]
                    M[i, i] += self.circuit_params[cstr]

            # Add the resonator capacitances
            if self.CG.resonators_cap[node] is not None:
                cstr1 = self.CG.resonators_cap[node]['coupling']
                cstr2 = self.CG.resonators_cap[node]['Cr']
                Cc = self.circuit_params[cstr1]
                Cr = self.circuit_params[cstr2]

                # Effective capacitance due to the resonator
                M[i, i] += Cc*Cr/(Cc + Cr)

        # Off-diagonals
        for edge in self.CG.circuit_graph.edges: # Use circuit graph edges to get capacitors
            cstr = self.CG.components_map[edge]
            if cstr[0] == self.CG._element_prefixes[0]:
                if edge[0] == 0 or edge[1] == 0:
                    continue
                i = self.nodes.index(edge[0])
                j = self.nodes.index(edge[1])
                M[i, j] -= self.circuit_params[cstr]
                M[j, i] -= self.circuit_params[cstr]

        if not parameterise:
            return M

        # Parameterise the matrix elements
        for i in range(M.shape[0]):
            for j in range(i, M.shape[1]):
                if M[i, j] == 0 or len(M[i, j].free_symbols) < 2:
                    continue

                name = "C%i%i" % (i, j)
                self.addParameter(name)
                self.addParameterisation(name, M[i, j])
                M[i, j] = self.getSymbol(name)
                M[j, i] = self.getSymbol(name)

        return M

    def getInverseCapacitanceMatrix(self, parameterise=True):
        # Try to invert the inductance matrix as-is
        try:
            if self.use_transform:
                M = self.R*self.getCapacitanceMatrix(parameterise=parameterise)**(-1)*self.RT
            else:
                M = self.getCapacitanceMatrix(parameterise=parameterise)**(-1)
        except Exception:
            print("Capacitance matrix is singular, need at least one capacitor connected to every node.")
            raise
        
        # Check for complex infinities
        if M.has(sy.zoo):
            raise Exception("Capacitance matrix is singular, need at least one capacitor connected to every node.")
        
        if not parameterise:
            return M
        
        # Parameterise the matrix elements
        for i in range(M.shape[0]):
            for j in range(i, M.shape[1]):
                if M[i, j] == 0 or len(M[i, j].free_symbols) < 2:
                    continue
                
                name = "C%i%ii" % (i, j)
                self.addParameter(name)
                self.addParameterisation(name, M[i, j])
                M[i, j] = self.getSymbol(name)
                M[j, i] = self.getSymbol(name)
        
        return M
    
    def getSingleParticleChargingEnergies(self):
        ret = {}
        Cinv = self.getInverseCapacitanceMatrix()
        for i, node in enumerate(self.nodes):
            ret[node] = 0.5 * self.qcp**2 * Cinv[i,i]
        return ret
    
    #
    # FLUX
    #
    def getFluxVector(self, mode="node"):
        P = sy.Matrix([self.node_dofs[node][0] for node in self.nodes])
        if mode == "node":
            return P
        elif mode == "branch":
            return self.Rnb * P
    
    def getRedFluxVector(self, mode="node"):
        p = sy.Matrix([sy.symbols("%s_{%i}" % (self.redflux_prefix, node)) for node in self.nodes])
        if mode == "node":
            return p
        elif mode == "branch":
            return self.Rnb * p
    
    def getCurrentVector(self, mode="node"):
        return sy.Matrix([sy.symbols("I_{%i}" % node) for node in self.nodes])
    
    def moveFluxBias(self, orig_edge, new_edge):
        if orig_edge not in self.edges:
            raise Exception("Edge %s not part of the conductive circuit subgraph." % repr(orig_edge))
        if new_edge not in self.edges:
            raise Exception("Edge %s not part of the conductive circuit subgraph." % repr(new_edge))
        
        # Check the original edge has a flux bias term
        if self.flux_bias[orig_edge] == 0.0:
            raise Exception("Edge %s does not contain a flux bias term." % repr(orig_edge))
        
        # Check both edges are part of the same loop
        
        
        # FIXME: What if there are multiple bias terms on the edge?
        expr = self.flux_bias[orig_edge]
        self.flux_bias[orig_edge] = 0.0
        self.flux_bias[new_edge] = expr
        
        expr = self.red_flux_bias[orig_edge]
        self.red_flux_bias[orig_edge] = 0.0
        self.red_flux_bias[new_edge] = expr
        
        if self.CG.isInductiveEdge(new_edge):
            print("WARNING: Flux bias %s is on an inductive edge, and thus a suitable basis must be used." % (repr(expr)))
        print("Flux bias term %s is on edge %s (%s)." % (repr(expr), repr(new_edge), self.CG.components_map[new_edge]))
    
    def getFluxBiasVector(self, mode="node", form="flux"):
        bias_vec = list(np.zeros(self.Nb))
        if form == "flux":
            for i, edge in enumerate(self.edges):
                if self.CG.isJosephsonEdge(edge):
                    bias_vec[i] = self.flux_bias_prefactor[edge]*self.flux_bias[edge]
                else:
                    bias_vec[i] = 0.0
        elif form == "phase":
            for i, edge in enumerate(self.edges):
                if self.CG.isJosephsonEdge(edge):
                    bias_vec[i] = self.flux_bias_prefactor[edge]*self.red_flux_bias[edge]
                else:
                    bias_vec[i] = 0.0
        if mode == "node":
            return self.Rbn * sy.Matrix(bias_vec)
        elif mode == "branch":
            return sy.Matrix(bias_vec)
    
    def getFluxBiasMatrix(self, mode="node", form="flux"):
        return sy.diag(*self.getFluxBiasVector(mode=mode, form=form))
    
    def getFluxBiasVectorInd(self, mode="node"):
        bias_vec = list(np.zeros(self.Nb))
        for i, edge in enumerate(self.edges):
            if self.CG.isInductiveEdge(edge):
                bias_vec[i] = self.flux_bias_prefactor[edge]*self.flux_bias[edge]
            else:
                bias_vec[i] = 0.0
        
        # Remove the bias terms that would appear in the JJs in the node representation
        if mode == "node":
            return self.Rbn * sy.Matrix(bias_vec)
        elif mode == "branch":
            return sy.Matrix(bias_vec)
    
    def getInductanceMatrix(self, mode="node", parameterise=True):
        Mb = sy.eye(self.Nb) - sy.eye(self.Nb)
        
        # Diagonals
        for i, edge in enumerate(self.edges):
            if edge not in self.CG.flux_bias_edges:
                cstr = self.CG.components_map[edge]
                if cstr[0] == self.CG._element_prefixes[1]:
                    Mb[i, i] = self.circuit_params[cstr]
            else: # FIXME: Need to account for the case where the bias source inductance is not matched to the in-circuit inductance.
                cstr = self.CG.components_map[edge]
                if cstr[0] == self.CG._element_prefixes[1]:
                    L = self.circuit_params[cstr]
                    if self.CG.flux_bias_edges[edge] is None:
                        Mb[i, i] = L
                    else:
                        M = self.circuit_params[self.CG.flux_bias_edges[edge]]
                        Mb[i, i] = (L**2 - M**2)/L
        
        # Off-diagonals: always coupled branches
        for component, edges in self.CG.coupled_branches.items():
            edge1, edges2 = edges
            i = self.edges.index(edge1)
            j = self.edges.index(edge2)
            Mb[i, j] = self.circuit_params[component]
            Mb[j, i] = self.circuit_params[component]
        
        # Transform to node representation
        if mode == "node":
            return self.Rbn*Mb*self.Rnb
        elif mode == "branch":
            return Mb
    
    def getInverseInductanceMatrix(self, mode="node", parameterise=True):
        # Off-diagonals
        Mb = self.getInductanceMatrix(mode="branch", parameterise=parameterise)
        
        # Take the pseudo-inverse of the branch inductance matrix
        if self.use_transform:
            if mode == "node":
                return self.RinvT*self.Rbn*Mb.pinv()*self.Rnb*self.Rinv
            elif mode == "branch":
                return self.Rnb*self.RinvT*self.Rbn*Mb.pinv()*self.Rnb*self.Rinv*self.Rbn
        if mode == "node":
            return self.Rbn*Mb.pinv()*self.Rnb
        elif mode == "branch":
            return Mb.pinv()
    
    #
    # JOSEPHSON JUNCTIONS
    #
    def getJosephsonVector(self):
        vec = list(np.zeros(self.Nb))
        for i, edge in enumerate(self.edges):
            cstr = self.CG.components_map[edge]
            if cstr[0] == self.CG._element_prefixes[2]:
                vec[i] = self.circuit_params[cstr]
        return sy.Matrix(vec)
    
    def getJosephsonEnergies(self):
        ret = {}
        Jvec = self.getJosephsonVector()
        for i, edge in enumerate(self.edges):
            ret[edge] = self.phi0*Jvec[i]/(2*self.pi)
        return ret
    
    def getClassicalJosephsonEnergies(self,mode="node",sym_lev="lowest",as_equ=False):
        Jvec = self.getJosephsonVector().transpose()
        Jf = self.getRedFluxVector(mode="branch") + \
        self.getFluxBiasVector(mode="branch", form="phase")
        Jcos = sy.Matrix([sy.cos(e) for e in Jf])
        return -Jvec*Jcos
    
    def getQuantumJosephsonEnergies(self):
        # Get Josephson energies
        Jvec = self.getJosephsonVector()
        
        # Create the transformed branch vector
        Pp = self.Rnb*self.Rinv*self.node_vector
        
        # Create the exponential bias terms
        Pbias = self.getFluxBiasVector(mode="branch", form="phase")
        Pexp_p = sy.Matrix([sy.exp(1j*Pbias[i]) for i in range(self.Nb)])
        Pexp_m = sy.Matrix([sy.exp(-1j*Pbias[i]) for i in range(self.Nb)])
        
        # Construct the cosine terms in terms of displacement operators
        ret = 0
        for i, edge in enumerate(self.edges):
            if Jvec[i] == 0:
                continue
            
            prod1 = Pexp_p[i]
            prod2 = Pexp_m[i]
            if len(Pp[i].atoms()) > 2: # Case where there is sum of elements
                
                # Left
                for arg in Pp[i].args:
                    if arg.args[0] > 0:
                        prod1 *= self.jdisp_vector[arg.args[1]]
                    else:
                        prod1 *= self.jdisp_adj_vector[arg.args[1]]
                
                # Right
                for arg in Pp[i].args:
                    if arg.args[0] < 0:
                        prod2 *= self.jdisp_vector[arg.args[1]]
                    else:
                        prod2 *= self.jdisp_adj_vector[arg.args[1]]
            else: # Case where there is a single element
                if Pp[i].args[0] > 0:
                    prod1 *= self.jdisp_vector[Pp[i].args[1]]
                    prod2 *= self.jdisp_adj_vector[Pp[i].args[1]]
                else:
                    prod1 *= self.jdisp_adj_vector[Pp[i].args[1]]
                    prod2 *= self.jdisp_vector[Pp[i].args[1]]
        
            ret += -0.5*Jvec[i]*(prod1 + prod2)
        
        return sy.Matrix([ret])
    
    #
    # PHASESLIP NANOWIRES
    #
    def getPhaseSlipVector(self):
        vec = list(np.zeros(self.Nb))
        for i, edge in enumerate(self.edges):
            cstr = self.CG.components_map[edge]
            if cstr[0] == self.CG._element_prefixes[3]:
                vec[i] = self.circuit_params[cstr]
        return sy.Matrix(vec)
    
    def getPhaseSlipEnergies(self):
        pass
    
    def getQuantumPhaseSlipEnergies(self):
        # Get PhaseSlip energies
        Pvec = self.getPhaseSlipVector()
        
        # Create the transformed branch vector
        Pp = self.Rnb*self.Rinv*self.node_vector
        
        # Create the exponential bias terms in branch form
        Qbias = self.Rnb*self.getChargeBiasVector(form="phase")
        Qexp_p = sy.Matrix([sy.exp(1j*Qbias[i]) for i in range(self.Nb)])
        Qexp_m = sy.Matrix([sy.exp(-1j*Qbias[i]) for i in range(self.Nb)])
        
        # Construct the cosine terms in terms of displacement operators
        ret = 0
        for i, edge in enumerate(self.edges):
            if Pvec[i] == 0:
                continue
            
            prod1 = Qexp_p[i]
            prod2 = Qexp_m[i]
            if len(Pp[i].atoms()) > 2: # Case where there is sum of elements
                
                # Left
                for arg in Pp[i].args:
                    if arg.args[0] > 0:
                        prod1 *= self.pdisp_vector[arg.args[1]]
                    else:
                        prod1 *= self.pdisp_adj_vector[arg.args[1]]
                
                # Right
                for arg in Pp[i].args:
                    if arg.args[0] < 0:
                        prod2 *= self.pdisp_vector[arg.args[1]]
                    else:
                        prod2 *= self.pdisp_adj_vector[arg.args[1]]
            else: # Case where there is a single element
                if Pp[i].args[0] > 0:
                    prod1 *= self.pdisp_vector[Pp[i].args[1]]
                    prod2 *= self.pdisp_adj_vector[Pp[i].args[1]]
                else:
                    prod1 *= self.pdisp_adj_vector[Pp[i].args[1]]
                    prod2 *= self.pdisp_vector[Pp[i].args[1]]
        
            ret += -0.5*Pvec[i]*(prod1 + prod2)
        
        return sy.Matrix([ret])
    
    #
    # HAMILTONIAN
    #
    def getChargingEnergies(self):
        Q = self.getChargeVector()
        Qe = self.getChargeBiasVector()
        Cinv = self.getInverseCapacitanceMatrix()
        return 0.5 * (Q + Qe).transpose() * Cinv * (Q + Qe)
    
    def getFluxEnergies(self):
        P = self.getFluxVector()
        Pe = self.getFluxBiasVectorInd()
        Linv = self.getInverseInductanceMatrix()
        return 0.5 * (P + Pe).transpose() * Linv * (P + Pe)
        #return 0.5 * P.transpose() * Linv * P
    
    def getClassicalHamiltonian(self,mode="node",sym_lev="lowest",as_equ=False):
        return self.getChargingEnergies()\
        +self.getFluxEnergies()\
        +self.getClassicalJosephsonEnergies()
    
    def getQuantumHamiltonian(self):
        return self.getChargingEnergies()\
        +self.getFluxEnergies()\
        +self.getQuantumJosephsonEnergies()\
        +self.getQuantumPhaseSlipEnergies()
    
    #
    # INTERNAL
    #
    def _create_node_dofs(self):
        
        # Create node vector
        self.node_vector = sy.Matrix([sy.symbols("n_{%i}" % n) for n in self.nodes])
        self.node_map = {n: self.node_vector[i] for i, n in enumerate(self.nodes)}
        self.node_map_rev = {v: k for k, v in self.node_map.items()}
        #self.mode_vector = sy.Matrix([sy.symbols("m_{%i}" % n) for n in self.coordinate_modes.keys()])
        
        # Create mode vector
        self.node_dofs = {}
        self.classical_node_dofs = {}
        for node in self.CG.sc_spanning_tree_wc.nodes:
            self.node_dofs[node] = \
                    sy.symbols("%s_{%i} %s_{%i} %s_{%i} %s_{%i} %s_{%i} %s_{%i}" % \
                    (
                        self.flux_prefix, node,
                        self.charge_prefix, node,
                        "D", node,
                        "D^{\\dagger}", node,
                        "S", node,
                        "S^{\\dagger}", node
                    ), commutative=False)
            self.classical_node_dofs[node] = \
                    sy.symbols("%s_{%i} %s_{%i} %s_{%i}" % \
                    (
                        self.flux_prefix, node,
                        self.redflux_prefix, node,
                        self.charge_prefix, node
                    ))
        
        # Get Josephson displacement operator maps
        self.jdisp_vector = {self.node_vector[i]: self.node_dofs[n][2] for i, n in enumerate(self.nodes)}
        self.jdisp_adj_vector = {self.node_vector[i]: self.node_dofs[n][3] for i, n in enumerate(self.nodes)}
        
        # Get Josephson displacement operator partial charges resulting from mode transformation
        Pp = self.R * self.getFluxVector()
        self.cooper_disp = {}
        for i, node in enumerate(self.nodes):
            if len(Pp[i].atoms()) > 2: # Case where there is sum of elements
                self.cooper_disp[node] = abs(float(Pp[i].args[0].args[0]))
            else: # Case where there is a single element
                self.cooper_disp[node] = abs(float(Pp[i].args[0]))
        
        # Get Phase Slip displacement operator maps
        self.pdisp_vector = {self.node_vector[i]: self.node_dofs[n][4] for i, n in enumerate(self.nodes)}
        self.pdisp_adj_vector = {self.node_vector[i]: self.node_dofs[n][5] for i, n in enumerate(self.nodes)}
        
        # Get phase-slip displacement operator partial fluxons resulting from mode transformation
        Qp = self.RinvT * self.getChargeVector()
        self.fluxon_disp = {}
        for i, node in enumerate(self.nodes):
            if len(Qp[i].atoms()) > 2: # Case where there is sum of elements
                self.fluxon_disp[node] = abs(float(Qp[i].args[0].args[0]))
            else: # Case where there is a single element
                self.fluxon_disp[node] = abs(float(Qp[i].args[0]))
    
    # FIXME: Is this still required?
    def _add_branch_dofs(self):
        self.branch_dofs = {}
        self.classical_branch_dofs = {}
        for edge in self.CG.sc_spanning_tree_wc.edges:
            self.branch_dofs[edge] = \
                    sy.symbols("%s_{%i%i-%i} %s_{%i%i-%i} %s_{%i%i-%i} %s_{%i%i-%i}" % \
                    (
                        self.flux_prefix, edge[0], edge[1], edge[2],
                        self.charge_prefix, edge[0], edge[1], edge[2],
                        "D", edge[0], edge[1], edge[2],
                        "D^{\\dagger}", edge[0], edge[1], edge[2]
                    ), commutative=False)
            self.classical_branch_dofs[edge] = \
                    sy.symbols("%s_{%i%i-%i} %s_{%i%i-%i} %s_{%i%i-%i}" % \
                    (
                        self.flux_prefix, edge[0], edge[1], edge[2],
                        self.redflux_prefix, edge[0], edge[1], edge[2],
                        self.charge_prefix, edge[0], edge[1], edge[2]
                    ))
    
    def _add_loop_dofs(self):
        pass
    
    def _get_topology_matrices(self):
        # nx incidence matrix is Nn rows by Nb columns
        I = np.array(nx.incidence_matrix(self.CG.sc_spanning_tree_wc, oriented=True).todense()*-1)
        
        # Remove the row that corresponds to the ground node
        nodes = list(self.CG.sc_spanning_tree_wc.nodes)
        i = nodes.index(0)
        self.Rbn = sy.Matrix(np.delete(I, (i), axis=0)) # Branch to node
        self.Rnb = self.Rbn.T # Node to branch
    
    def _create_circuit_symbols(self):
        
        # Circuit components
        self.circuit_params = {} # FIXME: This attribute is now redundant, use the getSymbol function from parameters parent class.
        components = nx.get_edge_attributes(self.CG.circuit_graph, 'component').values()
        for component in components:
            self.circuit_params[component] = sy.symbols("%s_{%s}" % (component[0], component[1:]))
            self.addParameter(component)
        
        # Mutually coupled branches
        for component, edges in self.CG.coupled_branches.items():
            self.circuit_params[component] = sy.symbols("%s_{%s}" % (component[0], component[1:]))
            self.addParameter(component)
        
        # Charge bias capacitors
        for component in self.CG.charge_bias_nodes.values():
            if component is not None:
                self.circuit_params[component] = sy.symbols("%s_{%s}" % (component[0], component[1:]))
                self.addParameter(component)
        
        # Flux bias mutual inductors
        for component in self.CG.flux_bias_edges.values():
            if component is not None:
                self.circuit_params[component] = sy.symbols("%s_{%s}" % (component[0], component[1:]))
                self.addParameter(component)
    
    def _has_resonators(self):
        for node, resonator in self.CG.resonators_cap.items():
            if resonator is not None:
                return True
        return False
    
    def _create_resonator_terms(self):
        for node, resonator in self.CG.resonators_cap.items():
            if resonator is None:
                continue
            self.resonator_symbols_cap[node] = {}
            
            # Update the circuit parameters and create the symbols
            for k, var in resonator.items():
                self.circuit_params[var] = sy.symbols("%s_{%s}" % (var[0], var[1:]))
                self.resonator_symbols_cap[node][k] = self.circuit_params[var]
    
    def _create_loaded_resonator_parameters(self):
        
        # Make a copy of the circuit graph and add branches that correspond to resonators
        coupled_nodes = {}
        CG = copy.deepcopy(self.CG)
        CG.removeAllResonators()
        next_node = max(self.nodes) + 1
        for node, resonator in self.CG.resonators_cap.items():
            if resonator is None:
                continue
            CG.addBranch(node, next_node, resonator["coupling"])
            CG.addBranch(next_node, 0, resonator["Cr"])
            CG.addBranch(next_node, 0, resonator["Lr"])
            
            # Update the current parameter collection
            self.addParameters(*resonator.values())
            
            # Keep track of coupled branches for convenience
            coupled_nodes[node] = (node, next_node)
            next_node += 1
        Nn = len(CG.circuit_graph.nodes)-1
        nodes = list(CG.circuit_graph.nodes)
        nodes.remove(0)
        
        # Get the indices we'll use for getting the matrix elements
        coupled_indices = {node: (nodes.index(n[0]), nodes.index(n[1])) for node, n in coupled_nodes.items()}
        
        # Get the new system matrices
        SS = SymbolicSystem(CG, quiet=True)
        Cinv = SS.getInverseCapacitanceMatrix()
        Linv = SS.getInverseInductanceMatrix()
        
        # Get the coupling terms
        for node, indices in coupled_indices.items():
            
            # Get the matrix elements
            i, j = indices
            
            # Construct the resonator terms
            Ci_jj = SS.getParametricExpression(SS.getParameterFromSymbol(Cinv[j, j]), expand=True)
            Ci_ij = SS.getParametricExpression(SS.getParameterFromSymbol(Cinv[i, j]), expand=True)
            Li_jj = Linv[j, j]#SS.getParametricExpression(SS.getParameterFromSymbol(Linv[j, j]), expand=True)
            #Li_ij = SS.getParametricExpression(SS.getParameterFromSymbol(Linv[i, j]), expand=True)
            frd = sy.sqrt(Ci_jj * Li_jj)
            Zrd = sy.sqrt(Ci_jj / Li_jj)
            gC = Ci_ij / sy.sqrt(2*Zrd)
            self.resonator_symbols_expr[node] = {
                "gC": gC,
                "frl": frd,
                "Zrl": Zrd
            }
            
            # Add parameterisations
            fr = self.getSymbol(self.CG.resonators_cap[node]["fr"])
            Zr = self.getSymbol(self.CG.resonators_cap[node]["Zr"])
            
            # Determine resonator capacitance and inductance from the design resonant freq and impedance
            self.addParameterisation(self.CG.resonators_cap[node]["Cr"], 0.5/(np.pi * fr * Zr))
            self.addParameterisation(self.CG.resonators_cap[node]["Lr"], 0.5 * Zr/(np.pi * fr))
            
            # Get the loaded resonator parameters
            self.addParameterisation(self.CG.resonators_cap[node]["frl"], frd)
            self.addParameterisation(self.CG.resonators_cap[node]["Zrl"], Zrd)
            
            # Get the coupling term
            self.addParameterisation(self.CG.resonators_cap[node]["gC"], gC)

    def _create_flux_bias_symbols(self):
        # First add empty flux bias placeholders
        for edge in self.edges:
            self.flux_bias_prefactor[edge] = 1.0
            self.flux_bias[edge] = 0.0
            self.red_flux_bias[edge] = 0.0
            self.exp_flux_bias[edge] = 1.0

        # Iterate through the user selected flux biased edges
        for edge in self.CG.flux_bias_edges:
            self.flux_bias[edge] += sy.symbols("%s_{%i%i-%ie}" % (self.flux_prefix, edge[0], edge[1], edge[2]))
            self.flux_bias_names["phi%i%i-%ie"%edge] = edge
            self.red_flux_bias[edge] += sy.symbols("%s_{%i%i-%ie}" % (self.redflux_prefix, edge[0], edge[1], edge[2]))
            self.exp_flux_bias[edge] *= sy.symbols("e^{i%s_{%i%i-%ie}}" % (self.redflux_prefix, edge[0], edge[1], edge[2]))
            self.addParameter(
                "phi%i%i-%ie" % edge,
                sy.symbols("%s_{%i%i-%ie}" % (self.flux_prefix, edge[0], edge[1], edge[2]))
            )
            print("Flux bias term %s is on edge %s (%s)." % ("phi%i%i-%ie" % edge, repr(edge), self.CG.components_map[edge]))

    def _create_charge_bias_symbols(self):
        # First add empty charge bias placeholders
        for node in self.nodes:
            self.charge_bias[node] = 0.0
            self.red_charge_bias[node] = 0.0

        # Iterate through the user selected charge biased nodes
        for node in self.CG.charge_bias_nodes:
            self.charge_bias[node] = sy.symbols("%s_{%ie}" % (self.charge_prefix, node))
            self.charge_bias_names["%s%ie" % (self.charge_prefix, node)] = node
            self.red_charge_bias[node] = sy.symbols("%s_{%ie}" % (self.redcharge_prefix, node))
            self.addParameter(
                "%s%ie" % (self.charge_prefix, node),
                sy.symbols("%s_{%ie}" % (self.charge_prefix, node))
            )

    def _create_coordinate_transforms(self):
        self.coordinate_modes = {}
        
        # Find oscillator modes
        if self.use_transform:
            # Get the inverse inductance matrix in node representation
            self.use_transform = False
            Linv = self.getInverseInductanceMatrix()
            self.use_transform = True
            
            # Get the transformation matrices
            V, D = Linv.diagonalize()
            if V.free_symbols != set():
                print("Warning: A suitable transformation couldn't be identified automatically. Set one manually using the setTransform function.")
                self.use_transform = False
                self._create_coordinate_transforms()
                return
            
            self.R = V
            self.RT = V.T
            self.Rinv = V**(-1)
            self.RinvT = self.Rinv.T
            
            # Update the coordinate modes
            M = self.RinvT*Linv*self.Rinv
            for i in range(M.shape[0]):
                if M[i, i] != 0:
                    self.coordinate_modes[self.nodes[i]] = "oscillator"
                else:
                    self.coordinate_modes[self.nodes[i]] = "charge"
        else:
            Linv = self.getInverseInductanceMatrix()
            
            # Create the identity transforms
            M = sy.Matrix(np.diag([1.0]*Linv.shape[0]))
            self.R = M
            self.RT = M
            self.Rinv = M
            self.RinvT = M
            
            all_indices = set(range(Linv.shape[0]))
            ch_indices = set()
            index = 0
            while ch_indices != all_indices:
                if Linv[index, index] == 0:
                    self.coordinate_modes[self.nodes[index]] = "charge"
                    ch_indices.add(index)
                    index += 1
                    continue
                
                # Check the row
                # FIXME: This is too harsh: block diagonal matrices are actually acceptable
                # as couplings between oscillator modes are allowed.
                coupled = False
                for i in range(index+1, Linv.shape[0]):
                    if Linv[index, i] != 0:
                        coupled = True
                        self.coordinate_modes[self.nodes[i]] = "charge"
                        ch_indices.add(i)
                if coupled:
                    self.coordinate_modes[self.nodes[index]] = "charge"
                else:
                    self.coordinate_modes[self.nodes[index]] = "oscillator"
                ch_indices.add(index)
                index += 1
                















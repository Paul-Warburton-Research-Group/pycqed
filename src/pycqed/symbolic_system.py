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
    
    def __init__(self, graph, dof_prefixes=["\\Phi", "\\phi", "Q", "q"]):
        """
        """
        
        # Init the base class
        super().__init__([])
        
        # The CircuitGraph instance
        self.CG = graph
        if 0 not in self.CG.circuit_graph.nodes:
            raise Exception("CircuitGraph instance must include the ground node as 0.")
        
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
        
        # Populate the degrees of freedom
        self._add_node_dofs()
        self._add_branch_dofs()
        
        # Create the circuit parameters
        self._create_circuit_symbols()
        self._create_flux_bias_symbols()
        self._create_charge_bias_symbols()
        
        # Get the resonator parameters
        if self._has_resonators():
            self._create_resonator_terms()
            self._create_loaded_resonator_parameters()
        
        # Get transformation matrices
        self._get_topology_matrices()
    
    #
    # PARAMETERS
    #
    def getParameterDict(self):
        ret = self.circuit_params.copy()
        ret.update(dict([(k,self.flux_bias[v]) for k,v in self.flux_bias_names.items()]))
        ret.update(dict([(k,self.charge_bias[v]) for k,v in self.charge_bias_names.items()]))
        return ret
    
    def getDoFSymbol(self, pos, dof_type):
        index = self.__dof_map[dof_type]
        return self.node_dofs[pos][index]
    
    def getDoFSymbolAsMatrix(self, pos, dof_type, n):
        return sy.MatrixSymbol(self.getDoFSymbol(pos, dof_type), n, n)
    
    def getDoFSymbolList(self, dof_type):
        return {node: self.getDoFSymbol(node, dof_type) for node in self.nodes}
    
    #
    # TRANSFORM
    #
    
    
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
            raise Exception("Charge bias phase not implemented.")
            
            # Implement this when loop charges are implemented
        return sy.Matrix(bias_vec)
    
    def getCapacitanceMatrix(self):
        M = sy.eye(self.Nn) - sy.eye(self.Nn)
        
        # Diagonal
        for i, node in enumerate(self.nodes):
            sym = 0.0
            for edge in self.CG.circuit_graph.edges(node, keys=True):
                cstr = self.CG.components_map[edge]
                if cstr[0] == self.CG._element_prefixes[0]:
                    M[i, i] += self.circuit_params[cstr]
            
            # Add the gate capacitances
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
        
        # FIXME: Parameterisations of the matrix elements here will likely lead to significant performance boost when inverting
        
        return M
    
    def getInverseCapacitanceMatrix(self):
        # Try to invert the inductance matrix as-is
        return self.getCapacitanceMatrix()**(-1)
        
        # This should NOT be done
        # Failing that, take the pseudo-inverse of the node capacitance matrix
        # and transform that if required
        #return self.getCapacitanceMatrix().pinv()
    
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
    
    def getFluxBiasVector(self, mode="node", form="flux"):
        bias_vec = list(np.zeros(self.Nb))
        if form == "flux":
            for i,edge in enumerate(self.edges):
                bias_vec[i] = self.flux_bias_prefactor[edge]*self.flux_bias[edge]
        elif form == "phase":
            for i,edge in enumerate(self.edges):
                bias_vec[i] = self.flux_bias_prefactor[edge]*self.red_flux_bias[edge]
        elif form == "expphase":
            for i,edge in enumerate(self.edges):
                bias_vec[i] = self.exp_flux_bias[edge]
        if mode == "node":
            return self.Rbn * sy.Matrix(bias_vec)
        elif mode == "branch":
            return sy.Matrix(bias_vec)
    
    def getFluxBiasMatrix(self, mode="node", form="flux"):
        return sy.diag(*self.getFluxBiasVector(mode=mode, form=form))
    
    def getInductanceMatrix(self, mode="node"):
        Mb = sy.eye(self.Nb) - sy.eye(self.Nb)
        
        # Diagonals
        for i, edge in enumerate(self.edges):
            cstr = self.CG.components_map[edge]
            if cstr[0] == self.CG._element_prefixes[1]:
                Mb[i, i] = self.circuit_params[cstr]
        
        # Off-diagonals
        
        
        # Transform to node representation
        if mode == "node":
            return self.Rbn*Mb*self.Rnb
        elif mode == "branch":
            return Mb
    
    def getInverseInductanceMatrix(self, mode="node"):
        Mb = sy.eye(self.Nb) - sy.eye(self.Nb)
        
        # Diagonals
        for i, edge in enumerate(self.edges):
            cstr = self.CG.components_map[edge]
            if cstr[0] == self.CG._element_prefixes[1]:
                Mb[i, i] = self.circuit_params[cstr]
        
        # Off-diagonals
        
        
        # Take the pseudo-inverse of the branch inductance matrix
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
    
    def getLeftDecompFluxVector(self):
        return (self.Rnb + np.abs(self.Rnb))*self.getFluxVector()/2
    
    def getRightDecompFluxVector(self):
        return (self.Rnb - np.abs(self.Rnb))*self.getFluxVector()/2
    
    def getLeftDisplacementOpMatrix(self, adjoint=False, as_vec=False):
        vec = list(np.zeros(self.Nb))
        ind2 = 2 if not adjoint else 3
        ind1 = 3 if not adjoint else 2

        # Use left decomposed flux vector
        lv = self.getLeftDecompFluxVector()
        rv = self.getRightDecompFluxVector()
        for i, edge in enumerate(self.edges):
            l = lv[i]
            r = rv[i]
            if l == 0:
                vec[i] = 1
            else:
                if r.args != ():
                    if l.args[0] == 1 and r.args[0] == -1:
                        vec[i] = self.node_dofs[edge[0]][ind2]
                else:
                    vec[i] = self.node_dofs[edge[0]][ind2]
        if as_vec:
            return sy.Matrix(vec)
        else:
            return sy.diag(*vec)
    
    def getRightDisplacementOpVector(self, adjoint=False):
        vec = list(np.zeros(self.Nb))
        ind2 = 2 if not adjoint else 3
        ind1 = 3 if not adjoint else 2


        # Use right decomposed flux vector
        lv = self.getLeftDecompFluxVector()
        rv = self.getRightDecompFluxVector()
        for i, edge in enumerate(self.edges):
            l = lv[i]
            r = rv[i]
            if r == 0:
                vec[i] = 1
            else:
                if l.args != ():
                    if l.args[0] == 1 and r.args[0] == -1:
                        vec[i] = self.node_dofs[edge[1]][ind1]
                else:
                    vec[i] = self.node_dofs[edge[1]][ind2]
        return sy.Matrix(vec)
    
    def getClassicalJosephsonEnergies(self,mode="node",sym_lev="lowest",as_equ=False):
        Jvec = self.getJosephsonVector().transpose()
        Jf = self.getRedFluxVector(mode="branch") + \
        self.getFluxBiasVector(mode="branch", form="phase")
        Jcos = sy.Matrix([sy.cos(e) for e in Jf])
        return -Jvec*Jcos
    
    def getQuantumJosephsonEnergies(self):
        Dl = self.getLeftDisplacementOpMatrix()
        Dld = self.getLeftDisplacementOpMatrix(adjoint=True)
        Dr = self.getRightDisplacementOpVector()
        Drd = self.getRightDisplacementOpVector(adjoint=True)
        pb = self.getFluxBiasMatrix(mode="branch", form="expphase")
        pbd = self.getFluxBiasMatrix(mode="branch", form="expphase").conjugate()
        J = self.getJosephsonVector().transpose()
        return -J*0.5*(pb*Dl*Dr + pbd*Dld*Drd)
    
    #
    # PHASESLIP NANOWIRES
    #
    
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
        Pe = 0.0#self.getFluxBiasVector()
        Linv = self.getInverseInductanceMatrix()
        #return 0.5 * (P + Pe).transpose() * Linv * (P + Pe)
        return 0.5 * P.transpose() * Linv * P
    
    def getClassicalHamiltonian(self,mode="node",sym_lev="lowest",as_equ=False):
        return self.getChargingEnergies()\
        +self.getFluxEnergies()\
        +self.getClassicalJosephsonEnergies()
    
    def getQuantumHamiltonian(self):
        return self.getChargingEnergies()\
        +self.getFluxEnergies()\
        +self.getQuantumJosephsonEnergies()
    
    #
    # INTERNAL
    #
    def _add_node_dofs(self):
        self.node_dofs = {}
        self.classical_node_dofs = {}
        for node in self.CG.sc_spanning_tree_wc.nodes:
            self.node_dofs[node] = \
                    sy.symbols("%s_{%i} %s_{%i} %s_{%i} %s_{%i}" % \
                    (
                        self.flux_prefix, node,
                        self.charge_prefix, node,
                        "D", node,
                        "D^{\\dagger}", node
                    ), commutative=False)
            self.classical_node_dofs[node] = \
                    sy.symbols("%s_{%i} %s_{%i} %s_{%i}" % \
                    (
                        self.flux_prefix, node,
                        self.redflux_prefix, node,
                        self.charge_prefix, node
                    ))
    
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
    
    def _create_circuit_symbols(self):
        
        # Circuit components
        self.circuit_params = {}
        components = nx.get_edge_attributes(self.CG.circuit_graph, 'component').values()
        for component in components:
            self.circuit_params[component] = sy.symbols("%s_{%s}" % (component[0], component[1:]))
            self.addParameter(component)
        
        # Charge bias capacitors
        for component in self.CG.charge_bias_nodes.values():
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
        SS = SymbolicSystem(CG)
        Cinv = SS.getInverseCapacitanceMatrix()
        Linv = SS.getInverseInductanceMatrix()
        
        # Get the coupling terms
        for node, indices in coupled_indices.items():
            i, j = indices
            frd = sy.sqrt(Cinv[j, j]*Linv[j, j])
            Zrd = sy.sqrt(Cinv[j, j]/Linv[j, j])
            gC = Cinv[i, j]/sy.sqrt(2*Zrd)
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
            
    
    def _get_topology_matrices(self):
        # nx incidence matrix is Nn rows by Nb columns
        I = np.array(nx.incidence_matrix(self.CG.sc_spanning_tree_wc, oriented=True).todense()*-1)
        
        # Remove the row that corresponds to the ground node
        nodes = list(self.CG.sc_spanning_tree_wc.nodes)
        i = nodes.index(0)
        self.Rbn = sy.Matrix(np.delete(I, (i), axis=0)) # Branch to node
        self.Rnb = self.Rbn.T # Node to branch
    
    def _create_flux_bias_symbols(self):
        for edge in self.edges:
            self.flux_bias_prefactor[edge] = 1.0
            self.flux_bias[edge] = 0.0
            self.red_flux_bias[edge] = 0.0
            self.exp_flux_bias[edge] = 1.0
            
            if edge in self.CG.closure_branches or (edge[1], edge[0], edge[2]) in self.CG.closure_branches:
                self.flux_bias[edge] = sy.symbols("%s_{%i%i-%ie}" % (self.flux_prefix, edge[0], edge[1], edge[2]))
                
                self.flux_bias_names["phi%i%i-%ie"%edge] = edge
                self.red_flux_bias[edge] = sy.symbols("%s_{%i%i-%ie}" % (self.redflux_prefix, edge[0], edge[1], edge[2]))
                self.exp_flux_bias[edge] = sy.symbols("e^{i%s_{%i%i-%ie}}" % (self.redflux_prefix, edge[0], edge[1], edge[2]))
                
                self.addParameter(
                    "phi%i%i-%ie" % edge,
                    sy.symbols("%s_{%i%i-%ie}" % (self.flux_prefix, edge[0], edge[1], edge[2]))
                )
    
    def _create_charge_bias_symbols(self):
        for node in self.nodes:
            if self.CG.charge_bias_nodes[node] is None:
                self.charge_bias[node] = 0.0
            else:
                self.charge_bias[node] = sy.symbols("%s_{%ie}" % (self.charge_prefix, node))
                self.charge_bias_names["%s%ie" % (self.charge_prefix, node)] = node
                
                self.addParameter(
                    "%s%ie" % (self.charge_prefix, node),
                    sy.symbols("%s_{%ie}" % (self.charge_prefix, node))
                )
        
        #    self.red_charge_bias[edge] = 1.0

















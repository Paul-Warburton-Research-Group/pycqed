""" The :py:mod:`pycqed.src.isingmodel` module defines the class :class:`IsingGraph` which is used to generate arbitrary Ising Hamiltonians from a given graph.
"""
import time
import numpy as np
import networkx as nx
import qutip as qt
import itertools as itt
import matplotlib as mpl
import matplotlib.pyplot as plt

from . import util
from . import units
from . import dataspec as ds

class IsingGraph:
    """ Class to convert a series of graphs to an Ising Hamiltonian.
    """
    
    # Graph visual properties
    __node_color = 'C0'
    __node_size = 3000
    __edge_color = 'k'
    __edge_size = 2
    __label_font_size = 16
    __label_font_weight = "normal"
    __paulis = {
        "i": qt.qeye(2),
        "x": qt.sigmax(),
        "y": qt.sigmay(),
        "z": qt.sigmaz()
    }
    __node_labels = [
        "node_name",
        "paulis"
    ]
    
    def __init__(self, graph_family=None):
        
        self.graph_family = {}                          # The family of graphs
        self.coefs_of_order = {}                        # The data structure used to generate any Ising H
        self.node_labels = {}                           # The labels for the nodes
        self.edge_labels = {}                           # The labels for the edges
        self.node_indices = {}                          # node -> index
        self.edge_indices = {}                          # edge -> index
        self.node_labels_style = "node_name"            # What to use as labels for the nodes
        self.E = []
        self.V = []
        self.tmax = 0.0
        
        # Set the graph family right away if specified
        if graph_family is not None:
            self.graph_family = graph_family
            
            # Generate the required terms
            for k, G in graph_family.items():
                self.setNthOrderGraph(G, k)
        
    def setNthOrderGraph(self, Gn, n):
        """ Sets a graph for the n-th order coupling terms of the Ising Hamiltonian. The graph for orders 1 and 2 should be the same, and any higher order graph can be the same or different. Also initialises the corresponding coefficients to the default 0 value.
        """
        if n <= 0:
            raise Exception("Specified order 'n' should be and integer greater than 0.")
        if not isinstance(Gn, nx.Graph):
            raise Exception("Specified graph should be a networkx.Graph instance.")
        
        nodes = list(Gn.nodes)
        edges = list(Gn.edges)
        
        # Order 1 and 2 are handled at the same time
        if n == 1 or n == 2:
            self.N_qubits = len(self.graph_family[1].nodes)
            self.graph_family[1] = Gn
            self.graph_family[2] = Gn
            self.node_indices = {nodes[i]:i for i in range(len(nodes))}
            self.edge_indices[2] = {
                (
                    self.node_indices[edges[i][0]],
                    self.node_indices[edges[i][1]]
                ):i for i in range(len(edges))
            }
            
            # Generate the coefficient symbols
            terms1 = ["h"+"".join(x) for x in itt.product(['x','y','z'], repeat=(1))]
            terms2 = ["J"+"".join(x) for x in itt.product(['x','y','z'], repeat=(2))]
            
            # Init the coefs data structure for these
            coefs1 = {k:0.0 for k in terms1}
            coefs2 = {k:0.0 for k in terms2}
            
            # Assign for each node and edge
            self.coefs_of_order[1] = {(i,):coefs1.copy() for k,i in self.node_indices.items()}
            self.coefs_of_order[2] = {e:coefs2.copy() for e,i in self.edge_indices[2].items()}
            
        else:
            self.graph_family[n] = Gn
            
            # Get the cycles of order n
            cycles = self._find_cycles_of_length(n)
            index_cycles = [[self.node_indices[node] for node in c] for c in cycles]
            self.edge_indices[n] = {tuple(cycles[i]):i for i in range(len(cycles))}
            
            # Generate the coefficient symbols
            terms = ["J"+"".join(x) for x in itt.product(['x','y','z'], repeat=(n))]
            
            # Init the coefs data structure for these
            coefs = {k:0.0 for k in terms}
            
            # Init the coefs data structure
            self.coefs_of_order[n] = {e:coefs.copy() for e,i in self.edge_indices[n].items()}
    
    def getNthOrderGraph(self, n):
        """ Gets the n-th order graph.
        """
        if n <= 0:
            raise Exception("Specified order 'n' should be an integer greater than 0.")
        return self.graph_family[n]
    
    def setNthOrderCoefs(self, n, params):
        """ Sets the N local parameters of the Ising Hamiltonian.
        """
        if n <= 0:
            raise Exception("Specified order 'n' should be an integer greater than 0.")
        
        # Check the node and edges exist
        in_set = set(params.keys())
        total_set = set(self.coefs_of_order[n].keys())
        if not in_set <= total_set:
            raise Exception("Nodes or edges specified in 'params' do not exist.")
        
        # Check the parameters exist for each node and edge specified
        if n == 1:
            terms = ["h"+"".join(x) for x in itt.product(['x','y','z'], repeat=(n))]
        else:
            terms = ["J"+"".join(x) for x in itt.product(['x','y','z'], repeat=(n))]
        total_set = set(terms)
        for key, val in params.items():
            in_set = set(val.keys())
            if not in_set <= total_set:
                raise Exception("Parameter name for node or edge '%s' does not exist." % (repr(key)))
        
        # Update the parameters for each coordinate specified
        for coord in params.keys():
            self.coefs_of_order[n][coord].update(params[coord])
    
    def getNthOrderCoefs(self, n):
        """ Gets the N local parameters of the Ising Hamiltonian.
        """
        if n <= 0:
            raise Exception("Specified order 'n' should be an integer greater than 0.")
        return self.coefs_of_order[n]
    
    def getNthOrderTerm(self, term):
        """ Gets specific N local parameters of the Ising Hamiltonian.
        """
        return {k:v[term] for k,v in self.getNthOrderCoefs(len(term[1:])).items()}
    
    def setCoef(self, coord, term, value):
        """ Sets a given term in the full Hamiltonian
        """
        if type(coord) in [int, np.int64]:
            self.coefs_of_order[1][(coord,)][term] = value
        elif type(coord) in [list, tuple]:
            self.coefs_of_order[len(coord)][tuple(coord)][term] = value
    
    def getCoef(self, coord, term):
        """ Gets a given term in the full Hamiltonian
        """
        if type(coord) in [int, np.int64]:
            return self.coefs_of_order[1][(coord,)][term]
        elif type(coord) in [list, tuple]:
            return self.coefs_of_order[len(coord)][tuple(coord)][term]
    
    def getOperator(self, coord, term):
        """ Gets the operator associated with the specified coordinate and term.
        """
        kron_pos = [self.__paulis["i"]]*self.N_qubits
        if type(coord) in [int, np.int64]:
            kron_pos[coord] = self.__paulis[term[1]]
        elif type(coord) in [list, tuple]:
            for i, p in enumerate(coord):
                kron_pos[p] = self.__paulis[term[i+1]]
        return qt.tensor(*kron_pos)
    
    def getBitString(self, spinbasis='z', state=0, args={}):
        """ Generates a bit string that indicates the direction of spins for a given state. '0' is down and '1' is up.
        """
        if len(self.V) == 0:
            self._diagonalise_hamiltonian(args=args)
        s = ["0"]*self.N_qubits
        for i in range(self.N_qubits):
            s[i] = str(int((1+(self.V[state].overlap(self.getOperator(i, 'h'+spinbasis)*self.V[state])).real)/2))
        return "".join(s)
    
    def getHamiltonian(self):
        """ Generates the final Hamiltonian from the supplied graphs.
        """
        
        # Handle units here? (use pycqed.Units)
        factor = 2*np.pi
        
        Hti = 0.0
        Htd = []
        kron_pos = [self.__paulis["i"]]*self.N_qubits
        for order, coefs in self.coefs_of_order.items():
            for pos, coef in coefs.items():
                for sym, val in coef.items():
                    if val == 0.0:
                        continue
                    kron_pos = [self.__paulis["i"]]*self.N_qubits
                    for i,p in enumerate(pos):
                        kron_pos[p] = self.__paulis[sym[i+1]]
                    if callable(val):
                        Htd.append([factor*qt.tensor(*kron_pos), val])
                    else:
                        Hti += val*qt.tensor(*kron_pos)
        if Htd == []:
            return factor*Hti
        if type(Hti) is qt.Qobj:
            return [factor*Hti, *Htd]
        return [*Htd]
    
    def drawNthOrderGraph(self, n):
        """ Draws the specified graph.
        """
        if n <= 0:
            raise Exception("Specified order 'n' should be an integer greater than 0.")
        if n not in self.graph_family.keys():
            raise Exception("Specified order 'n=%i' does not have an associated graph." % n)
        
        # Make our own figure
        fig, ax = plt.subplots(1,1,constrained_layout=True,figsize=(7,7))
        
        # Get layout first as it is randomly generated
        L = nx.spring_layout(self.graph_family[n])
        
        self._update_node_labels()
        nx.draw(
            self.graph_family[n],
            pos=L,
            with_labels=True,
            node_color=self.__node_color,
            node_size=self.__node_size,
            width=self.__edge_size,
            labels=self.node_labels,
            font_weight=self.__label_font_weight,
            font_size=self.__label_font_size,
            ax=ax
        )
        
        self._update_edge_labels()
        nx.draw_networkx_edge_labels(
            self.graph_family[n],
            L,
            edge_labels=self.edge_labels,
            font_weight=self.__label_font_weight,
            font_size=self.__label_font_size,
            ax=ax
        )
    
    def getNodes(self):
        """ Gets the node names of the graph.
        """
        return list(self.graph_family[1].nodes)
    
    def getEdges(self, n=1):
        """ Gets the edge names of the graph.
        """
        if n == 1 or n == 2:
            return list(self.graph_family[1].edges)
        else:
            return list(self.coefs_of_order[n].keys())
    
    def getEigenEnergies(self, args={}):
        """ Get the eigenenergies of the Hamiltonian
        """
        if len(self.E) == 0:
            self._diagonalise_hamiltonian(args=args)
        return self.E
    
    def getEigenVectors(self, args={}):
        """ Get the eigenvectors of the Hamiltonian
        """
        if len(self.V) == 0:
            self._diagonalise_hamiltonian(args=args)
        return self.V
    
    ###################################################################################################################
    #       Internal Functions
    ###################################################################################################################
    
    def _diagonalise_hamiltonian(self, args={}):
        H = self.getHamiltonian()
        if type(H) is qt.Qobj:
            Hq = H
        else:
            Hq = qt.QobjEvo(H, args=args)(self.tmax)
        self.E, self.V = util.diagDenseH(Hq, eigvalues=Hq.shape[0], get_vectors=True)
    
    def _update_node_labels(self):
        
        # Get nodes
        nodes = self.getNodes()
        
        # Get parameters
        if self.node_labels_style == "node_name":
            for node in nodes:
                self.node_labels[node] = str(node)
        elif self.node_labels_style == "paulis":
            for node in nodes:
                s = ""
                for k,v in self.coefs_of_order[1][(node,)].items():
                    s += "%s=%.1f\n" % (k,v)
                self.node_labels[node] = s.strip('\n')
    
    def _update_edge_labels(self):
        
        # Get edges
        edges = self.getEdges()
        
        # Get parameters
        if self.node_labels_style == "node_name":
            for edge in edges:
                self.edge_labels[edge] = "(%i, %i)" % (edge[0],edge[1])
        elif self.node_labels_style == "paulis":
            pass
    
    def _find_cycles_of_length(self, n, source=None, cycle_length_limit=None):
        """Adapted from https://gist.github.com/joe-jordan/6548029
        
        To make this more efficient:
         - Need a way to stop looking for longer cycles than set by n
         - More effectively filter out the cycles of length n
         - Go from nodes to their indices
        
        """
        G = self.graph_family[n]
        if source is None:
            # produce edges for all components
            nodes=[list(i)[0] for i in nx.connected_components(G)]
        else:
            # produce edges for components with source
            nodes=[source]
        
        # extra variables for cycle detection:
        cycle_stack = []
        output_cycles = set()
        
        def get_hashable_cycle(cycle):
            """cycle as a tuple in a deterministic order."""
            m = min(cycle)
            mi = cycle.index(m)
            mi_plus_1 = mi + 1 if mi < len(cycle) - 1 else 0
            if cycle[mi-1] > cycle[mi_plus_1]:
                result = cycle[mi:] + cycle[:mi]
            else:
                result = list(reversed(cycle[:mi_plus_1])) + list(reversed(cycle[mi_plus_1:]))
            return tuple(result)
        
        for start in nodes:
            if start in cycle_stack:
                continue
            cycle_stack.append(start)
            
            stack = [(start,iter(G[start]))]
            while stack:
                parent,children = stack[-1]
                try:
                    child = next(children)
                    
                    if child not in cycle_stack:
                        cycle_stack.append(child)
                        stack.append((child,iter(G[child])))
                    else:
                        i = cycle_stack.index(child)
                        if i < len(cycle_stack) - 2:
                          output_cycles.add(get_hashable_cycle(cycle_stack[i:]))
                    
                except StopIteration:
                    stack.pop()
                    cycle_stack.pop()
        
        unique_cycles = list(np.unique([sorted(list(i)) for i in output_cycles]))
        if type(unique_cycles[0]) in [int, np.int64]:
            return [unique_cycles]
        return [x for x in unique_cycles if len(x) == n]
    

class QuantumAnnealing:
    """ Class to describe a generic quantum annealing process.
    """
    
    def __init__(self, Hp, Hd, args, Hc=None, schedules=[]):
        
        # Master equation solver options
        self.solver = qt.mesolve
        self.me_options = qt.Options(nsteps=1000000000)
        
        # QobjEvo Hamiltonian
        self.Htot = None
        
        # Keep IsingGraph instances
        self.Hp = Hp
        self.Hd = Hd
        self.Hc = Hc
        
        # Determine the type of the Hamiltonians and construct the final annealing hamiltonian
        Htot_l = []
        if schedules != []:
            sA, sB, sC = schedules
        else:
            sA = A
            sB = B
            sC = C
        Htot_l.extend(self._format_hamiltonian(Hd, sA))
        Htot_l.extend(self._format_hamiltonian(Hp, sB))
        Htot_l.extend(self._format_hamiltonian(Hc, sC))
        self.Htot = qt.QobjEvo(Htot_l, args=args)
        
        # Get useful constants
        self.N_qubits = Hp.N_qubits
        self.neigs = 2**self.N_qubits
        
        # Get driver and problem eigenstates
        # FIXME: Need to decide what solver to use here, sparse or dense.
        Hdq = self.Htot(0.0)
        Hpq = self.Htot(args["tan"])
        if self.N_qubits <= 9:
            self.Ed, self.Vd = util.diagDenseH(Hdq, eigvalues=self.neigs, get_vectors=True)
            self.Ep, self.Vp = util.diagDenseH(Hpq, eigvalues=self.neigs, get_vectors=True)
            
            # Object to pass to dynamics solvers
            self.Hobject = self.Htot
            
            # Initial state for dynamics solvers
            self.Vinit = self.Vd[0]
            
            self.eigensolver = util.diagDenseH
            
        else:
            print("WARNING: Number of qubits greater than 9. Only solving for initial state.")
            # Limit number of eigenvalues considered
            #self.neigs = int(np.log(2**self.N_qubits)*self.N_qubits)
            
            # Estimate lower bounds ground state energy and set sigma to less than that
            sigmaD = None#-self.N_qubits * self.Hd.getCoef(0, 'hx') * 3*np.pi
            sigmaP = None#min([Hpq.data[k,k] for k in range(Hpq.shape[0])])*1.5
            which = "SA" # Get algebraicly small values
            
            # Sparse solve
            self.Ed, self.Vd = util.diagSparseH(Hdq, eigvalues=1, get_vectors=True, sparsesolveropts={'sigma':sigmaD, 'which':which})
            #self.Ep, self.Vp = util.diagSparseH(Hpq, eigvalues=1, get_vectors=True, sparsesolveropts={'sigma':sigmaP, 'which':which})
            
            # Object to pass to dynamics solvers
            self.Hobject = self.Htot
            
            # Initial state for dynamics solvers
            self.Vinit = self.Vd[0]
            
            self.eigensolver = util.diagSparseH
        
    
    def setSolverOptions(self, solver, **kwargs):
        """ Configures the solver options.
        """
        self.solver = solver
        for k, v in kwargs.items():
            setattr(self.me_options, k, v)
    
    def setSolverInitialState(self, state):
        """ Sets the initial state to use with solvers.
        """
        self.Vinit = state
    
    def setSolverObject(self, obj):
        """ Sets the object to pass to the solver.
        """
        self.Hobject = obj
    
    def getHammingWeightOp(self):
        """ Build the Hamming Weight operator associated with the problem Hamiltonian.
        """
        nodes = self.Hp.getNodes()
        sz0 = self.Hp.getOperator(nodes[0], 'hz')
        I = qt.qeye(2**self.N_qubits)
        I.dims = sz0.dims
        HW = I - I
        for n in nodes:
            HW += 0.5*(I - self.Hp.getOperator(n, 'hz'))
        return HW
    
    def getHamiltonian(self, t):
        """ Gets the Hamiltonian at time t.
        """
        return self.Htot(t)
    
    def annealInstantaneousGap(self, tan_pts, sparams={}, gaps=None, timesweep=False, rev=False):
        """ Runs an annealing schedule and gets the instantaneous transition energies.
        """
        # Time initialisation
        if timesweep:
            init_time = time.time()
        
        # Update args
        if sparams:
            self.Htot.arguments(sparams)
        
        if gaps is None:
            gaps = self.neigs-1 # Get all gaps
        
        # Handle units here? (use pycqed.Units)
        factor = 2*np.pi
        
        Egaps = np.array([[0.0]*len(tan_pts)]*(gaps+1))
        Hs = None
        
        if timesweep:
            loop_time = time.time()
        for i,t in enumerate(tan_pts):
            Ei = util.diagDenseH(self.Htot(t), eigvalues=(gaps+1)) if not rev else util.diagDenseH(self.Htot(t), eigvalues=(gaps+1))[::-1]
            for j in range(gaps+1):
                Egaps[j,i] = Ei[j]-Ei[0]
        
        # Report timings
        if timesweep:
            end_time = time.time()
            print ("Sweep Duration:")
            print ("  Initialization:\t%.3f s" % (loop_time-init_time))
            print ("  Loop duration:\t%.3f s" % (end_time-loop_time))
            print ("  Avg iteration:\t%.3f s" % ((end_time-loop_time)/len(tan_pts)))
        return Egaps/factor
    
    def annealInstantaneousLevels(self, tan_pts, sparams={}, levels=None, timesweep=False, rev=False):
        """ Runs an annealing schedule and gets the instantaneous energy levels.
        """
        # Time initialisation
        if timesweep:
            init_time = time.time()
        
        # Update args
        if sparams:
            self.Htot.arguments(sparams)
        
        if levels is None:
            levels = self.neigs # Get all levels
        
        # Handle units here? (use pycqed.Units)
        factor = 2*np.pi
        
        Elev = np.array([[0.0]*len(tan_pts)]*levels)
        Hs = None
        
        if timesweep:
            loop_time = time.time()
        for i,t in enumerate(tan_pts):
            Ei = util.diagDenseH(self.Htot(t), eigvalues=levels) if not rev else util.diagDenseH(self.Htot(t), eigvalues=levels)[::-1]
            for j in range(levels):
                Elev[j,i] = Ei[j]
        
        # Report timings
        if timesweep:
            end_time = time.time()
            print ("Sweep Duration:")
            print ("  Initialization:\t%.3f s" % (loop_time-init_time))
            print ("  Loop duration:\t%.3f s" % (end_time-loop_time))
            print ("  Avg iteration:\t%.3f s" % ((end_time-loop_time)/len(tan_pts)))
        return Elev/factor
    
    def annealInstantaneousStates(self, tan_pts, sparams={}, states=5, timesweep=False, rev=False, sparse_opts={}):
        """ Runs an annealing schedule and gets the instantaneous states.
        """
        # Time initialisation
        if timesweep:
            init_time = time.time()
        
        # Update args
        if sparams:
            self.Htot.arguments(sparams)
        
        E = np.array([[0.0]*len(tan_pts)]*states)
        V = []
        if timesweep:
            loop_time = time.time()
        for i,t in enumerate(tan_pts):
            Ei,Vi = self.eigensolver(self.Htot(t), eigvalues=states, get_vectors=True, sparsesolveropts=sparse_opts)
            V.append(Vi)
            for j in range(states):
                E[j,i] = Ei[j]
        
        # Report timings
        if timesweep:
            end_time = time.time()
            print ("Sweep Duration:")
            print ("  Initialization:\t%.3f s" % (loop_time-init_time))
            print ("  Loop duration:\t%.3f s" % (end_time-loop_time))
            print ("  Avg iteration:\t%.3f s" % ((end_time-loop_time)/len(tan_pts)))
        return E, V
    
    def annealInstantaneousExpectations(self, tan_pts, operator, sparams={}, timesweep=False, rev=False):
        """ Runs an annealing schedule and gets the instantaneous expectation value of the given operator.
        """
        # Time initialisation
        if timesweep:
            init_time = time.time()
        
        # Update args
        if sparams:
            self.Htot.arguments(sparams)
        
        res = []
        if timesweep:
            loop_time = time.time()
        for i,t in enumerate(tan_pts):
            Ei,Vi = util.diagDenseH(self.Htot(t), eigvalues=1, get_vectors=True)
            res.append(qt.expect(operator,Vi))
        
        # Report timings
        if timesweep:
            end_time = time.time()
            print ("Sweep Duration:")
            print ("  Initialization:\t%.3f s" % (loop_time-init_time))
            print ("  Loop duration:\t%.3f s" % (end_time-loop_time))
            print ("  Avg iteration:\t%.3f s" % ((end_time-loop_time)/len(tan_pts)))
        return res
    
    def annealMinimumGaps(self, tan_pts, Egaps):
        """ Finds where the minimum instantaneous transition energies occur in normalised time.
        """
        minima = [0]*(len(Egaps)-1)
        for i,Epts in enumerate(Egaps):
            # Discard first point (no gap)
            if i == 0:
                continue
            
            # Find minimum index and save xy point
            j = np.argmin(Epts)
            minima[i-1] = [tan_pts[j],Epts[j]]
        return np.array(minima)
    
    def annealInstantaneousOverlaps(self, tpts, sparams={}, timeit=False, coefs=5, signed=False, signed_state=0, timesweep=False):
        """ Gets coefficients of each eigenstate in terms of the problem solution state
        """
        
        # Time initialisation
        if timesweep:
            init_time = time.time()
        
        # Update args
        if sparams:
            self.Htot.arguments(sparams)
        
        # Get anneal states
        ci2 = np.array([[0.0j]*len(tpts)]*coefs)
        Ei = None
        Vi = None
        Hs = None
        
        if timesweep:
            loop_time = time.time()
        mul = 1.0
        for i,t in enumerate(tpts):
            Ei,Vi = util.diagDenseH(self.Htot(t), eigvalues=self.neigs, get_vectors=True)
            
            # Fix the sign of the selected state
            if np.sign(self.Vp[0].overlap(Vi[signed_state])) > 0.0:
                mul = 1.0
            else:
                mul = -1.0
            
            # Project
            for j,Vik in enumerate(Vi[:coefs]):
                ci = self.Vp[0].overlap(Vik)
                ci2[j,i] = mul*np.sign(ci)*np.absolute(ci)**2 if signed else np.absolute(ci)**2
        
        # Report timings
        if timesweep:
            end_time = time.time()
            print ("Sweep Duration:")
            print ("  Initialization:\t%.3f s" % (loop_time-init_time))
            print ("  Loop duration:\t%.3f s" % (end_time-loop_time))
            print ("  Avg iteration:\t%.3f s" % ((end_time-loop_time)/len(tpts)))
        return ci2
    
    def annealInstantaneousPopulations(self, tan_pts, sparams={}, timeit=False, coefs=5, signed=False, timesweep=False):
        """ Gets coefficients of each eigenstate in terms of the problem solution state
        """
        
        # Time initialisation
        if timesweep:
            init_time = time.time()
        
        # Update args
        if sparams:
            self.Htot.arguments(sparams)
        
        # Get anneal states
        ci2 = np.array([[0.0j]*len(tan_pts)]*coefs)
        Ei = None
        Vi = None
        Hs = None
        
        if timesweep:
            loop_time = time.time()
        for i,t in enumerate(tan_pts):
            Ei,Vi = util.diagDenseH(self.Htot(t), eigvalues=self.neigs, get_vectors=True)
            
            # Project
            for j,Vpk in enumerate(self.Vp[:coefs]):
                ci = Vi[0].overlap(Vpk)
                ci2[j,i] = np.sign(ci)*np.absolute(ci)**2 if signed else np.absolute(ci)**2
        
        # Report timings
        if timesweep:
            end_time = time.time()
            print ("Sweep Duration:")
            print ("  Initialization:\t%.3f s" % (loop_time-init_time))
            print ("  Loop duration:\t%.3f s" % (end_time-loop_time))
            print ("  Avg iteration:\t%.3f s" % ((end_time-loop_time)/len(tan_pts)))
        return ci2
    
    def annealStateProbability(self, tan_pts, sparams={}, states=None, timesweep=False, **kwargs):
        """ Finds the probabilities of finding the states of the system as a function of total annealing time.
        """
        
        # Time initialisation
        if timesweep:
            init_time = time.time()
        
        if states is None:
            states = self.neigs # Get all states
        
        # Update args
        if sparams:
            self.Htot.arguments(sparams)
            sparams_l = sparams.copy()
        else:
            sparams_l = self.Htot.args.copy()
        
        # Solve the time dependence
        p = np.array([np.zeros(len(tan_pts))]*states)
        if timesweep:
            loop_time = time.time()
        for i,tan in enumerate(tan_pts):
            times = np.array([0, tan]) # Solve for final time only, initial time must be specified too
            
            # Use time dependent Hamiltonian and specify initial state
            sparams_l["tan"] = tan
            result = self.solver(self.Hobject, self._get_initial_state(), times, args=sparams_l, options=self.me_options, **kwargs)
            
            # Save relevant results
            if self.solver == qt.mesolve:
                for j in range(states):
                    p[j,i] = qt.expect(result.states[-1], self.Vp[j])
            else:
                for j in range(states):
                    p[j,i] = qt.expect(qt.ket2dm(result.states[-1]), self.Vp[j])
        
        # Report timings
        if timesweep:
            end_time = time.time()
            print ("Sweep Duration:")
            print ("  Initialization:\t%.3f s" % (loop_time-init_time))
            print ("  Loop duration:\t%.3f s" % (end_time-loop_time))
            print ("  Avg iteration:\t%.3f s" % ((end_time-loop_time)/len(tan_pts)))
        return p
    
    def numberOfRuns(self, prob_tan, psuccess=0.99):
        """ Finds the number of runs required to meet the given success probability.
        """
        return np.log(1.0 - psuccess)/np.log(1.0 - np.array(prob_tan))
    
    def timeToSolution(self, tan_pts, prob_tan, psuccess=0.99, N=1, Nmax=1):
        """ Finds the time to solution given a desired probability.
        """
        return tan_pts*self.numberOfRuns(prob_tan,psuccess=psuccess)*N/Nmax
    
    def annealOperatorExpectations(self, tan_pts, operators, sparams={}, timesweep=False, **kwargs):
        """ Finds the expectation values of the given operators as a function of total annealing time.
        """
        
        # Time initialisation
        if timesweep:
            init_time = time.time()
        
        # Update args
        if sparams:
            self.Htot.arguments(sparams)
            sparams_l = sparams.copy()
        else:
            sparams_l = self.Htot.args.copy()
        
        # Solve the time dependence
        exp = np.array([np.zeros(len(tan_pts))]*len(operators))
        if timesweep:
            loop_time = time.time()
        for i,tan in enumerate(tan_pts):
            times = np.array([0, tan]) # Solve for final time only, initial time must be specified too
            
            # Use time dependent Hamiltonian and specify initial state
            sparams_l["tan"] = tan
            result = self.solver(self.Hobject, self._get_initial_state(), times, e_ops=operators, args=sparams_l, options=self.me_options, **kwargs)
            
            # Save relevant results
            for j in range(len(operators)):
                exp[j,i] = result.expect[j]
        
        # Report timings
        if timesweep:
            end_time = time.time()
            print ("Sweep Duration:")
            print ("  Initialization:\t%.3f s" % (loop_time-init_time))
            print ("  Loop duration:\t%.3f s" % (end_time-loop_time))
            print ("  Avg iteration:\t%.3f s" % ((end_time-loop_time)/len(tan_pts)))
        return exp
    
    def evolveStateProbability(self, t_pts, sparams={}, states=None, timesweep=False, **kwargs):
        """ Finds the evolution of the states during an anneal.
        """
        
        # Time initialisation
        if timesweep:
            init_time = time.time()
        
        if states is None:
            states = self.Hobject(0).shape[0] # Get all states
        
        # Update args
        if sparams:
            self.Htot.arguments(sparams)
            sparams_l = sparams.copy()
        else:
            sparams_l = self.Htot.args.copy()
        
        # Solve the time dependence
        p = np.array([np.zeros(len(t_pts))]*states)
        if timesweep:
            loop_time = time.time()
        result = self.solver(self.Hobject, self._get_initial_state(), t_pts, args=sparams_l, options=self.me_options, **kwargs)
        
        # Save relevant results
        if self.solver == qt.mesolve:
            for i in range(len(t_pts)):
                for j in range(states):
                    p[j,i] = qt.expect(result.states[i], self.Vp[j])
        else:
            for i in range(len(t_pts)):
                for j in range(states):
                    p[j,i] = qt.expect(qt.ket2dm(result.states[i]), self.Vp[j])
        
        # Report timings
        if timesweep:
            end_time = time.time()
            print ("Sweep Duration:")
            print ("  Initialization:\t%.3f s" % (loop_time-init_time))
            print ("  Loop duration:\t%.3f s" % (end_time-loop_time))
            print ("  Avg iteration:\t%.3f s" % ((end_time-loop_time)/len(t_pts)))
        return p
    
    def evolveStates(self, t_pts, sparams={}, states=None, timesweep=False, **kwargs):
        """ Finds the evolution of the states during an anneal.
        """
        
        # Time initialisation
        if timesweep:
            init_time = time.time()
        
        if states is None:
            states = self.Htot(0).shape[0] # Get all states
        
        # Update args
        if sparams:
            self.Htot.arguments(sparams)
            sparams_l = sparams.copy()
        else:
            sparams_l = self.Htot.args.copy()
        
        # Solve the time dependence
        p = np.array([np.zeros(len(t_pts))]*states)
        if timesweep:
            loop_time = time.time()
        result = self.solver(self.Hobject, self._get_initial_state(), t_pts, args=sparams_l, options=self.me_options, **kwargs)
        
        # Report timings
        if timesweep:
            end_time = time.time()
            print ("Sweep Duration:")
            print ("  Initialization:\t%.3f s" % (loop_time-init_time))
            print ("  Loop duration:\t%.3f s" % (end_time-loop_time))
            print ("  Avg iteration:\t%.3f s" % ((end_time-loop_time)/len(t_pts)))
        
        return result.states
    
    def evolveOperatorExpectations(self, t_pts, operators, sparams={}, timesweep=False, **kwargs):
        """ Finds the expectation values of the given operators during an anneal.
        """
        
        # Time initialisation
        if timesweep:
            init_time = time.time()
        
        # Update args
        if sparams:
            self.Htot.arguments(sparams)
            sparams_l = sparams.copy()
        else:
            sparams_l = self.Htot.args.copy()
        
        # Solve the time dependence
        exp = np.array([np.zeros(len(t_pts))]*len(operators))
        if timesweep:
            loop_time = time.time()
        result = self.solver(self.Hobject, self._get_initial_state(), t_pts, e_ops=operators, args=sparams_l, options=self.me_options, **kwargs)
        
        # Save relevant results
        #for j in range(len(operators)):
        #    exp[j] = result.expect[j]
        
        # Report timings
        if timesweep:
            end_time = time.time()
            print ("Sweep Duration:")
            print ("  Initialization:\t%.3f s" % (loop_time-init_time))
            print ("  Loop duration:\t%.3f s" % (end_time-loop_time))
            print ("  Avg iteration:\t%.3f s" % ((end_time-loop_time)/len(t_pts)))
        return result.expect
    
    ###################################################################################################################
    #       Internal Functions
    ###################################################################################################################
    
    def _format_hamiltonian(self, H, schedule):
        Hl = []
        if H is None:
            return Hl
        H_qobj = H.getHamiltonian()
        if type(H_qobj) is qt.Qobj:
            Hl.append([H_qobj, schedule])
        elif type(H_qobj) is list:
            if type(H_qobj[0]) is qt.Qobj:
                Hl.append([H_qobj[0], schedule])
                Hl.extend(H_qobj[1:])
            elif type(H_qobj[0]) is float:
                Hl.extend(H_qobj[1:])
            else:
                return H_qobj
        return Hl
    
    def _get_initial_state(self):
        # Prep initial state
        if self.solver == qt.mesolve:
            return self.Vinit*self.Vinit.dag()
        elif self.solver == qt.sesolve or self.solver == qt.mcsolve:
            return self.Vinit
    
    def _prepare_evolution(self, tpts=None):
        
        # Annealing H
        #if tpts is not None:
        #    self.Htot.tlist = tpts
        Hd = self.Htot(0)
        Hp = self.Htot(self.Htot.args['tan'])
        
        # Get initial ground state
        Ed,Vd = util.diagDenseH(Hd, eigvalues=Hd.shape[0], get_vectors=True)
        
        # Get final ground state
        Ep,Vp = util.diagDenseH(Hp, eigvalues=Hp.shape[0], get_vectors=True)
        
        # Prep initial state
        if self.solver == qt.mesolve:
            Vinit = Vd[0]*Vd[0].dag()
        elif self.solver == qt.sesolve or self.solver == qt.mcsolve:
            Vinit = Vd[0]
        
        return Vinit, Vp

class AnnealingSchedule:
    """ Class to describe annealing schedules.
    """
    
    def __init__(self):
        self.start_time = 0.0               # The default start time of the anneal
        self.end_time = 1.0                 # The detault end time of the anneal
        self.A = None
        self.B = None
        self.C = None
        
    
    # Transverse Field Schedules
    def A(t,p):
        if p["stype"] == "linear":
            if t < 0:
                return 1.0
            elif t >= 0 and t <= p["tan"]:
                return (1.0-t/p["tan"])
            else:
                return 0.0
        elif p["stype"] == "hsine":
            if t < 0:
                return 0.0
            elif t >= 0 and t <= p["tan"]:
                return np.sin(t/p["tan"]*np.pi)**p["dist"]
            else:
                return 0.0
        elif p["stype"] == "interf":
            if t < 0:
                return 1.0
            elif t >= 0 and t/p["tan"] < p["smg"]+p["sw"]/2:
                return 1.0-t/p["tan"]
            elif t/p["tan"] >= p["smg"]+p["sw"]/2 and t/p["tan"] < p["smg"]+1.5*p["sw"]:
                return t/p["tan"]-(p["smg"]+p["sw"]/2)*2+1.0
            elif t/p["tan"] >= p["smg"]+1.5*p["sw"] and t/p["tan"] <= p["smg"]+1.5*p["sw"]+1-p["smg"]+p["sw"]/2:
                return -t/p["tan"] + (p["smg"]+1.5*p["sw"]+1-p["smg"]+p["sw"]/2)
            elif t/p["tan"] > p["smg"]+1.5*p["sw"]+1-p["smg"]+p["sw"]/2:
                return 0.0

    # Problem Field Schedules
    def B(t,p):
        if p["stype"] == "linear":
            if t < 0:
                return 0.0
            elif t >= 0 and t <= p["tan"]:
                return (t/p["tan"])
            else:
                return 1.0
        elif p["stype"] == "hsine":
            if t < 0:
                return 1.0
            elif t >= 0 and t <= p["tan"]:
                return 1-np.sin(t/p["tan"]*np.pi)
            else:
                return 1.0
        elif p["stype"] == "interf":
            if t < 0:
                return 0.0
            elif t >= 0 and t/p["tan"] < p["smg"]+p["sw"]/2:
                return t/p["tan"]
            elif t/p["tan"] >= p["smg"]+p["sw"]/2 and t/p["tan"] < p["smg"]+1.5*p["sw"]:
                return -t/p["tan"]+(p["smg"]+p["sw"]/2)*2
            elif t/p["tan"] >= p["smg"]+1.5*p["sw"] and t/p["tan"] <= p["smg"]+1.5*p["sw"]+1-p["smg"]+p["sw"]/2:
                return 1+t/p["tan"] - (p["smg"]+1.5*p["sw"]+1-p["smg"]+p["sw"]/2)
            elif t/p["tan"] > p["smg"]+1.5*p["sw"]+1-p["smg"]+p["sw"]/2:
                return 1.0

    # Catalyst Field Schedules
    def C(t,p):
        if p["stype"] == "linear":
            if t < 0:
                return 0.0
            elif t >= 0 and t <= p["tan"]:
                return (1.0-t/p["tan"])*t/p["tan"]
            else:
                return 0.0


class QuantumComputation:
    """ Class to describe a generic quantum annealing process.
    """
    
    def __init__(self):
        pass



# Transverse Field Schedules
def A(t,p):
    if p["stype"] == "linear":
        if t < 0:
            return 1.0
        elif t >= 0 and t <= p["tan"]:
            return (1.0-t/p["tan"])
        else:
            return 0.0
    elif p["stype"] == "hsine":
        if t < 0:
            return 0.0
        elif t >= 0 and t <= p["tan"]:
            return np.sin(t/p["tan"]*np.pi)**p["dist"]
        else:
            return 0.0
    elif p["stype"] == "interf":
        if t < 0:
            return 1.0
        elif t >= 0 and t/p["tan"] < p["smg"]+p["sw"]/2:
            return 1.0-t/p["tan"]
        elif t/p["tan"] >= p["smg"]+p["sw"]/2 and t/p["tan"] < p["smg"]+1.5*p["sw"]:
            return t/p["tan"]-(p["smg"]+p["sw"]/2)*2+1.0
        elif t/p["tan"] >= p["smg"]+1.5*p["sw"] and t/p["tan"] <= p["smg"]+1.5*p["sw"]+1-p["smg"]+p["sw"]/2:
            return -t/p["tan"] + (p["smg"]+1.5*p["sw"]+1-p["smg"]+p["sw"]/2)
        elif t/p["tan"] > p["smg"]+1.5*p["sw"]+1-p["smg"]+p["sw"]/2:
            return 0.0

# Problem Field Schedules
def B(t,p):
    if p["stype"] == "linear":
        if t < 0:
            return 0.0
        elif t >= 0 and t <= p["tan"]:
            return (t/p["tan"])
        else:
            return 1.0
    elif p["stype"] == "hsine":
        if t < 0:
            return 1.0
        elif t >= 0 and t <= p["tan"]:
            return 1-np.sin(t/p["tan"]*np.pi)
        else:
            return 1.0
    elif p["stype"] == "interf":
        if t < 0:
            return 0.0
        elif t >= 0 and t/p["tan"] < p["smg"]+p["sw"]/2:
            return t/p["tan"]
        elif t/p["tan"] >= p["smg"]+p["sw"]/2 and t/p["tan"] < p["smg"]+1.5*p["sw"]:
            return -t/p["tan"]+(p["smg"]+p["sw"]/2)*2
        elif t/p["tan"] >= p["smg"]+1.5*p["sw"] and t/p["tan"] <= p["smg"]+1.5*p["sw"]+1-p["smg"]+p["sw"]/2:
            return 1+t/p["tan"] - (p["smg"]+1.5*p["sw"]+1-p["smg"]+p["sw"]/2)
        elif t/p["tan"] > p["smg"]+1.5*p["sw"]+1-p["smg"]+p["sw"]/2:
            return 1.0

# Catalyst Field Schedules
def C(t,p):
    if p["stype"] == "linear":
        if t < 0:
            return 0.0
        elif t >= 0 and t <= p["tan"]:
            return (1.0-t/p["tan"])*t/p["tan"]
        else:
            return 0.0


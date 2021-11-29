""" The :py:mod:`pycqed.src.circuitspec` module defines the classes :class:`CircuitSpec` and :class:`SubCircuitSpec`.

The class :class:`CircuitSpec` is used to draw a circuit and build it's constitutive equations. From the circuit diagram, a graph is built in which the edges have associated circuit components. The charge, flux and Josephson components of the total energy Hamiltonian are then generated symbolically so that they can be inspected and used as is by the user. It is possible to then define subcircuits derived from an instance of this class.

The class :class:`SubCircuitSpec` is used to partition a larger circuit into subcircuits, which include loading effects from other neighbouring circuits. It effectively has the same functionality as the :class:`CircuitSpec` class but differs in the sense that it's constitutive equations are statically defined and it is not possible to add branches and components. This class is designed such that it can be passed to an instance of :class:`HamilSpec` and will behave in exactly the same way as when an instance of :class:`CircuitSpec` is passed.
"""
import SchemDraw as schem
import SchemDraw.elements as e
import networkx as nx
import numpy as np
import sympy as sy
import matplotlib.pyplot as plt
import io
import copy

from . import text2latex as t2l
from . import units as un


class CircuitSpec:
    """This class is used to draw a circuit and build it's constitutive equations. From the circuit diagram, a graph is built in which the edges have associated circuit components. The charge, flux and Josephson components of the total energy Hamiltonian are then generated symbolically so that they can be inspected and used as is by the user. It is possible to then define subcircuits derived from an instance of this class."""
    
    # Class variables for direction control of parallel components
    __dir_map = {'right':0,'up':1,'left':2,'down':3}
    __dir_arr = ['right','up','left','down']
    
    # Circuit representations
    __modes = ["node","branch","trans"]
    
    # Symbolic representation levels
    __sym_lev = ['highest','lowest']
    
    # Mapping of the DoF structures
    __dof_map = {'flux':0,'charge':1,'disp':2,'disp_adj':3}
    
    # Bias types
    __bias_types = ["charge","flux"]
    
    # Flux operator forms
    __flux_op_form = ["flux","phase","expphase"]
    
    # Charge operator forms
    __charge_op_form = ["charge","phase"]
    
    # Subsystem coupling types
    __cpl_types = ["capacitive","inductive","galvanic"]
    
    # Subsystem types
    __subsys_types = ["linear_resonator","nonlinear_resonator"]
    
    #
    #
    #
    def __init__(self, circuit_name, elem_prefixes=["C","L","I","M"], dof_prefixes=["\\Phi", "\\phi", "Q", "q"], gnd_node_clr="blue", circ_node_clr="red"):
        
        # Save args for making copies
        self.kwargs = [elem_prefixes, dof_prefixes, gnd_node_clr, circ_node_clr]
        
        # Init data structures
        self.circ_name = circuit_name
        self.graph = nx.DiGraph(circuit_name=self.circ_name) # TODO: Think of using MultiDiGraph to allow more than two branches between nodes
        self.drawing = schem.Drawing()
        
        # Tracker variables
        self.ncap = 0                           # Total number of capacitors
        self.nind = 0                           # Total number of inductors
        self.njjs = 0                           # Total number of JJs
        self.nmut = 0                           # Total number of mutual inductors
        self.schem_code = []                    # Holds all elements necessary to regenerate the circuit as is
        self.schem_code_block = []              # Block of drawing code associated with edge (temporary)
        self.schem_code_blocks = {}             # Holds all blocks necessary to regenerate/edit circuit edges
        self.node_circ_coords = {}              # Holds the circuit diagram coordinates of the nodes
        self.code_elems = 0                     # Counts the number of SchemDraw elements
        self.code_lines = 0
        
        # Circuit parameter symbols
        self.circ_params = {}
        
        # Assign element prefixes
        self.cap_prefix = elem_prefixes[0]
        self.ind_prefix = elem_prefixes[1]
        self.jj_prefix = elem_prefixes[2]
        self.mut_prefix = elem_prefixes[3]
        
        # Assign degree of freedom prefixes
        self.flux_prefix = dof_prefixes[0]
        self.redflux_prefix = dof_prefixes[1]
        self.charge_prefix = dof_prefixes[2]
        self.redcharge_prefix = dof_prefixes[3]
        
        # Define physical constant symbols
        self.phi0 = sy.symbols("\\Phi_0") # Flux quantum
        self.Rq0 = sy.symbols("R_Q")      # Resistance quantum
        self.qcp = 2*sy.symbols("e")      # Cooper pair charge
        self.pi = sy.pi                   # Pi
        
        # Degrees of freedom
        self.node_dofs = {}
        self.classical_node_dofs = {}
        self.branch_dofs = {}
        self.classical_branch_dofs = {}
        
        # Flux and Charge bias terms (edge -> symbol mapping)
        self.flux_bias = {}
        self.flux_bias_prefactor = {}
        self.red_flux_bias = {}
        self.exp_flux_bias = {}
        self.charge_bias = {}
        self.red_charge_bias = {}
        
        # Flux and Charge bias names (name -> edge)
        self.flux_bias_names = {}
        self.charge_bias_names = {}
        
        # Flux and Charge bias components
        self.flux_bias_mutuals = {}
        self.charge_bias_caps = {}
        
        # Node properties (circuit diagram)
        self.circ_node_clr = circ_node_clr
        self.gnd_node_clr = gnd_node_clr
        
        # Coupled subsystems
        self.coupled_subsys = {}
        
        # Extended circuit
        self.rcircuit = None
        
        # Extended circuit loaded system matrices
        self.rcircuit_sys_matrices = {}
        
        # Subcircuit defined by hierarchize function
        self.subcircuits = {}
        self.subcircuit_coupling_edges = []
        
        # Add the first ground node
        self.addSchemElem(e.GND)
        self.pushSchemDraw()
        start_node = self.addSchemElem(e.DOT,lftlabel="0",color=self.gnd_node_clr)
        self.graph.add_node(0,ntype="Ground")
        self.node_circ_coords[0] = start_node.end
        
        # Add this code to the full code
        self.schem_code.extend(self.schem_code_block)
        
    #
    #
    #
    def showCircuitStats(self):
        print ("Total Component Count:")
        print ("  Capacitors: %i" % self.ncap)
        print ("  Inductors: %i" % self.nind)
        print ("  Josephson Junctions: %i" % self.njjs)
        print ("  Mutual Inductances: %i" % self.nmut)
        print ("")
        print ("Graph Properties:")
        print ("  Nodes: %i" % (len(list(self.graph.nodes))))
        print ("  Edges: %i" % len(list(self.graph.edges)))
        print ("  Spanning Branches: %i" % len(self.getSpanningBranches()))
        print ("  Closure Branches: %i" % len(self.getClosureBranches()))
        print ("")
        print ("Circuit Properties:")
        print ("  Irreducible Loops: %i" % len(self.getIrreducibleLoops()))
        print ("  Superconducting Loops: %i" % len(self.getSuperconductingLoops()))
        print ("  Flux Bias Lines: %i" % len([v for v in list(self.flux_bias.values()) if v != 0.0]))
        print ("  Charge Bias Lines: %i" % len([v for v in list(self.charge_bias.values()) if v != 0.0]))
    
    ###################################################################################################################
    #       Circuit Graph Functions
    ###################################################################################################################
    
    #
    #
    #
    def getClosureBranches(self):
        edges = []
        x = nx.get_edge_attributes(self.graph,'etype')
        for edge in list(x.keys()):
            if x[edge] == 'C':
                edges.append(edge)
        return edges
    
    #
    #
    #
    def getSpanningBranches(self):
        edges = []
        x = nx.get_edge_attributes(self.graph,'etype')
        for edge in list(x.keys()):
            if x[edge] == 'S':
                edges.append(edge)
        return edges
    
    #
    #
    #
    def getIrreducibleLoops(self):
        
        # Get loops comprising at least 3 nodes
        #loops = list(nx.cycle_basis(self.graph.to_undirected()))
        #newloops = []
        #for loop in loops:
        #    newloop = loop
        #    while newloop[0] != min(loop):
        #        newloop=np.roll(newloop,1)
        #    newloops.append(list(newloop))
        #loops = newloops
        
        # Find loops comprising two nodes
        #sloops = list(nx.simple_cycles(self.graph))
        #for sl in sloops:
        #    if sl not in loops:
        #        loops.append(sl)
        
        #return loops
        return list(nx.simple_cycles(self.graph))
    
    #
    #
    #
    def getSuperconductingLoops(self):
        x = nx.get_edge_attributes(self.graph,'edge_components')
        loops = self.getIrreducibleLoops()
        
        # A superconducting loop shouldn't have capacitors interrupting it
        #
        # FIXME: Doesn't detect superconducting loops when there are parallel branches
        #
        scloops = []
        for loop in loops:
            cbreak = False
            for i in range(len(loop)):
                if i < len(loop)-1:
                    edge = (loop[i],loop[i+1])
                else:
                    edge = (loop[-1],loop[0])
                try:
                    if len(x[edge]['L']) == 0 and len(x[edge]['J']) == 0:
                        cbreak = True
                except KeyError:
                    if len(x[edge[::-1]]['L']) == 0 and len(x[edge[::-1]]['J']) == 0:
                        cbreak = True
            if not cbreak:
                scloops.append(loop)
        return scloops
    
    #
    #
    #
    def getSuperconductingLoopEdges(self):
        scloops = self.getSuperconductingLoops()
        scloop_edges = []
        for loop in scloops:
            l = len(loop)
            a = []
            for i in range(l):
                a.append((loop[i%l],loop[(i+1)%l]))
            scloop_edges.append(a)
        return scloop_edges
    
    #
    #
    #
    def getNonCapacitiveEdges(self):
        x = nx.get_edge_attributes(self.graph,'edge_components')
        edge_list = []
        for edge in self.getEdgeList():
            if len(x[edge]['C']) == 0:
                edge_list.append(edge)
        return edge_list
    
    #
    #
    #
    def getNodeList(self):
        return list(self.graph.nodes)[1:]
    
    #
    #
    #
    def getEdgeList(self):
        return list(self.graph.edges)
    
    def addBranchToGraph(self, component_list, start_node, end_node):
        # Check there are no inductors in branch if a JJ is present
        prefixes = []
        for comp in component_list:
            prefixes.append(comp[0])
        if self.ind_prefix in prefixes and self.jj_prefix in prefixes:
            raise Exception("cannot have an inductor and JJ in the same branch.")

        # Get structured components list
        components = self._get_components_from_strings(component_list)
        
        # Get a structured list of the components connected between start and end node
        caps, caps_sym, inds, inds_sym, jjs, jjs_sym = self._get_component_types(components)
        
        # Update the parameter list
        all_str = []
        all_str.extend(caps)
        all_str.extend(inds)
        all_str.extend(jjs)
        all_sym = []
        all_sym.extend(caps_sym)
        all_sym.extend(inds_sym)
        all_sym.extend(jjs_sym)
        for i,s in enumerate(all_str):
            self.circ_params[s] = all_sym[i]
        
        if end_node != 0:
            if end_node in self.graph.nodes:
                self.graph.add_edge(
                    start_node,
                    end_node,
                    etype="C",
                    edge_components={"C":caps,"L":inds,"J":jjs,"Csym":caps_sym,"Lsym":inds_sym,"Jsym":jjs_sym}
                )

                # Create symbolic DoFs
                self._add_branch_dofs(start_node, end_node)

                # Add empty entry in flux bias vector
                self._add_empty_flux_bias(start_node, end_node)
            else:
                self.graph.add_node(end_node,ntype="Circuit")
                self.graph.add_edge(
                    start_node,
                    end_node,
                    etype="S",
                    edge_components={"C":caps,"L":inds,"J":jjs,"Csym":caps_sym,"Lsym":inds_sym,"Jsym":jjs_sym}
                )

                # Create symbolic DoFs
                self._add_branch_dofs(start_node,end_node)
                self._add_node_dofs(end_node)

                # Add empty entry in flux bias vector
                self._add_empty_flux_bias(start_node,end_node)

                # Set charge bias and gate capacitor terms to zero
                self.charge_bias[end_node] = 0.0
                self.charge_bias_caps[end_node] = 0.0
        else:
            self.graph.add_edge(
                start_node,
                end_node,
                etype="C",
                edge_components={"C":caps,"L":inds,"J":jjs,"Csym":caps_sym,"Lsym":inds_sym,"Jsym":jjs_sym}
            )

            # Create symbolic DoFs
            self._add_branch_dofs(start_node,end_node)

            # Add empty entry in flux bias vector
            self._add_empty_flux_bias(start_node,end_node)

            # Update any flux bias terms
            self._update_flux_bias()
    
    #
    #
    #
    def drawGraph(self,filename=None,output="svg",inline=False,**kwargs):
        fig,ax = plt.subplots(1,1,constrained_layout=True)
        nx.draw(self.graph, with_labels=True, font_weight='bold',pos=nx.circular_layout(self.graph),ax=ax,node_color="C0",node_size=1500,font_size=16,**kwargs)
        edge_labels = nx.get_edge_attributes(self.graph,'etype')
        nx.draw_networkx_edge_labels(self.graph, nx.circular_layout(self.graph), edge_labels=edge_labels,node_color="C0",node_size=1500,font_size=16,**kwargs)
        
        if inline:
            return None
        
        if filename is not None:
            fig.savefig(filename)
            plt.close()
            return None
        
        # Get output into a buffered stream
        imgdata = io.StringIO()
        fig.savefig(imgdata,format=output,bbox_extra_artists=ax.get_default_bbox_extra_artists(), bbox_inches='tight', transparent=False)
        plt.close()
        return imgdata.getvalue()
    
    def drawSubGraph(self, nodes, filename=None, output="svg", inline=False, **kwargs):
        
        # Get the subgraph
        G = self.graph.subgraph(nodes).copy()
        
        # Render
        fig,ax = plt.subplots(1,1,constrained_layout=True)
        nx.draw(G, with_labels=True, font_weight='bold',pos=nx.circular_layout(G),ax=ax,node_color="C0",node_size=1500,font_size=16,**kwargs)
        edge_labels = nx.get_edge_attributes(G,'etype')
        nx.draw_networkx_edge_labels(G, nx.circular_layout(G), edge_labels=edge_labels,node_color="C0",node_size=1500,font_size=16,**kwargs)
        
        if inline:
            return None
        
        if filename is not None:
            fig.savefig(filename)
            plt.close()
            return None
        
        # Get output into a buffered stream
        imgdata = io.StringIO()
        fig.savefig(imgdata,format=output,bbox_extra_artists=ax.get_default_bbox_extra_artists(), bbox_inches='tight', transparent=False)
        plt.close()
        return imgdata.getvalue()
    
    ###################################################################################################################
    #       Symbolic Math and Equation Functions
    ###################################################################################################################
    
    #
    #
    #
    def getParameterDict(self):
        ret = self.circ_params.copy()
        ret.update(dict([(k,self.flux_bias[v]) for k,v in self.flux_bias_names.items()]))
        ret.update(dict([(k,self.charge_bias[v]) for k,v in self.charge_bias_names.items()]))
        return ret
    
    #
    #
    #
    def getDoFSymbol(self,pos,dof_type,mode="node"):
        self._check_args(mode=mode,dof=dof_type)
        
        index = self.__dof_map[dof_type]
        if mode == "node":
            self._check_args(node=pos)
            return self.node_dofs[pos][index]
        elif mode == "branch":
            self._check_args(edge=pos)
            return self.branch_dofs[pos][index]
    
    #
    #
    #
    def getDoFSymbolAsMatrix(self,pos,dof_type,n,mode="node"):
        return sy.MatrixSymbol(self.getDoFSymbol(pos,dof_type,mode=mode),n,n)
    
    #
    #
    #
    def getDoFSymbolList(self,dof_type,mode="node"):
        self._check_args(mode=mode,dof=dof_type)
        ret = {}
        if mode == "node":
            for node in list(self.graph.nodes):
                if node == 0:
                    continue
                ret[node] = self.getDoFSymbol(node,dof_type,mode=mode)
            return ret
        elif mode == "branch":
            for edge in list(self.graph.edges):
                ret[edge] = self.getDoFSymbol(edge,dof_type,mode=mode)
            return ret
    
    #
    #
    #
    def getBiasSymbol(self,pos,bias_type):
        self._check_args(bias=bias_type)
        
        if bias_type == "charge":
            self._check_args(node=pos)
            return self.charge_bias[pos]
        elif bias_type == "flux":
            self._check_args(edge=pos)
            return self.flux_bias[pos]
    
    #
    #
    #
    def getBiasSymbolList(self,bias_type):
        self._check_args(bias=bias_type)
        if bias_type == "charge":
            return dict([(k,self.charge_bias[k]) for k in list(self.charge_bias.keys()) if self.charge_bias[k] != 0.0])
        elif bias_type == "flux":
            return dict([(k,self.flux_bias[k]) for k in list(self.flux_bias.keys()) if self.flux_bias[k] != 0.0])
    
    #
    #
    #
    def setModeTransformationMatrix(self,matrix_data,auto=False,mode="node"):
        self._check_args(mode=mode)
        if auto:
            if mode == "node":
                self.Rtrans_n = sy.eye(len(list(self.graph.nodes))-1)
            elif mode == "branch":
                self.Rtrans_b = sy.eye(len(list(self.graph.edges)))
        else:
            if mode == "node":
                self.Rtrans_n = sy.Matrix(matrix_data)
            elif mode == "branch":
                self.Rtrans_b = sy.Matrix(matrix_data)
    
    #
    #
    #
    def getModeTransformationMatrix(self,mode="node",sym_lev="lowest",as_equ=False):
        self._check_args(mode=mode,sym_lev=sym_lev)
        if mode == "node":
            if not hasattr(self,"Rtrans_n"):
                self.setModeTransformationMatrix(None,auto=True,mode=mode)
        elif mode == "branch":
            if not hasattr(self,"Rtrans_b"):
                self.setModeTransformationMatrix(None,auto=True,mode=mode)
        
        if as_equ:
            sym = self.getModeTransformationMatrix(mode=mode,sym_lev="highest")
            if mode == "node":
                return sy.Eq(sym,self.Rtrans_n)
            elif mode == "branch":
                return sy.Eq(sym,self.Rtrans_b)
        
        if sym_lev == "lowest":
            if mode == "node":
                return self.Rtrans_n
            elif mode == "branch":
                return self.Rtrans_b
        elif sym_lev == "highest":
            if mode == "node":
                return sy.MatrixSymbol(sy.symbols("\\hat{\\mathbf{R}}_{nt}"),len(list(self.graph.nodes))-1,len(list(self.graph.nodes))-1)
            elif mode == "branch":
                return sy.MatrixSymbol(sy.symbols("\\hat{\\mathbf{R}}_{bt}"),len(list(self.graph.edges)),len(list(self.graph.edges)))
    
    #
    #
    #
    def getCapacitanceMatrix(self,mode="node",sym_lev="lowest",as_equ=False):
        self._check_args(mode=mode,sym_lev=sym_lev)
        if as_equ:
            sym = self.getCapacitanceMatrix(mode=mode,sym_lev="highest")
            mat = self.getCapacitanceMatrix(mode=mode,sym_lev="lowest")
            return sy.Eq(sym,mat)
        
        if sym_lev == "lowest":
            # Get all edges
            x = nx.get_edge_attributes(self.graph,'edge_components')
            
            # Construct node capacitance matrix first
            # Get diagonals
            M = sy.eye(len(list(self.graph.nodes))-1) - sy.eye(len(list(self.graph.nodes))-1)
            for node in list(self.graph.nodes):
                if node == 0:
                    continue
                all_edges = list(self.graph.in_edges(node))
                all_edges.extend(list(self.graph.out_edges(node)))
                expr = 0
                for edge in all_edges:
                    for C in x[edge]['Csym']:
                        expr += C
                M[node-1,node-1] = expr

            # Get off diagonals
            for edge in list(x.keys()):
                if edge[0] == 0 or edge[1] == 0:
                    continue
                if len(x[edge]['Csym']) == 0:
                    continue
                else:
                    expr = 0
                    for C in x[edge]['Csym']:
                        expr -= C
                    M[edge[0]-1,edge[1]-1] = expr
                    M[edge[1]-1,edge[0]-1] = expr
            
            if mode == "node":
                return M
            elif mode == "branch":
                return self.getNodeToBranchMatrix()*M*self.getBranchToNodeMatrix()
        elif sym_lev == "highest":
            if mode == "node":
                return sy.MatrixSymbol(sy.symbols("\\hat{\\mathbf{C}}_{n}"),len(list(self.graph.nodes))-1,len(list(self.graph.nodes))-1)
            elif mode == "branch":
                return sy.MatrixSymbol(sy.symbols("\\hat{\\mathbf{C}}_{b}"),len(list(self.graph.edges)),len(list(self.graph.edges)))
    
    #
    #
    #
    def getGateCapacitanceMatrix(self,mode="node",sym_lev="lowest",as_equ=False):
        self._check_args(mode=mode,sym_lev=sym_lev)
        if as_equ:
            sym = self.getGateCapacitanceMatrix(mode=mode,sym_lev="highest")
            mat = self.getGateCapacitanceMatrix(mode=mode,sym_lev="lowest")
            return sy.Eq(sym,mat)
        
        if sym_lev == "lowest":
            M = sy.eye(len(list(self.graph.nodes))-1) - sy.eye(len(list(self.graph.nodes))-1)
            for node in list(self.graph.nodes):
                if node == 0:
                    continue
                M[node-1,node-1] = self.charge_bias_caps[node]
            if mode == "node":
                return M
            elif mode == "branch":
                return self.getNodeToBranchMatrix()*M*self.getBranchToNodeMatrix()
        elif sym_lev == "highest":
            if mode == "node":
                return sy.MatrixSymbol(sy.symbols("\\hat{\\mathbf{C}}_{gn}"),len(list(self.graph.nodes))-1,len(list(self.graph.nodes))-1)
            elif mode == "branch":
                return sy.MatrixSymbol(sy.symbols("\\hat{\\mathbf{C}}_{gb}"),len(list(self.graph.edges)),len(list(self.graph.edges)))
    
    #
    #
    #
    def getInductanceMatrix(self,mode="node",sym_lev="lowest",as_equ=False):
        self._check_args(mode=mode,sym_lev=sym_lev)
        if as_equ:
            sym = self.getInductanceMatrix(mode=mode,sym_lev="highest")
            mat = self.getInductanceMatrix(mode=mode,sym_lev="lowest")
            return sy.Eq(sym,mat)
        
        if sym_lev == "lowest":
            # Get all edges
            x = nx.get_edge_attributes(self.graph,'edge_components')
            
            # Get diagonals
            M = sy.eye(len(list(self.graph.edges))) - sy.eye(len(list(self.graph.edges)))
            branch_indices = {}
            for i,edge in enumerate(list(self.graph.edges)):
                expr = 0
                for L in x[edge]['Lsym']:
                    expr += 1/L # Do inductors in parallel
                if expr != 0:
                    expr = 1/expr
                M[i,i] = expr
                branch_indices[edge] = i
            
            # Get off diagonals
            coupled_branches = nx.get_edge_attributes(self.graph,'mutual')
            for k,v in coupled_branches.items():
                i = branch_indices[k]
                for elem in v:
                    val = elem[0]
                    sym = elem[1]
                    j = branch_indices[val]
                    expr = self.circ_params[sym]
                    M[i,j] = expr
                    M[j,i] = expr
            
            if mode == "node":
                return self.getBranchToNodeMatrix()*M*self.getNodeToBranchMatrix()
            elif mode == "branch":
                return M
        elif sym_lev == "highest":
            if mode == "node":
                return sy.MatrixSymbol(sy.symbols("\\hat{\\mathbf{L}}_{n}"),len(list(self.graph.nodes))-1,len(list(self.graph.nodes))-1)
            elif mode == "branch":
                return sy.MatrixSymbol(sy.symbols("\\hat{\\mathbf{L}}_{b}"),len(list(self.graph.edges)),len(list(self.graph.edges)))
    
    #
    #
    #
    def getBiasInductanceMatrix(self,mode="node",sym_lev="lowest",as_equ=False):
        pass
    
    #
    #
    #
    def getInverseCapacitanceMatrix(self,mode="node",sym_lev="lowest",as_equ=False):
        self._check_args(mode=mode,sym_lev=sym_lev)
        if as_equ:
            sym = self.getInverseCapacitanceMatrix(mode=mode,sym_lev="highest")
            mat = self.getInverseCapacitanceMatrix(mode=mode,sym_lev="lowest")
            return sy.Eq(sym,mat)
        
        if sym_lev == "lowest":
            # Try to invert the inductance matrix as-is
            try:
                return (self.getCapacitanceMatrix(mode=mode)+self.getGateCapacitanceMatrix(mode=mode))**(-1)
            except ValueError:
                pass
            
            # Failing that, take the pseudo-inverse of the node capacitance matrix
            # and transform that if required
            if mode == "node":
                return (self.getCapacitanceMatrix(mode="node")+self.getGateCapacitanceMatrix(mode="node")).pinv()
            elif mode == "branch":
                return self.getNodeToBranchMatrix()*(self.getCapacitanceMatrix(mode="node")\
                       +self.getGateCapacitanceMatrix(mode="node")).pinv()*self.getBranchToNodeMatrix()
        elif sym_lev == "highest":
            return (self.getCapacitanceMatrix(mode=mode,sym_lev="highest")+self.getGateCapacitanceMatrix(mode=mode,sym_lev="highest"))**(-1)
    
    #
    #
    #
    def getInverseInductanceMatrix(self,mode="node",sym_lev="lowest",as_equ=False):
        self._check_args(mode=mode,sym_lev=sym_lev)
        if as_equ:
            sym = self.getInverseInductanceMatrix(mode=mode,sym_lev="highest")
            mat = self.getInverseInductanceMatrix(mode=mode,sym_lev="lowest")
            return sy.Eq(sym,mat)
        
        if sym_lev == "lowest":
            
            # If there is only one node, and there is no inductance, just return zero
            if len(list(self.graph.nodes))-1 < 2 or len(list(self.graph.edges)) < 2:
                # Get all edges
                x = nx.get_edge_attributes(self.graph,'edge_components')
                zero = True
                for i,edge in enumerate(list(self.graph.edges)):
                    for L in x[edge]['Lsym']:
                        if L != 0.0:
                            zero = False
                            continue
                if zero:
                    return sy.eye(1)-sy.eye(1)
            
            # Try to invert the inductance matrix as-is
            try:
                return self.getInductanceMatrix(mode=mode)**(-1)
            except ValueError:
                pass
            
            # Failing that, take the pseudo-inverse of the branch inductance matrix
            # and transform that if required
            if mode == "node":
                return self.getBranchToNodeMatrix()*self.getInductanceMatrix(mode="branch").pinv()*self.getNodeToBranchMatrix()
            elif mode == "branch":
                #return sy.simplify(self.getInductanceMatrix(mode="branch").pinv())
                return self.getInductanceMatrix(mode="branch").pinv()
        elif sym_lev == "highest":
            return (self.getInductanceMatrix(mode=mode,sym_lev="highest"))**(-1)
    
    #
    #
    #
    def getJosephsonMatrix(self,mode="branch",sym_lev="lowest",as_equ=False):
        self._check_args(mode=mode,sym_lev=sym_lev)
        if as_equ:
            sym = self.getJosephsonMatrix(mode=mode,sym_lev="highest")
            mat = self.getJosephsonMatrix(mode=mode,sym_lev="lowest")
            return sy.Eq(sym,mat)
        
        if sym_lev == "lowest":
            # Get all edges
            x = nx.get_edge_attributes(self.graph,'edge_components')
            
            M = sy.eye(len(list(self.graph.edges))) - sy.eye(len(list(self.graph.edges)))
            for i,edge in enumerate(list(self.graph.edges)):
                expr = 0
                for J in x[edge]['Jsym']:
                    expr = J
                M[i,i] = expr
            
            if mode == "node":
                return self.getBranchToNodeMatrix()*M*self.getNodeToBranchMatrix()
            elif mode == "branch":
                return M
        elif sym_lev == "highest":
            return sy.MatrixSymbol(sy.symbols("\\hat{\\mathbf{J}}_{b}"),len(list(self.graph.edges)),len(list(self.graph.edges)))
    
    #
    #
    #
    def getJosephsonVector(self,sym_lev="lowest",as_equ=False):
        self._check_args(sym_lev=sym_lev)
        if as_equ:
            sym = self.getJosephsonVector(sym_lev="highest")
            mat = self.getJosephsonVector(sym_lev="lowest")
            return sy.Eq(sym,mat)
        
        if sym_lev == "lowest":
            # Get all edges
            x = nx.get_edge_attributes(self.graph,'edge_components')
            
            vec = list(np.zeros(len(list(self.graph.edges))))
            for i,edge in enumerate(list(self.graph.edges)):
                expr = 0
                for J in x[edge]['Jsym']:
                    expr += J
                vec[i] = expr
            return sy.Matrix(vec)
        elif sym_lev == "highest":
            return sy.MatrixSymbol(sy.symbols("\\vec{\\mathbf{J}}_{b}"),len(list(self.graph.edges)),1)
    
    #
    #
    #
    def getFluxBiasVector(self,mode="node",form="flux",sym_lev="lowest",as_equ=False):
        self._check_args(mode=mode,sym_lev=sym_lev,fl_form=form)
        if as_equ:
            sym = self.getFluxBiasVector(mode=mode,form=form,sym_lev="highest")
            mat = self.getFluxBiasVector(mode=mode,form=form,sym_lev="lowest")
            return sy.Eq(sym,mat)
        
        if sym_lev == "lowest":
            if form == "flux":
                bias_vec = list(np.zeros(len(list(self.graph.edges))))
                for i,edge in enumerate(list(self.graph.edges)):
                    bias_vec[i] = self.flux_bias_prefactor[edge]*self.flux_bias[edge]
            elif form == "phase":
                bias_vec = list(np.zeros(len(list(self.graph.edges))))
                for i,edge in enumerate(list(self.graph.edges)):
                    bias_vec[i] = self.flux_bias_prefactor[edge]*self.red_flux_bias[edge]#sy.symbols("%s_{%i%ie}" % (self.redflux_prefix,edge[0],edge[1]))
            elif form == "expphase":
                bias_vec = list(np.ones(len(list(self.graph.edges))))
                for i,edge in enumerate(list(self.graph.edges)):
                    bias_vec[i] = self.exp_flux_bias[edge]#sy.exp(1.0j*sy.symbols("%s_{%i%ie}" % (self.redflux_prefix,edge[0],edge[1])))
                    #bias_vec[i] = sy.exp(1.0j*self.red_flux_bias[edge])
            if mode == "node":
                return self.getBranchToNodeMatrix()*sy.Matrix(bias_vec)
            elif mode == "branch":
                return sy.Matrix(bias_vec)
        elif sym_lev == "highest":
            if mode == "node":
                if form == "flux":
                    return sy.MatrixSymbol(sy.symbols("\\vec{\\mathbf{\\Phi}}_{ne}"),len(list(self.graph.nodes))-1,1)
                elif form == "phase":
                    return sy.MatrixSymbol(sy.symbols("\\vec{\\mathbf{\\phi}}_{ne}"),len(list(self.graph.nodes))-1,1)
                elif form == "expphase":
                    return sy.MatrixSymbol(sy.symbols("\\vec{\\mathbf{\\phi^{\\mathrm{e}}}}_{ne}"),len(list(self.graph.nodes))-1,1)
            elif mode == "branch":
                if form == "flux":
                    return sy.MatrixSymbol(sy.symbols("\\vec{\\mathbf{\\Phi}}_{be}"),len(list(self.graph.edges)),1)
                elif form == "phase":
                    return sy.MatrixSymbol(sy.symbols("\\vec{\\mathbf{\\phi}}_{be}"),len(list(self.graph.edges)),1)
                elif form == "expphase":
                    return sy.MatrixSymbol(sy.symbols("\\vec{\\mathbf{\\phi^{\\mathrm{e}}}}_{be}"),len(list(self.graph.edges)),1)
    
    #
    #
    #
    def moveFluxBias(self,old_edge,new_edge):
        # Get available edges in loops
        #
        # FIXME: Doesn't consider the case where an edge is shared by two loops
        #
        for loop in self.getSuperconductingLoopEdges():
            if old_edge in loop:
                break

        # Check selected edge is in loop
        if new_edge not in loop:
            raise Exception("new_edge %s not in selected loop %s" % (repr(new_edge),repr(loop)))
        
        # Check old_edge has a bias term
        if self.flux_bias[old_edge] == 0.0:
            raise Exception("old_edge %s has no bias term" % (repr(new_edge)))
        
        # Get symbol names
        names = {v:k for k,v in self.flux_bias_names.items()}
        self.flux_bias_names[names[old_edge]] = new_edge
        
        # Get symbol and replace
        s = self.flux_bias[old_edge]
        p = self.flux_bias_prefactor[old_edge]
        self.flux_bias[old_edge] = 0.0
        self.flux_bias_prefactor[old_edge] = 1.0
        self.flux_bias[new_edge] = s
        self.flux_bias_prefactor[new_edge] = p
        s = self.red_flux_bias[old_edge]
        self.red_flux_bias[old_edge] = 0.0
        self.red_flux_bias[new_edge] = s
        s = self.exp_flux_bias[old_edge]
        self.exp_flux_bias[old_edge] = 1.0
        self.exp_flux_bias[new_edge] = s
    
    #
    #
    #
    def symmetrizeFluxBias(self, old_edge, new_edge):
        # Get available edges in loops
        #
        # FIXME: Doesn't consider the case where an edge is shared by two loops
        #
        for loop in self.getSuperconductingLoopEdges():
            if old_edge in loop:
                break

        # Check selected edge is in loop
        if new_edge not in loop:
            raise Exception("new_edge %s not in selected loop %s" % (repr(new_edge), repr(loop)))
        
        # Check old_edge has a bias term
        if self.flux_bias[old_edge] == 0.0:
            raise Exception("old_edge %s has no bias term" % (repr(new_edge)))
        
        # Get symbol names
        names = {v:k for k,v in self.flux_bias_names.items()}
        self.flux_bias_names[names[old_edge]] = new_edge
        
        # Get symbol and replace
        s = self.flux_bias[old_edge]
        p = self.flux_bias_prefactor[old_edge]
        self.flux_bias[old_edge] = s
        self.flux_bias[new_edge] = s
        self.flux_bias_prefactor[old_edge] = 0.5*p
        self.flux_bias_prefactor[new_edge] = 0.5*p
        s = self.red_flux_bias[old_edge]
        self.red_flux_bias[old_edge] = s
        self.red_flux_bias[new_edge] = s
        s = self.exp_flux_bias[old_edge]
        self.exp_flux_bias[old_edge] = s
        self.exp_flux_bias[new_edge] = s
    
    #
    #
    #
    def getFluxBiasMatrix(self,mode="node",form="flux",sym_lev="lowest",as_equ=False):
        self._check_args(mode=mode,sym_lev=sym_lev,fl_form=form)
        if as_equ:
            sym = self.getFluxBiasMatrix(mode=mode,form=form,sym_lev="highest")
            mat = self.getFluxBiasMatrix(mode=mode,form=form,sym_lev="lowest")
            return sy.Eq(sym,mat)
        
        if sym_lev == "lowest":
            return sy.diag(*self.getFluxBiasVector(mode=mode,form=form))
        elif sym_lev == "highest":
            if mode == "node":
                if form == "flux":
                    return sy.MatrixSymbol(sy.symbols("\\hat{\\mathbf{\\Phi}}_{ne}"),len(list(self.graph.nodes))-1,len(list(self.graph.nodes))-1)
                elif form == "phase":
                    return sy.MatrixSymbol(sy.symbols("\\hat{\\mathbf{\\phi}}_{ne}"),len(list(self.graph.nodes))-1,len(list(self.graph.nodes))-1)
                elif form == "expphase":
                    return sy.MatrixSymbol(sy.symbols("\\hat{\\mathbf{\\phi^{\\mathrm{e}}}}_{ne}"),len(list(self.graph.nodes))-1,len(list(self.graph.nodes))-1)
            elif mode == "branch":
                if form == "flux":
                    return sy.MatrixSymbol(sy.symbols("\\hat{\\mathbf{\\Phi}}_{be}"),len(list(self.graph.edges)),len(list(self.graph.edges)))
                elif form == "phase":
                    return sy.MatrixSymbol(sy.symbols("\\hat{\\mathbf{\\phi}}_{be}"),len(list(self.graph.edges)),len(list(self.graph.edges)))
                elif form == "expphase":
                    return sy.MatrixSymbol(sy.symbols("\\hat{\\mathbf{\\phi^{\\mathrm{e}}}}_{be}"),len(list(self.graph.edges)),len(list(self.graph.edges)))
    
    #
    #
    #
    def getChargeBiasVector(self,mode="node",form="charge",sym_lev="lowest",as_equ=False):
        self._check_args(mode=mode,sym_lev=sym_lev,ch_form=form)
        if as_equ:
            sym = self.getChargeBiasVector(form=form,sym_lev="highest")
            mat = self.getChargeBiasVector(form=form,sym_lev="lowest")
            return sy.Eq(sym,mat)
        
        if sym_lev == "lowest":
            if form == "charge":
                bias_vec = list(np.zeros(len(list(self.graph.nodes))-1))
                nodes = list(self.graph.nodes)
                nodes.remove(0)
                for i,node in enumerate(nodes):
                    bias_vec[i] = self.charge_bias[node]
            elif form == "phase": # THIS WILL BREAK, FIX
                bias_vec = list(np.zeros(len(list(self.graph.edges))))
                for i,node in enumerate(list(self.graph.edges)):
                    if node in self.getClosureBranches():
                        bias_vec[i] = sy.symbols("%s_{%i%ie}" % (self.redcharge_prefix,edge[0],edge[1]))
            if mode == "node":
                return sy.Matrix(bias_vec)
            elif mode == "branch":
                return self.getNodeToBranchMatrix()*sy.Matrix(bias_vec)
        elif sym_lev == "highest":
            if mode == "node":
                if form == "charge":
                    return sy.MatrixSymbol(sy.symbols("\\vec{\\mathbf{Q}}_{ne}"),len(list(self.graph.nodes))-1,1)
                elif form == "phase":
                    return sy.MatrixSymbol(sy.symbols("\\vec{\\mathbf{q}}_{ne}"),len(list(self.graph.nodes))-1,1)
            elif mode == "branch":
                if form == "charge":
                    return sy.MatrixSymbol(sy.symbols("\\vec{\\mathbf{Q}}_{be}"),len(list(self.graph.edges)),1)
                elif form == "phase":
                    return sy.MatrixSymbol(sy.symbols("\\vec{\\mathbf{q}}_{be}"),len(list(self.graph.edges)),1)
    
    #
    #
    #
    def getChargeVector(self,mode="node",sym_lev="lowest",as_equ=False):
        self._check_args(mode=mode,sym_lev=sym_lev)
        if as_equ:
            sym = self.getChargeVector(mode=mode,sym_lev="highest")
            mat = self.getChargeVector(mode=mode,sym_lev="lowest")
            return sy.Eq(sym,mat)
        
        if sym_lev == "lowest":
            if mode == "node":
                return sy.Matrix([self.node_dofs[n][1] for n in list(self.graph.nodes)[1:]])
            elif mode == "branch":
                return sy.Matrix([self.branch_dofs[e][1] for e in list(self.graph.edges)])
        elif sym_lev == "highest":
            if mode == "node":
                return sy.MatrixSymbol(sy.symbols("\\vec{\\mathbf{Q}}_{n}"),len(list(self.graph.nodes))-1,1)
            elif mode == "branch":
                return sy.MatrixSymbol(sy.symbols("\\vec{\\mathbf{Q}}_{b}"),len(list(self.graph.edges)),1)
    
    #
    #
    #
    def getVoltageVector(mode="node",sym_lev="lowest",if_equ=False):
        self._check_args(mode=mode,sym_lev=sym_lev)
        if as_equ:
            sym = self.getVoltageVector(mode=mode,sym_lev="highest")
            mat = self.getVoltageVector(mode=mode,sym_lev="lowest")
            return sy.Eq(sym,mat)
        
        if sym_lev == "lowest":
            if mode == "node":
                return sy.Matrix([sy.symbols("V_{%i}" % n) for n in list(self.graph.nodes)[1:]])
            elif mode == "branch":
                return sy.Matrix([sy.symbols("V_{%i%i}" % (e[0],e[1])) for e in list(self.graph.edges)])
        elif sym_lev == "highest":
            if mode == "node":
                return sy.MatrixSymbol(sy.symbols("\\vec{\\mathbf{Q}}_{n}"),len(list(self.graph.nodes))-1,1)
            elif mode == "branch":
                return sy.MatrixSymbol(sy.symbols("\\vec{\\mathbf{Q}}_{b}"),len(list(self.graph.edges)),1)
    
    #
    #
    #
    def getFluxVector(self,mode="node",sym_lev="lowest",as_equ=False):
        self._check_args(mode=mode,sym_lev=sym_lev)
        if as_equ:
            sym = self.getFluxVector(mode=mode,sym_lev="highest")
            mat = self.getFluxVector(mode=mode,sym_lev="lowest")
            return sy.Eq(sym,mat)
        
        if sym_lev == "lowest":
            if mode == "node":
                return sy.Matrix([self.node_dofs[n][0] for n in list(self.graph.nodes)[1:]])
            elif mode == "branch":
                return sy.Matrix([self.branch_dofs[e][0] for e in list(self.graph.edges)])
        elif sym_lev == "highest":
            if mode == "node":
                return sy.MatrixSymbol(sy.symbols("\\vec{\\mathbf{\\Phi}}_{n}"),len(list(self.graph.nodes))-1,1)
            elif mode == "branch":
                return sy.MatrixSymbol(sy.symbols("\\vec{\\mathbf{\\Phi}}_{b}"),len(list(self.graph.edges)),1)
    
    #
    #
    #
    def getRedFluxVector(self,mode="node",sym_lev="lowest",as_equ=False):
        self._check_args(mode=mode,sym_lev=sym_lev)
        if as_equ:
            sym = self.getRedFluxVector(mode=mode,sym_lev="highest")
            mat = self.getRedFluxVector(mode=mode,sym_lev="lowest")
            return sy.Eq(sym,mat)
        
        if sym_lev == "lowest":
            if mode == "node":
                return sy.Matrix([sy.symbols("%s_{%i}" % (self.redflux_prefix,n)) for n in list(self.graph.nodes)[1:]])
            elif mode == "branch":
                return sy.Matrix([sy.symbols("%s_{%i%i}" % (self.redflux_prefix,e[0],e[1])) for e in list(self.graph.edges)])
        elif sym_lev == "highest":
            if mode == "node":
                return sy.MatrixSymbol(sy.symbols("\\vec{\\mathbf{\\phi}}_{n}"),len(list(self.graph.nodes))-1,1)
            elif mode == "branch":
                return sy.MatrixSymbol(sy.symbols("\\vec{\\mathbf{\\phi}}_{b}"),len(list(self.graph.edges)),1)
    
    #
    #
    #
    def getFluxExpr(self,edge,etype="phase"):
        if etype == "phase":
            return 2*sy.pi*self.flux_bias_prefactor[edge]*self.flux_bias[edge]/self.phi0
        elif etype == "expphase":
            return sy.exp(1.0j*2*sy.pi*self.flux_bias_prefactor[edge]*self.flux_bias[edge]/self.phi0)
    
    #
    #
    #
    def getCurrentVector(self,mode="node",sym_lev="lowest",as_equ=False):
        self._check_args(mode=mode,sym_lev=sym_lev)
        if as_equ:
            sym = self.getCurrentVector(mode=mode,sym_lev="highest")
            mat = self.getCurrentVector(mode=mode,sym_lev="lowest")
            return sy.Eq(sym,mat)
        
        if mode == "node":
            return sy.Matrix([sy.symbols("I_{%i}" % n) for n in list(self.graph.nodes)[1:]])
        elif mode == "branch":
            return sy.Matrix([sy.symbols("I_{%i%i}" % (e[0],e[1])) for e in list(self.graph.edges)])
    
    #
    #
    #
    def getChargingEnergies(self,mode="node",sym_lev="lowest",as_equ=False):
        self._check_args(mode=mode,sym_lev=sym_lev)
        if as_equ:
            sym = self.getChargingEnergies(mode=mode,sym_lev="highest")
            mat = self.getChargingEnergies(mode=mode,sym_lev="lowest")
            return sy.Eq(sym,mat)
        
        if sym_lev == "lowest":
            return 0.5*(self.getChargeVector(mode=mode)+self.getChargeBiasVector(mode=mode)).transpose()\
            *self.getInverseCapacitanceMatrix(mode=mode)\
            *(self.getChargeVector(mode=mode)+self.getChargeBiasVector(mode=mode))
        elif sym_lev == "highest":
            return 0.5*(self.getChargeVector(mode=mode,sym_lev="highest")+self.getChargeBiasVector(mode=mode,sym_lev="highest")).transpose()\
            *self.getInverseCapacitanceMatrix(mode=mode,sym_lev="highest")\
            *(self.getChargeVector(mode=mode,sym_lev="highest")+self.getChargeBiasVector(mode=mode,sym_lev="highest"))
    
    #
    #
    #
    def getSingleParticleChargingEnergies(self,mode="node"):
        self._check_args(mode=mode)
        ret = {}
        Cinv = self.getInverseCapacitanceMatrix(mode=mode)
        if mode == "node":
            for i,node in enumerate(self.getNodeList()):
                ret[node] = 0.5 * self.qcp**2 * Cinv[i,i]
        elif mode == "branch":
            for i,edge in enumerate(self.getEdgeList()):
                ret[edge] = 0.5 * self.qcp**2 * Cinv[i,i]
        return ret
    
    #
    #
    #
    def getFluxEnergies(self,mode="node",sym_lev="lowest",as_equ=False):
        self._check_args(mode=mode,sym_lev=sym_lev)
        if as_equ:
            sym = self.getFluxEnergies(mode=mode,sym_lev="highest")
            mat = self.getFluxEnergies(mode=mode,sym_lev="lowest")
            return sy.Eq(sym,mat)
        
        if sym_lev == "lowest":
            # In the case of two branches in a loop, bias terms always goes to the JJ
            if len(self.getEdgeList()) == 2 and len(self.getNodeList()) == 1:
                return 0.5*self.getFluxVector(mode=mode)\
                *self.getInverseInductanceMatrix(mode=mode)\
                *self.getFluxVector(mode=mode)
            else:
                # FIXME: Need to check that a suitable basis is used to include the bias term here
                #return 0.5*(self.getFluxVector(mode=mode)+self.getFluxBiasVector(mode=mode)).transpose()\
                #*self.getInverseInductanceMatrix(mode=mode)\
                #*(self.getFluxVector(mode=mode)+self.getFluxBiasVector(mode=mode))
                return 0.5*(self.getFluxVector(mode=mode)).transpose()\
                *self.getInverseInductanceMatrix(mode=mode)\
                *(self.getFluxVector(mode=mode))
        elif sym_lev == "highest":
            return 0.5*(self.getFluxVector(mode=mode,sym_lev="highest")+self.getFluxBiasVector(mode=mode,sym_lev="highest")).transpose()\
            *self.getInverseInductanceMatrix(mode=mode,sym_lev="highest")\
            *(self.getFluxVector(mode=mode,sym_lev="highest")+self.getFluxBiasVector(mode=mode,sym_lev="highest"))
    
    #
    #
    #
    def getSingleParticleFluxEnergies(self,mode="node"):
        self._check_args(mode=mode)
        ret = {}
        Linv = self.getInverseInductanceMatrix(mode=mode)
        if mode == "node":
            for i,node in enumerate(self.getNodeList()):
                ret[node] = 0.5 * self.phi0**2 * Linv[i,i]
        elif mode == "branch":
            for i,edge in enumerate(self.getEdgeList()):
                ret[edge] = 0.5 * self.phi0**2 * Linv[i,i]
        return ret
    
    #
    #
    #
    def getClassicalJosephsonEnergies(self,mode="node",sym_lev="lowest",as_equ=False):
        self._check_args(mode=mode,sym_lev=sym_lev)
        if as_equ:
            sym = self.getClassicalJosephsonEnergies(mode=mode,sym_lev="highest")
            mat = self.getClassicalJosephsonEnergies(mode=mode,sym_lev="lowest")
            return sy.Eq(sym,mat)
        
        if sym_lev == "lowest":
            if mode == "node":
                Jvec = self.getJosephsonVector().transpose()
                Jf = self.getNodeToBranchMatrix()*self.getRedFluxVector(mode="node")+self.getFluxBiasVector(mode="branch",form="phase")
                Jcos = sy.Matrix([sy.cos(e) for e in Jf])
                return -Jvec*Jcos
            elif mode == "branch":
                Jvec = self.getJosephsonVector().transpose()
                Jf = self.getRedFluxVector(mode="branch")+self.getFluxBiasVector(mode="branch",form="phase")
                Jcos = sy.Matrix([sy.cos(e) for e in Jf])
                return -Jvec*Jcos
        elif sym_lev == "highest":
            if mode == "node":
                return -self.getJosephsonVector(sym_lev="highest").transpose() \
                *sy.MatrixSymbol("\\hat{\\mathbf{\\mathrm{cos}}}",len(list(self.graph.edges)),len(list(self.graph.edges)))\
                *(self.getNodeToBranchMatrix(sym_lev="highest")*self.getRedFluxVector(mode="node",sym_lev="highest")\
                +self.getFluxBiasVector(mode="branch",form="phase",sym_lev="highest"))
            elif mode == "branch":
                return -self.getJosephsonVector(sym_lev="highest").transpose() \
                *(self.getRedFluxVector(mode="branch",sym_lev="highest")\
                +self.getFluxBiasVector(mode="branch",form="phase",sym_lev="highest"))
    
    #
    #
    #
    def getJosephsonEnergies(self):
        ret = {}
        Jvec = self.getJosephsonVector()
        for i,edge in enumerate(self.getEdgeList()):
            ret[edge] = self.phi0*Jvec[i]/(2*self.pi)
        return ret
    
    #
    #
    #
    def getLeftDisplacementOpMatrix(self,adjoint=False,mode="node",sym_lev="lowest",as_equ=False,as_vec=False):
        self._check_args(mode=mode,sym_lev=sym_lev)
        if as_equ:
            sym = self.getLeftDisplacementOpMatrix(adjoint=adjoint,mode=mode,sym_lev="highest")
            mat = self.getLeftDisplacementOpMatrix(adjoint=adjoint,mode=mode,sym_lev="lowest")
            return sy.Eq(sym,mat)
        
        if sym_lev == "lowest":
            if mode == "node":
                vec = list(np.zeros(len(list(self.graph.edges))))
                edges = list(self.graph.edges)
                ind2 = 2 if not adjoint else 3
                ind1 = 3 if not adjoint else 2
                
                # Use left decomposed flux vector
                lv = self.getLeftDecompFluxVector()
                rv = self.getRightDecompFluxVector()
                for i in range(len(edges)):
                    l = lv[i]
                    r = rv[i]
                    if l == 0:
                        vec[i] = 1
                    else:
                        if r.args != ():
                            if l.args[0] == 1 and r.args[0] == -1:
                                vec[i] = self.node_dofs[edges[i][0]][ind2]
                        else:
                            vec[i] = self.node_dofs[edges[i][0]][ind2]
                if as_vec:
                    return sy.Matrix(vec)
                else:
                    return sy.diag(*vec)
            elif mode == "branch":
                
                pass
        elif sym_lev == "highest":
            if mode == "node":
                if not adjoint:
                    return sy.MatrixSymbol(sy.symbols("\\hat{\\mathbf{D}}_{nl}"),len(list(self.graph.edges)),len(list(self.graph.edges)))
                else:
                    return sy.MatrixSymbol(sy.symbols("\\hat{\\mathbf{D}}_{nl}^{\\dagger}"),len(list(self.graph.edges)),len(list(self.graph.edges)))
            elif mode == "branch":
                pass
    
    #
    #
    #
    def getRightDisplacementOpVector(self,adjoint=False,mode="node",sym_lev="lowest",as_equ=False):
        self._check_args(mode=mode,sym_lev=sym_lev)
        if as_equ:
            sym = self.getRightDisplacementOpVector(adjoint=adjoint,mode=mode,sym_lev="highest")
            mat = self.getRightDisplacementOpVector(adjoint=adjoint,mode=mode,sym_lev="lowest")
            return sy.Eq(sym,mat)
        
        if sym_lev == "lowest":
            if mode == "node":
                vec = list(np.zeros(len(list(self.graph.edges))))
                edges = list(self.graph.edges)
                ind2 = 2 if not adjoint else 3
                ind1 = 3 if not adjoint else 2
                
                
                # Use right decomposed flux vector
                lv = self.getLeftDecompFluxVector()
                rv = self.getRightDecompFluxVector()
                for i in range(len(edges)):
                    l = lv[i]
                    r = rv[i]
                    if r == 0:
                        vec[i] = 1
                    else:
                        if l.args != ():
                            if l.args[0] == 1 and r.args[0] == -1:
                                vec[i] = self.node_dofs[edges[i][1]][ind1]
                        else:
                            vec[i] = self.node_dofs[edges[i][1]][ind2]
                return sy.Matrix(vec)
            elif mode == "branch":
                
                pass
        elif sym_lev == "highest":
            if mode == "node":
                if not adjoint:
                    return sy.MatrixSymbol(sy.symbols("\\vec{\\mathbf{D}}_{nr}"),len(list(self.graph.edges)),1)
                else:
                    return sy.MatrixSymbol(sy.symbols("\\vec{\\mathbf{D}}_{nr}^{\\dagger}"),len(list(self.graph.edges)),1)
            elif mode == "branch":
                pass
    
    #
    #
    #
    def getQuantumJosephsonEnergies(self,mode="node",sym_lev="lowest",as_equ=False):
        self._check_args(mode=mode,sym_lev=sym_lev)
        if as_equ:
            sym = self.getQuantumJosephsonEnergies(mode=mode,sym_lev="highest")
            mat = self.getQuantumJosephsonEnergies(mode=mode,sym_lev="lowest")
            return sy.Eq(sym,mat)
        
        if sym_lev == "lowest":
            if mode == "node":
                Dl = self.getLeftDisplacementOpMatrix()
                Dld = self.getLeftDisplacementOpMatrix(adjoint=True)
                Dr = self.getRightDisplacementOpVector()
                Drd = self.getRightDisplacementOpVector(adjoint=True)
                pb = self.getFluxBiasMatrix(mode="branch",form="expphase")
                pbd = self.getFluxBiasMatrix(mode="branch",form="expphase").conjugate()
                J = self.getJosephsonVector().transpose()
                return -J*0.5*(pb*Dl*Dr + pbd*Dld*Drd)
        elif sym_lev == "highest":
            if mode == "node":
                Dl = self.getLeftDisplacementOpMatrix(sym_lev=sym_lev)
                Dld = self.getLeftDisplacementOpMatrix(adjoint=True,sym_lev=sym_lev)
                Dr = self.getRightDisplacementOpVector(sym_lev=sym_lev)
                Drd = self.getRightDisplacementOpVector(adjoint=True,sym_lev=sym_lev)
                pb = self.getFluxBiasMatrix(mode="branch",form="expphase",sym_lev=sym_lev)
                pbd = self.getFluxBiasMatrix(mode="branch",form="expphase",sym_lev=sym_lev).conjugate()
                J = self.getJosephsonVector(sym_lev=sym_lev).transpose()
                return -J*0.5*(pb*Dl*Dr + pbd*Dld*Drd)
    
    #
    #
    #
    def getClassicalHamiltonian(self,mode="node",sym_lev="lowest",as_equ=False):
        self._check_args(mode=mode,sym_lev=sym_lev)
        if as_equ:
            sym = self.getClassicalHamiltonian(mode=mode,sym_lev="highest")
            mat = self.getChargingEnergies(mode=mode,sym_lev="highest")\
            +self.getFluxEnergies(mode=mode,sym_lev="highest")\
            +self.getClassicalJosephsonEnergies(mode=mode,sym_lev="highest")
            return sy.Eq(sym,mat)
        
        if sym_lev == "lowest":
            return self.getChargingEnergies(mode=mode,sym_lev=sym_lev)\
            +self.getFluxEnergies(mode=mode,sym_lev=sym_lev)\
            +self.getClassicalJosephsonEnergies(mode=mode,sym_lev=sym_lev)
        elif sym_lev == "highest":
            return sy.MatrixSymbol(sy.symbols("\\hat{\\mathbf{H}}_{\\mathrm{c}}"),1,1)
    
    #
    #
    #
    def getQuantumHamiltonian(self,mode="node",sym_lev="lowest",as_equ=False):
        self._check_args(mode=mode,sym_lev=sym_lev)
        if as_equ:
            sym = self.getQuantumHamiltonian(mode=mode,sym_lev="highest")
            mat = self.getChargingEnergies(mode=mode,sym_lev="highest")\
            +self.getFluxEnergies(mode=mode,sym_lev="highest")\
            +self.getQuantumJosephsonEnergies(mode=mode,sym_lev="highest")
            return sy.Eq(sym,mat)
        
        if sym_lev == "lowest":
            return self.getChargingEnergies(mode=mode,sym_lev=sym_lev)\
            +self.getFluxEnergies(mode=mode,sym_lev=sym_lev)\
            +self.getQuantumJosephsonEnergies(mode=mode,sym_lev=sym_lev)
        elif sym_lev == "highest":
            return sy.MatrixSymbol(sy.symbols("\\hat{\\mathbf{H}}_{\\mathrm{q}}"),1,1)
    
    #
    #
    #
    def getClassicalPotential(self,mode="node",sym_lev="lowest",as_equ=False):
        self._check_args(mode=mode,sym_lev=sym_lev)
        if as_equ:
            sym = self.getClassicalPotential(mode=mode,sym_lev="highest")
            mat = self.getFluxEnergies(mode=mode,sym_lev="highest")\
            +self.getClassicalJosephsonEnergies(mode=mode,sym_lev="highest")
            return sy.Eq(sym,mat)
        
        if sym_lev == "lowest":
            return self.getFluxEnergies(mode=mode,sym_lev=sym_lev)\
            +self.getClassicalJosephsonEnergies(mode=mode,sym_lev=sym_lev)
        elif sym_lev == "highest":
            return sy.MatrixSymbol(sy.symbols("V_{\\mathrm{c}}"),1,1)
    
    #
    #
    #
    def getLeftDecompFluxVector(self,mode="node",sym_lev="lowest",as_equ=False):
        self._check_args(mode=mode,sym_lev=sym_lev)
        if as_equ:
            sym = self.getLeftDecompFluxVector(sym_lev="highest")
            mat = self.getLeftDecompFluxVector(sym_lev="lowest")
            return sy.Eq(sym,mat)
        
        if sym_lev == "lowest":
            if mode == "node":
                return (self.getNodeToBranchMatrix()+np.abs(self.getNodeToBranchMatrix()))*self.getFluxVector(mode="node")/2
            elif mode == "branch":
                pass
        elif sym_lev == "highest":
            if mode == "node":
                return sy.MatrixSymbol(sy.symbols("\\vec{\\mathbf{\\Phi}}_{nl}"),len(list(self.graph.edges)),1)
        
    
    #
    #
    #
    def getRightDecompFluxVector(self,mode="node",sym_lev="lowest",as_equ=False):
        self._check_args(mode=mode,sym_lev=sym_lev)
        if as_equ:
            sym = self.getRightDecompFluxVector(sym_lev="highest")
            mat = self.getRightDecompFluxVector(sym_lev="lowest")
            return sy.Eq(sym,mat)
        
        if sym_lev == "lowest":
            if mode == "node":
                return (self.getNodeToBranchMatrix()-np.abs(self.getNodeToBranchMatrix()))*self.getFluxVector(mode="node")/2
            elif mode == "branch":
                pass
        elif sym_lev == "highest":
            if mode == "node":
                return sy.MatrixSymbol(sy.symbols("\\vec{\\mathbf{\\Phi}}_{nr}"),len(list(self.graph.edges)),1)
    
    #
    #
    #
    def getNodeToBranchMatrix(self,sym_lev="lowest",as_equ=False):
        self._check_args(sym_lev=sym_lev)
        if as_equ:
            sym = self.getNodeToBranchMatrix(sym_lev="highest")
            mat = self.getNodeToBranchMatrix(sym_lev="lowest")
            return sy.Eq(sym,mat)
        
        if sym_lev == "lowest":
            # Get the incidence matrix of the graph
            I = np.array(nx.incidence_matrix(self.graph,oriented=True).todense()*-1)
            
            # Remove the first row (always corresponds to the ground node)
            Il = np.delete(I,(0),axis=0)
            
            # Transpose of reduced incidence matrix
            return sy.Matrix(Il.T)
        elif sym_lev == "highest":
            return sy.MatrixSymbol(sy.symbols("\\hat{\\mathbf{R}}_{nb}"),len(list(self.graph.edges)),len(list(self.graph.nodes))-1)
    
    #
    #
    #
    def getBranchToNodeMatrix(self,sym_lev="lowest",as_equ=False):
        self._check_args(sym_lev=sym_lev)
        if as_equ:
            sym = self.getBranchToNodeMatrix(sym_lev="highest")
            mat = self.getBranchToNodeMatrix(sym_lev="lowest")
            return sy.Eq(sym,mat)
        
        if sym_lev == "lowest":
            # Get the incidence matrix of the graph
            I = np.array(nx.incidence_matrix(self.graph,oriented=True).todense()*-1)
            
            # Remove the first row (always corresponds to the ground node)
            Il = np.delete(I,(0),axis=0)
            
            # Just the reduced incidence matrix
            return sy.Matrix(Il)
        elif sym_lev == "highest":
            return sy.MatrixSymbol(sy.symbols("\\hat{\\mathbf{R}}_{bn}"),len(list(self.graph.nodes))-1,len(list(self.graph.edges)))
    
    ###################################################################################################################
    #       Circuit Drawing Functions
    ###################################################################################################################
    
    #
    #
    #
    def addSchemElem(self, elm_def, **kwargs):
        elem = self.drawing.add(elm_def,**kwargs)
        self.schem_code_block.append(["add",self.code_elems,elm_def,kwargs])
        self.code_elems += 1
        self.code_lines += 1
        return elem
    
    #
    #
    #
    def pushSchemDraw(self):
        self.drawing.push()
        self.schem_code_block.append(["push"])
        self.code_lines += 1
    
    #
    #
    #
    def popSchemDraw(self):
        self.drawing.pop()
        self.schem_code_block.append(["pop"])
        self.code_lines += 1
    
    #
    #
    #
    def addGroundNode(self, pos):
        self.addSchemElem(e.GND, xy=pos)
        self.pushSchemDraw()
        start_node = self.addSchemElem(e.DOT,lftlabel="0",color=self.gnd_node_clr,xy=pos)
        self.node_circ_coords[0] = start_node.end
    
    #
    #
    #
    def addAnnotation(self,note,node,direction,scaling=0.5,lead_len=0.5,comp_len=0.6):
        
        # Handle orientations
        i = self.__dir_map[direction]
        
        self.pushSchemDraw()
        self.addSchemElem(e.LINE, d=self.__dir_arr[(i)%4], l=lead_len*self.drawing.unit,xy=self.node_circ_coords[node],lftlabel=note)
        self.popSchemDraw()
    
    #
    #
    #
    def addChargeBias(self,direction,node,scaling=0.5,lead_len=1.0,comp_len=0.6):
        
        # Get the end node after drawing
        self.schem_code_block = []
        components = self._get_components_from_strings(["Cg%i" % (node)])
        end_node,end_node_elem = self._add_components(components,direction,node,True,source=True,scaling=scaling,lead_len=lead_len,comp_len=comp_len)
        
        # Get a structured list of the components connected between start and end node
        caps, caps_sym, inds, inds_sym, jjs, jjs_sym = self._get_component_types(components)
        
        self.charge_bias[node] = sy.symbols("%s_{%ie}" % (self.charge_prefix, node))
        self.charge_bias_names["Q%ie"%(node)] = (node)
        self.charge_bias_caps[node] = sy.symbols("C_{g%i}" % (node))
        
        # Update circuit parameters
        self.circ_params["Cg%i" % (node)] = self.charge_bias_caps[node]
        
        # Save the schem code block associated with edge
        # WARNING THIS LIKELY OVERWRITES CODE DEFINED IN addBranch
        self.schem_code_blocks[(node, end_node)] = [self.schem_code_block, {"C":caps,"L":inds,"J":jjs,"Csym":caps_sym,"Lsym":inds_sym,"Jsym":jjs_sym}]
        
        # Add this code to the full code
        self.schem_code.extend(self.schem_code_block)
    
    #
    #
    #
    def addFluxBias(self,loop,name):
        pass
    
    #
    #
    #
    def addLine(self,direction,start_node,lead_len=1.0):
        # Handle orientations
        #i = self.__dir_map[direction]
        
        # Check it doesn't overlap with another, if it does, this is a closure branch
        overlapping_node = self._check_duplicate_node(end_node_elem.end)
        
        # Add the line first
        l = self.addSchemElem(e.LINE, d=direction, l=lead_len*self.drawing.unit,xy=self.node_circ_coords[start_node])
        self.node_circ_coords[start_node] = l.end
        
        # Remove selected node from drawing:
        #
        # FIXME: Requires changing the way code is saved to self.code_blocks -> Need a reference
        #
        #self.drawing._elm_list.pop() # Remove the selected node from drawing
        #self.schem_code_block.pop()
        
        # FIXME: When replacing the node, its coordinates will change, and it will be accessible
        # from the whole line, not a single point -> Change the way node overlapping is
        # done.
        #node_elem = self.addSchemElem(e.DOT,lftlabel="%i"%(start_node),color=self.circ_node_clr,xy=[l.here[0]/2,l.here[1]])
        #return node_elem
    
    #
    #
    #
    def addBranch(self,component_list,direction,start_node,end_node_ground,scaling=0.5,lead_len=1.0,comp_len=0.6,startline=None,endline=None,arm_len=0.0):
        
        # Check there are no inductors in branch if a JJ is present
        prefixes = []
        for comp in component_list:
            prefixes.append(comp[0])
        if self.ind_prefix in prefixes and self.jj_prefix in prefixes:
            raise Exception("cannot have an inductor and JJ in the same branch.")
        
        # Get structured components list
        components = self._get_components_from_strings(component_list)
        
        # Get the end node after drawing
        self.schem_code_block = []
        end_node,end_node_elem = self._add_components(
            components,
            direction,
            start_node,
            end_node_ground,
            scaling=scaling,
            lead_len=lead_len,
            comp_len=comp_len,
            startline=startline,
            endline=endline,
            arm_len=arm_len
        )
        
        # Get a structured list of the components connected between start and end node
        caps, caps_sym, inds, inds_sym, jjs, jjs_sym = self._get_component_types(components)
        
        # Update the parameter list
        all_str = []
        all_str.extend(caps)
        all_str.extend(inds)
        all_str.extend(jjs)
        all_sym = []
        all_sym.extend(caps_sym)
        all_sym.extend(inds_sym)
        all_sym.extend(jjs_sym)
        for i,s in enumerate(all_str):
            self.circ_params[s] = all_sym[i]
        
        # Update the graph and associate degrees of freedom with nodes and branches
        if not end_node_ground:
            
            # Check it doesn't overlap with another, if it does, this is a closure branch
            overlapping_node = self._check_duplicate_node(end_node_elem.end)
            if overlapping_node != None:
                self.drawing._elm_list.pop() # Remove the last node from drawing
                self.schem_code_block.pop()
                self.graph.add_edge(start_node,overlapping_node,etype="C",edge_components={"C":caps,"L":inds,"J":jjs,"Csym":caps_sym,"Lsym":inds_sym,"Jsym":jjs_sym})
                self.node_circ_coords[end_node] = end_node_elem.end # Updates latest ground terminal
                
                # Create symbolic DoFs
                self._add_branch_dofs(start_node,overlapping_node)
                
                # Add empty entry in flux bias vector
                self._add_empty_flux_bias(start_node,overlapping_node)
                
            else:
                self.graph.add_node(end_node,ntype="Circuit")
                self.graph.add_edge(start_node,end_node,etype="S",edge_components={"C":caps,"L":inds,"J":jjs,"Csym":caps_sym,"Lsym":inds_sym,"Jsym":jjs_sym})
                self.node_circ_coords[end_node] = end_node_elem.end
                
                # Create symbolic DoFs
                self._add_branch_dofs(start_node,end_node)
                self._add_node_dofs(end_node)
                
                # Add empty entry in flux bias vector
                self._add_empty_flux_bias(start_node,end_node)
                
                # Set charge bias and gate capacitor terms to zero
                self.charge_bias[end_node] = 0.0
                self.charge_bias_caps[end_node] = 0.0
        else:
            # Subsequent ground edges are closure branches
            end_node = 0
            self.graph.add_edge(start_node,end_node,etype="C",edge_components={"C":caps,"L":inds,"J":jjs,"Csym":caps_sym,"Lsym":inds_sym,"Jsym":jjs_sym})
            self.node_circ_coords[end_node] = end_node_elem.end # Updates latest ground terminal
            
            # Create symbolic DoFs
            self._add_branch_dofs(start_node,end_node)
            
            # Add empty entry in flux bias vector
            self._add_empty_flux_bias(start_node,end_node)
        
        # Update any flux bias terms
        self._update_flux_bias()
        
        # Save the schem code block associated with edge
        self.schem_code_blocks[(start_node,end_node)] = [self.schem_code_block,{"C":caps,"L":inds,"J":jjs,"Csym":caps_sym,"Lsym":inds_sym,"Jsym":jjs_sym}]
        
        # Add this code to the full code
        self.schem_code.extend(self.schem_code_block)
    
    #
    #  Branches need to include inductance
    #
    def coupleBranches(self, M, direction, start_branch, end_node_ground, scaling=0.5, lead_len=1.0, comp_len=0.6, endline=None, arm_len=0.0):
        pass
    
    #
    #  Branch self-inductance neglected, mutual inductor not drawn
    #
    def coupleBranchesImplicitly(self, M, branch1, branch2):
        
        if M[0] != self.mut_prefix:
            raise Exception("Invalid mutual inductance symbol, use the '%s' prefix." % (self.mut_prefix))
        
        # Create the symbol
        self.nmut += 1
        self.circ_params[M] = sy.symbols("%s_{%s}"%(M[0],M[1:]))
        
        # Get the existing mutually coupled branch definitions
        branch_mutuals = nx.get_edge_attributes(self.graph, name='mutual')
        
        # Initialise them if they exist already
        if branch_mutuals == {}:
            branch_mutuals = self._gen_branch_mutuals()
        
        # Set the associations
        branch_mutuals[branch1].append((branch2,M))
        branch_mutuals[branch2].append((branch1,M))
        nx.set_edge_attributes(self.graph, branch_mutuals, name='mutual')
    
    #
    #
    #
    def remBranch(self,edge):
        block,comps = self.schem_code_blocks[edge]
        
        # Get index range to delete from circuit
        block_range_min = block[0][1]
        block_range_max = block[-1][1]
        del self.drawing._elm_list[block_range_min:block_range_max]
        
        # Correct the number of components
        self.ncap -= len(comps["C"])
        self.nind -= len(comps["L"])
        self.njjs -= len(comps["J"])
        
        # Remove the edge from the graph
        self.graph.remove_edge(edge[0],edge[1])
        
        # Merge the two nodes if they only have one edge each left
        #if self.graph.degree(edge[0]) == 1 and self.graph.degree(edge[1]) == 1:
        #    self.graph.remove_node(edge[1])
            
            # 
    
    #
    ## Replace an existing branch with a new definition
    #
    def repBranch(self,component_list,edge):
        pass
    
    ##
    #
    #
    #
    def coupleLinearResonator(self, resonator_name, pos, coupling_type='capacitive', coupling_approx='RWA', d=['up','right']):
        # FIXME: Hardcoded to node representation
        # Multiple subsystems not implemented yet
        if self.coupled_subsys != {}:
            raise Exception("this circuit already has a coupled subsystem. Adding more will be implemented in the future.")
        
        # FIXME: Need to save the original circuit parameters and other properties
        
        
        # FIXME: Doesn't consider parasitic terms, should it?
        if coupling_type == "capacitive":
            self._check_args(node=pos,cpl_type=coupling_type)
            node_sel = pos
            nodes = list(self.graph.nodes)
            final_node = max(nodes)
            node_new = final_node+1
            
            # Create the new circuit
            Cc = self.cap_prefix+"%i%s"%(node_sel,resonator_name)
            Cr = self.cap_prefix+"%s"%(resonator_name)
            Lr = self.ind_prefix+"%s"%(resonator_name)
            if self.rcircuit is None:
                self.rcircuit = self.copy()
            self.rcircuit.addBranch([Cc],d[0],node_sel,False)
            self.rcircuit.addBranch([Cr,Lr],d[1],node_new,True)
        elif coupling_type == "inductive":
            self._check_args(node=pos,cpl_type=coupling_type)
            edge_sel = pos
            
            # Need to add mutual inductance to inductance matrix first,
            # then handle which node gets grounded and which one serves as the coupling node.
            # Mutual not implemented yet
            
            #nodes = list(self.graph.nodes)
            #node_new = max(nodes)+1
            
            # Create the new circuit
            #Cc = self.cap_prefix+"%i%s"%(node_sel,resonator_name)
            #Cr = self.cap_prefix+"%s"%(resonator_name)
            #Lr = self.ind_prefix+"%s"%(resonator_name)
            #rcircuit = self.copy()
            #rcircuit.addBranch([Cc],d[0],node_sel,False)
            #rcircuit.addBranch([Cr,Lr],d[1],node_new,True)
        elif coupling_type == "galvanic":
            self._check_args(node=pos,cpl_type=coupling_type)
            node_sel = pos
            
            # Not sure what to do here yet
        
        # Get the corrected qubit and resonator inverse matrices
        Cinv = self.rcircuit.getInverseCapacitanceMatrix()
        Linv = self.rcircuit.getInverseInductanceMatrix()
        qbtCinv = Cinv[:final_node,:final_node]
        resCinv = Cinv[node_new-1,node_new-1]
        qbtLinv = Linv[:final_node,:final_node]
        resLinv = Linv[node_new-1,node_new-1]
        
        # Save the loaded system matrices
        self.rcircuit_sys_matrices["Cinv"] = qbtCinv
        self.rcircuit_sys_matrices["Linv"] = qbtLinv
        #self.rcircuit_sys_matrices["Cinv_n"] = qbtCinv_n
        #self.rcircuit_sys_matrices["Linv_b"] = qbtLinv_b
        
        # Update the original circuit parameters list to include the new ones
        
        
        # Get the corrected resonator impedance and frequency
        Zr = sy.sqrt(resCinv/resLinv)
        wr = sy.sqrt(resCinv*resLinv)
        
        # Get the interaction term (NOTE: we are ignoring interacting terms involving other nodes, they are likely too weak to make a difference)
        intCinv = Cinv[0,node_new-1]
        intLinv = Linv[0,node_new-1]
        gC = intCinv/sy.sqrt(2*Zr) # Always use oscillator basis when resonator is defined as subsystem
        gL = intLinv*sy.sqrt(Zr/2)
        
        # Define the photon number
        np = "n%s"%(resonator_name)
        ns = sy.symbols("n_\\mathrm{%s}"%(resonator_name))
        
        # Create a coupled subsystem
        self.coupled_subsys[resonator_name] = {
            "subsys_type":"linear_resonator",
            "cpl_type":coupling_type,
            "cpl_approx":coupling_approx,
            "cpl_node":node_sel,
            "cpl_gC":gC,
            "cpl_gL":gL,
            "res_Cinv":resCinv,
            "res_Linv":resLinv,
            "parameters":{
                np:ns
            },
            "derived_parameters":{
                "Z":Zr,
                "w":wr,
                "gC":gC,
                "gL":gL
            },
            "derived_parameter_units":{
                "Z":"Impe",
                "w":"Freq",
                "gC":"ChgOscCpl",
                "gL":"FlxOscCpl"
            }
        }
        return self.rcircuit
    
    #
    #
    #
    def drawCircuit(self, filename=None, output="svg", inline=False):
        self.drawing.draw(showplot=False)
        if inline:
            return None
        
        if filename is not None:
            self.drawing.save(filename)
            plt.close()
            return None
        
        # Get output into a buffered stream
        imgdata = io.StringIO()
        self.drawing.fig.savefig(imgdata,format=output,bbox_extra_artists=self.drawing.ax.get_default_bbox_extra_artists(), bbox_inches='tight', transparent=False)
        plt.close()
        return imgdata.getvalue()
    
    #
    #
    #
    def newDrawingFromSchem(self):
        newdrawing = schem.Drawing()
        for i in range(len(self.schem_code)):
            if self.schem_code[i][0] == "add":
                newdrawing.add(self.schem_code[i][2],**self.schem_code[i][3])
            elif self.schem_code[i][0] == "push":
                newdrawing.push()
            elif self.schem_code[i][0] == "pop":
                newdrawing.pop()
        return newdrawing
    
    ###################################################################################################################
    #       Hierarchical Planning Functions
    ###################################################################################################################
    
    ##
    #
    # edge_groups is: {"circ1":[(0,1),(1,0)], "circ2": ...}
    #
    # The missing edges represent coupling terms between black boxes
    #
    #
    def hierarchize(self, edge_groups):
        
        if len(edge_groups) < 2:
            raise Exception("At least two groups of edges are required to define subsystems.")
        
        # Check the specified edges exist and get the edges that are not present in the groups
        edge_list = self.getEdgeList()
        node_list = self.getNodeList()
        
        # Get nodes associated with edges and the indices associated with each of them
        found_edges = []
        group_nodes = {}
        group_node_indices = {}
        group_edge_indices = {}
        for k,group in edge_groups.items():
            
            # Check edges
            local_group_nodes = []
            for edge in group:
                if edge not in edge_list:
                    raise Exception("Edge '%s' does not exist." % repr(edge))
                found_edges.append(edge)
                
                # Find nodes
                for node in edge:
                    if node != 0 and node not in local_group_nodes:
                        local_group_nodes.append(node)
            
            # Structure nodes and indices
            group_nodes[k] = local_group_nodes
            group_node_indices[k] = [node_list.index(n) for n in local_group_nodes]
            group_edge_indices[k] = [edge_list.index(e) for e in group]
            
            # Create SubCircuitSpec instances
            nodes = local_group_nodes.copy()
            nodes.append(0)
            self.subcircuits[k] = SubCircuitSpec(k, self, nodes, found_edges)
        
        # Get edges not in a group (coupling edges)
        for edge in edge_list:
            if edge not in found_edges:
                self.subcircuit_coupling_edges.append(edge)
        
        # Create a series of black boxes for each group?
    
    def getSubCircuit(self, name):
        return self.subcircuits[name]
    
    def getSubCircuitEdges(self, name):
        pass
    
    def getSubCircuitNodes(self, name):
        pass
    
    def getSubCircuitQuantumHamiltonian(self, name, **kwargs):
        return self.subcircuits[name].getQuantumHamiltonian(**kwargs)
    
    def getSubCircuitInteractionHamiltonian(self, **kwargs):
        
        # Empty if there is no subcircuit definition
        if self.subcircuits == {}:
            return 0.0
        
        # Generate masked inverse matrices from the coupled edges
        Cinv = self.getInverseCapacitanceMatrix()
        Linv = self.getInverseInductanceMatrix()
        C = self.getCapacitanceMatrix()
        Dl = self.getLeftDisplacementOpMatrix()
        Dld = self.getLeftDisplacementOpMatrix(adjoint=True)
        Dr = self.getRightDisplacementOpVector()
        Drd = self.getRightDisplacementOpVector(adjoint=True)
        pb = self.getFluxBiasMatrix(mode="branch",form="expphase")
        pbd = self.getFluxBiasMatrix(mode="branch",form="expphase").conjugate()
        J = self.getJosephsonVector().transpose()
        
        # Node representation interaction Hamiltonian
        N = len(self.getNodeList())
        for i in range(N):
            for j in range(N):
                edge1 = (i+1, j+1)
                edge2 = (j+1, i+1)
                if edge1 not in self.subcircuit_coupling_edges and edge2 not in self.subcircuit_coupling_edges:
                    
                    Cinv[i,j] = 0.0
                    Cinv[j,i] = 0.0
                    
                    Linv[i,j] = 0.0
                    Linv[j,i] = 0.0
        
        # Generate the terms of the Hamiltonian with masked inverse matrices
        Ec = 0.5*(self.getChargeVector()+self.getChargeBiasVector()).transpose()\
            *Cinv\
            *(self.getChargeVector()+self.getChargeBiasVector())
        El = 0.5*(self.getFluxVector()).transpose()\
                *Linv\
                *(self.getFluxVector())
        Ej = 0.0#-J*0.5*(pb*Dl*Dr + pbd*Dld*Drd)
        
        return Ec + El# + Ej
        #self._check_args(mode=mode,sym_lev=sym_lev)
        #if as_equ:
        #    sym = self.getQuantumHamiltonian(mode=mode,sym_lev="highest")
        #    mat = self.getChargingEnergies(mode=mode,sym_lev="highest")\
        #    +self.getFluxEnergies(mode=mode,sym_lev="highest")\
        #    +self.getQuantumJosephsonEnergies(mode=mode,sym_lev="highest")
        #    return sy.Eq(sym,mat)
        
        #if sym_lev == "lowest":
        #    return self.getChargingEnergies(mode=mode,sym_lev=sym_lev)\
        #    +self.getFluxEnergies(mode=mode,sym_lev=sym_lev)\
        #    +self.getQuantumJosephsonEnergies(mode=mode,sym_lev=sym_lev)
        #elif sym_lev == "highest":
        #    return sy.MatrixSymbol(sy.symbols("\\hat{\\mathbf{H}}_{\\mathrm{q}}"),1,1)
    
    def drawHierarchicalCircuit(self):
        pass
    
    ###################################################################################################################
    #       Generic Functions
    ###################################################################################################################
    
    ##
    #
    #
    #
    def copy(self):
        return copy.copy(self)
    
    ###################################################################################################################
    #       Internal Functions
    ###################################################################################################################
    
    ## Detect when two nodes overlap
    #
    #
    #
    def _check_duplicate_node(self,node_coords):
        for i in list(self.graph.nodes):
            if np.allclose(self.node_circ_coords[i],node_coords):
                return i
        return None
    
    #
    #
    #
    def _add_branch_dofs(self,start_node,end_node):
        self.branch_dofs[(start_node,end_node)] = \
                sy.symbols("%s_{%i%i} %s_{%i%i} %s_{%i%i} %s_{%i%i}" % \
                (
                    self.flux_prefix,start_node,end_node,
                    self.charge_prefix,start_node,end_node,
                    "D",start_node,end_node,
                    "D^{\\dagger}",start_node,end_node
                ),commutative=False)
        self.classical_branch_dofs[(start_node,end_node)] = sy.symbols("%s_{%i%i} %s_{%i%i} %s_{%i%i}" % \
        (self.flux_prefix,start_node,end_node,self.redflux_prefix,start_node,end_node,self.charge_prefix,start_node,end_node))
    
    #
    #
    #
    def _add_node_dofs(self,node):
        self.node_dofs[node] = \
                sy.symbols("%s_{%i} %s_{%i} %s_{%i} %s_{%i}" % \
                (
                    self.flux_prefix,node,
                    self.charge_prefix,node,
                    "D",node,
                    "D^{\\dagger}",node
                ),commutative=False)
        self.classical_node_dofs[node] = sy.symbols("%s_{%i} %s_{%i} %s_{%i}" % \
        (self.flux_prefix,node,self.redflux_prefix,node,self.charge_prefix,node))
    
    #
    #
    #
    def _add_empty_flux_bias(self,start_node,end_node):
        self.flux_bias_prefactor[(start_node,end_node)] = 1.0
        self.flux_bias[(start_node,end_node)] = 0.0
        self.red_flux_bias[(start_node,end_node)] = 0.0
        self.exp_flux_bias[(start_node,end_node)] = 1.0
    
    #
    #
    #
    def _add_flux_bias(self,start_node,end_node):
        
        # Normal flux term, only this one is actually used by HamilSpec
        self.flux_bias_prefactor[(start_node,end_node)] = 1.0
        self.flux_bias[(start_node,end_node)] = sy.symbols("%s_{%i%ie}" % (self.flux_prefix,start_node,end_node))
        self.flux_bias_names["phi%i%ie"%(start_node,end_node)] = (start_node,end_node)
        
        # Only here for pretty equations
        self.red_flux_bias[(start_node,end_node)] = sy.symbols("%s_{%i%ie}" % (self.redflux_prefix,start_node,end_node))
        self.exp_flux_bias[(start_node,end_node)] = sy.symbols("e^{i%s_{%i%ie}}" % (self.redflux_prefix,start_node,end_node))
    
    #
    #
    #
    def _check_args(self, **kwargs):
        if 'mode' in kwargs and kwargs['mode'] not in self.__modes:
            raise Exception("mode '%s' is not a valid mode. Available modes are %s" % (mode,repr(self.__modes)))
        if 'sym_lev' in kwargs and kwargs['sym_lev'] not in self.__sym_lev:
            raise Exception("sym_lev '%s' is not a valid symbolic level. Available levels are %s" % (sym_lev,repr(self.__sym_lev)))
        if 'dof' in kwargs and kwargs['dof'] not in list(self.__dof_map.keys()):
            raise Exception("DoF name '%s' is not a valid name. Available names are %s" % (dof,repr(list(self.__dof_map.keys()))))
        if 'node' in kwargs and kwargs['node'] not in list(self.graph.nodes)[1:]:
            raise Exception("node '%i' does no exist. Available nodes are %s" % (node,repr(list(self.graph.nodes)[1:])))
        if 'edge' in kwargs and kwargs['edge'] not in list(self.graph.edges):
            raise Exception("edge '%s' does no exist. Available edges are %s" % (edge,repr(list(self.graph.edges))))
        if 'fl_form' in kwargs and kwargs['fl_form'] not in self.__flux_op_form:
            raise Exception("flux operator form '%s' is not a valid form. Available forms are %s" % (fl_form,repr(self.__flux_op_form)))
        if 'ch_form' in kwargs and kwargs['ch_form'] not in self.__charge_op_form:
            raise Exception("charge operator form '%s' is not a valid form. Available forms are %s" % (ch_form,repr(self.__charge_op_form)))
        if 'bias' in kwargs and kwargs['bias'] not in self.__bias_types:
            raise Exception("bias type '%s' is not a valid type. Available types are %s" % (bias,repr(self.__bias_types)))
        if 'cpl_type' in kwargs and kwargs['cpl_type'] not in self.__cpl_types:
            raise Exception("coupling type '%s' is not a valid type. Available types are %s" % (cpl_type,repr(self.__cpl_types)))
        if 'subsys_type' in kwargs and kwargs['subsys_type'] not in self.__subsys_types:
            raise Exception("subsystem type '%s' is not a valid type. Available types are %s" % (subsys_type,repr(self.__subsys_types)))
    
    #
    #
    #
    def _gen_branch_mutuals(self):
        return {k:[] for k in self.graph.edges}
    
    #
    #
    #
    def _get_component_types(self,components):
        elems = components['elements']
        labels = components['labels']
        
        # Strings
        caps = []
        inds = []
        jjs = []
        
        # Symbolic Vars
        caps_sym = []
        inds_sym = []
        jjs_sym = []
        
        for i,elem in enumerate(elems):
            if elem == e.CAP:
                self.ncap += 1
                caps.append(labels[i])
                caps_sym.append(sy.symbols("%s_{%s}"%(labels[i][0],labels[i][1:])))
            elif elem == e.INDUCTOR:
                self.nind += 1
                inds.append(labels[i])
                inds_sym.append(sy.symbols("%s_{%s}"%(labels[i][0],labels[i][1:])))
            elif elem == e.JJ:
                self.njjs += 1
                jjs.append(labels[i])
                jjs_sym.append(sy.symbols("%s_{%s}"%(labels[i][0],labels[i][1:])))
        return (caps,caps_sym,inds,inds_sym,jjs,jjs_sym)
    
    #
    #
    #
    def _get_components_from_strings(self,strings):
        components = {'elements':[],'labels':[]}
        # Capacitors first
        for s in strings:
            if s[0] == self.cap_prefix:
                components['labels'].append(s)
                components['elements'].append(e.CAP)
        for s in strings:
            if s[0] == self.ind_prefix:
                components['labels'].append(s)
                components['elements'].append(e.INDUCTOR)
        for s in strings:
            if s[0] == self.jj_prefix:
                components['labels'].append(s)
                components['elements'].append(e.JJ)
        #for s in strings:
        #    if s[0] == self.mut_prefix: # Should never appear in parallel with anything
        #        components['labels'].append(s)
        #        components['elements'].append(e.JJ)
        return components
    
    #
    #
    #
    def _add_components(self,components,direction,start_node,end_node_ground,source=False,scaling=0.5,lead_len=1.0,comp_len=0.6,startline=None,endline=None,arm_len=0.0):
        
        # Get components
        elems = components['elements']
        labels = components['labels']
        
        # Handle orientations
        i = self.__dir_map[direction]
        
        # Setup spacing variables
        c = len(elems)
        spacing = scaling*self.drawing.unit
        tlen = scaling*self.drawing.unit*(c-1)
        elen = (lead_len-comp_len)*self.drawing.unit
        alen = abs(arm_len*self.drawing.unit)
        
        # Add startline if required
        pos = self.node_circ_coords[start_node]
        if startline is not None:
            d,l = startline
            pos = self.addSchemElem(e.LINE, d=d, l=l*self.drawing.unit,xy=self.node_circ_coords[start_node]).end
        
        # Draw branch components
        if arm_len > 0.0:
            self.addSchemElem(e.LINE, d=self.__dir_arr[(i+1)%4], l=alen,xy=pos)
            self.addSchemElem(e.LINE, d=self.__dir_arr[(i)%4], l=elen/2)
        elif arm_len < 0.0:
            self.addSchemElem(e.LINE, d=self.__dir_arr[(i-1)%4], l=alen,xy=pos)
            self.addSchemElem(e.LINE, d=self.__dir_arr[(i)%4], l=elen/2)
        else:
            self.addSchemElem(e.LINE, d=self.__dir_arr[(i)%4], l=elen/2,xy=pos)
        self.addSchemElem(e.LINE, d=self.__dir_arr[(i+1)%4], l=tlen/2)
        for j,elem in enumerate(elems):
            self.pushSchemDraw()
            self.addSchemElem(elem,label=("$%s_{%s}$"%(labels[j][0],labels[j][1:])),d=self.__dir_arr[(i)%4],l=comp_len*self.drawing.unit)
            if j < c-1:
                self.addSchemElem(e.LINE, d=self.__dir_arr[(i-1)%4], l=spacing)
                self.popSchemDraw()
                self.addSchemElem(e.LINE, d=self.__dir_arr[(i-1)%4], l=spacing)
        self.addSchemElem(e.LINE, d=self.__dir_arr[(i+1)%4], l=tlen/2)
        self.addSchemElem(e.LINE, d=self.__dir_arr[(i)%4], l=elen/2)
        if arm_len > 0.0:
            self.addSchemElem(e.LINE, d=self.__dir_arr[(i-1)%4], l=alen)
        elif arm_len < 0.0:
            self.addSchemElem(e.LINE, d=self.__dir_arr[(i+1)%4], l=alen)
        
        # Add endline if required
        if endline is not None:
            d,l = endline
            self.addSchemElem(e.LINE, d=d, l=l*self.drawing.unit)
        
        if not source:
            if not end_node_ground:
                # Get the next usable node number
                end_node = max(list(self.graph.nodes))+1
                
                # Draw and get the new circuit node element
                end_node_elem = self.addSchemElem(e.DOT,lftlabel="%i"%(end_node),color=self.circ_node_clr)
            else:
                end_node = 0
                end_node_elem = self.addSchemElem(e.DOT,lftlabel="0",color=self.gnd_node_clr)
                self.addSchemElem(e.GND)
        else:
            # Could draw a source here
            
            # Special node for voltage source
            end_node = -start_node
            end_node_elem = None
        
        # Draw end node and return it
        return (end_node,end_node_elem)
    
    #
    #
    #
    def _update_flux_bias(self):
        
        # Get the list of superconducting loops and edge components
        loops = self.getSuperconductingLoopEdges()
        if loops == []:
            return
        edge_comp = nx.get_edge_attributes(self.graph, 'edge_components')
        edge_types = nx.get_edge_attributes(self.graph, 'etype')
        
        # Update the flux bias entry on a single JJ term in a loop.
        for loop in loops:
            for edge in loop[::-1]: # Hack to look at closure branches first
                if edge_comp[edge]['J'] != []:
                    self._add_flux_bias(edge[0],edge[1])
                    break # Only add a single bias term per loop
    
    ## This effectively makes a deepcopy
    #
    #
    #
    def __copy__(self):
        
        # Reconstruct the class
        new = CircuitSpec(self.circ_name,*self.kwargs)
        
        # Set all the attributes
        new.ncap = copy.deepcopy(self.ncap)
        new.nind = copy.deepcopy(self.nind)
        new.njjs = copy.deepcopy(self.njjs)
        new.schem_code = copy.deepcopy(self.schem_code)
        new.schem_code_block = copy.deepcopy(self.schem_code_blocks)
        new.schem_code_blocks = copy.deepcopy(self.schem_code_blocks)
        new.node_circ_coords = copy.deepcopy(self.node_circ_coords)
        new.code_elems = copy.deepcopy(self.code_elems)
        new.code_lines = copy.deepcopy(self.code_lines)
        new.circ_params = copy.deepcopy(self.circ_params)
        new.cap_prefix = self.cap_prefix # Ref!
        new.ind_prefix = self.ind_prefix # Ref!
        new.jj_prefix = self.jj_prefix # Ref!
        new.flux_prefix = self.flux_prefix # Ref!
        new.redflux_prefix = self.redflux_prefix # Ref!
        new.charge_prefix = self.charge_prefix # Ref!
        new.redcharge_prefix = self.redcharge_prefix # Ref!
        new.phi0 = sy.symbols("\\Phi_0")
        new.Rq0 = sy.symbols("R_Q")
        new.qcp = 2*sy.symbols("e")
        new.pi = sy.pi
        new.node_dofs = copy.deepcopy(self.node_dofs)
        new.classical_node_dofs = copy.deepcopy(self.classical_node_dofs)
        new.branch_dofs = copy.deepcopy(self.branch_dofs)
        new.classical_branch_dofs = copy.deepcopy(self.classical_branch_dofs)
        new.flux_bias = copy.deepcopy(self.flux_bias)
        new.flux_bias_prefactor = copy.deepcopy(self.flux_bias_prefactor)
        new.red_flux_bias = copy.deepcopy(self.red_flux_bias)
        new.exp_flux_bias = copy.deepcopy(self.exp_flux_bias)
        new.charge_bias = copy.deepcopy(self.charge_bias)
        new.red_charge_bias = copy.deepcopy(self.red_charge_bias)
        new.flux_bias_names = copy.deepcopy(self.flux_bias_names)
        new.charge_bias_names = copy.deepcopy(self.charge_bias_names)
        new.flux_bias_mutuals = copy.deepcopy(self.flux_bias_mutuals)
        new.charge_bias_caps = copy.deepcopy(self.charge_bias_caps)
        new.circ_node_clr = self.circ_node_clr # Ref!
        new.gnd_node_clr = self.gnd_node_clr # Ref!
        new.coupled_subsys = copy.deepcopy(self.coupled_subsys)
        
        # Finally regen the drawing and the graph
        new.drawing = self.newDrawingFromSchem()
        new.graph = self.graph.copy()
        return new


class SubCircuitSpec(CircuitSpec):
    """This class is instantiated by the :func:`hierarchize` function to allow solving subcircuits individually.
    """
    
    def __init__(self, circuit_name, parent_circuit, nodes, edges):
        
        if parent_circuit is None:
            raise Exception("A parent circuit must be specified for SubCircuitSpec instance.")
        self.parent_circuit = parent_circuit
        
        # Init data structures
        self.circ_name = circuit_name
        self.graph = parent_circuit.graph.subgraph(nodes).copy()
        
        # Node and Edge index mappings
        self.node_indices = {node: self.getNodeList().index(node) for node in self.getNodeList()}
        
        # Define physical constant symbols
        self.phi0 = sy.symbols("\\Phi_0") # Flux quantum
        self.Rq0 = sy.symbols("R_Q")      # Resistance quantum
        self.qcp = 2*sy.symbols("e")      # Cooper pair charge
        self.pi = sy.pi                   # Pi
        
        # TO BE UPDATED BY PARENT_CIRCUIT
        self.kwargs = parent_circuit.kwargs
    
        # Tracker variables
        self.ncap = parent_circuit.ncap         # Total number of capacitors
        self.nind = parent_circuit.nind         # Total number of inductors
        self.njjs = parent_circuit.njjs         # Total number of JJs
        self.nmut = parent_circuit.nmut         # Total number of mutual inductors
        
        # Circuit parameter symbols
        self.circ_params = parent_circuit.circ_params
        
        # Assign element prefixes
        self.cap_prefix = parent_circuit.cap_prefix
        self.ind_prefix = parent_circuit.ind_prefix
        self.jj_prefix = parent_circuit.jj_prefix
        self.mut_prefix = parent_circuit.mut_prefix
        
        # Assign degree of freedom prefixes
        self.flux_prefix = parent_circuit.flux_prefix
        self.redflux_prefix = parent_circuit.redflux_prefix
        self.charge_prefix = parent_circuit.charge_prefix
        self.redcharge_prefix = parent_circuit.redcharge_prefix
        
        # Degrees of freedom
        self.node_dofs = parent_circuit.node_dofs
        self.classical_node_dofs = parent_circuit.classical_node_dofs
        self.branch_dofs = parent_circuit.branch_dofs
        self.classical_branch_dofs = parent_circuit.classical_branch_dofs
        
        # Flux and Charge bias terms (edge -> symbol mapping)
        self.flux_bias = parent_circuit.flux_bias
        self.flux_bias_prefactor = parent_circuit.flux_bias_prefactor
        self.red_flux_bias = parent_circuit.red_flux_bias
        self.exp_flux_bias = parent_circuit.exp_flux_bias
        self.charge_bias = parent_circuit.charge_bias
        self.red_charge_bias = parent_circuit.red_charge_bias
        
        # Flux and Charge bias names (name -> edge)
        self.flux_bias_names = parent_circuit.flux_bias_names
        self.charge_bias_names = parent_circuit.charge_bias_names
        
        # Flux and Charge bias components
        self.flux_bias_mutuals = parent_circuit.flux_bias_mutuals
        self.charge_bias_caps = parent_circuit.charge_bias_caps
        
        # FOR FUTURE USE
        """
        self.kwargs = [elem_prefixes, dof_prefixes, gnd_node_clr, circ_node_clr]
        self.graph = nx.DiGraph(circuit_name=self.circ_name) # TODO: Think of using MultiDiGraph to allow more than two branches between nodes
        self.drawing = schem.Drawing()
        self.schem_code = []                    # Holds all elements necessary to regenerate the circuit as is
        self.schem_code_block = []              # Block of drawing code associated with edge (temporary)
        self.schem_code_blocks = {}             # Holds all blocks necessary to regenerate/edit circuit edges
        self.node_circ_coords = {}              # Holds the circuit diagram coordinates of the nodes
        self.code_elems = 0                     # Counts the number of SchemDraw elements
        self.code_lines = 0
        # Node properties (circuit diagram)
        self.circ_node_clr = circ_node_clr
        self.gnd_node_clr = gnd_node_clr
        
        # Coupled subsystems
        self.coupled_subsys = {}
        
        # Extended circuit
        self.rcircuit = None
        
        # Extended circuit loaded system matrices
        self.rcircuit_sys_matrices = {}
        
        # Add the first ground node
        self.addSchemElem(e.GND)
        self.pushSchemDraw()
        start_node = self.addSchemElem(e.DOT,lftlabel="0",color=self.gnd_node_clr)
        self.graph.add_node(0,ntype="Ground")
        self.node_circ_coords[0] = start_node.end
        
        # Add this code to the full code
        self.schem_code.extend(self.schem_code_block)
        """
    
    ###################################################################################################################
    #       (Overloaded) Symbolic Math and Equation Functions
    ###################################################################################################################
    
    def getCapacitanceMatrix(self, mode="node", sym_lev="lowest", as_equ=False):
        self._check_args(mode=mode,sym_lev=sym_lev)
        if as_equ:
            sym = self.getCapacitanceMatrix(mode=mode,sym_lev="highest")
            mat = self.getCapacitanceMatrix(mode=mode,sym_lev="lowest")
            return sy.Eq(sym,mat)
        
        if sym_lev == "lowest":
            
            node_list = self.parent_circuit.getNodeList()
            node_indices = [node_list.index(n) for n in self.getNodeList()]
            
            M = self.parent_circuit.getCapacitanceMatrix()[node_indices[0]:node_indices[-1]+1,node_indices[0]:node_indices[-1]+1]
            
            if mode == "node":
                return M
            elif mode == "branch":
                return self.getNodeToBranchMatrix()*M*self.getBranchToNodeMatrix()
        elif sym_lev == "highest":
            if mode == "node":
                return sy.MatrixSymbol(sy.symbols("\\hat{\\mathbf{C}}_{n}"),len(list(self.graph.nodes))-1,len(list(self.graph.nodes))-1)
            elif mode == "branch":
                return sy.MatrixSymbol(sy.symbols("\\hat{\\mathbf{C}}_{b}"),len(list(self.graph.edges)),len(list(self.graph.edges)))
        
    def getGateCapacitanceMatrix(self, mode="node", sym_lev="lowest", as_equ=False):
        self._check_args(mode=mode,sym_lev=sym_lev)
        if as_equ:
            sym = self.getGateCapacitanceMatrix(mode=mode,sym_lev="highest")
            mat = self.getGateCapacitanceMatrix(mode=mode,sym_lev="lowest")
            return sy.Eq(sym,mat)
        
        if sym_lev == "lowest":
            M = sy.eye(len(list(self.graph.nodes))-1) - sy.eye(len(list(self.graph.nodes))-1)
            for node in list(self.graph.nodes):
                if node == 0:
                    continue
                M[self.node_indices[node],self.node_indices[node]] = self.parent_circuit.charge_bias_caps[node]
            if mode == "node":
                return M
            elif mode == "branch":
                return self.getNodeToBranchMatrix()*M*self.getBranchToNodeMatrix()
        elif sym_lev == "highest":
            if mode == "node":
                return sy.MatrixSymbol(sy.symbols("\\hat{\\mathbf{C}}_{gn}"),len(list(self.graph.nodes))-1,len(list(self.graph.nodes))-1)
            elif mode == "branch":
                return sy.MatrixSymbol(sy.symbols("\\hat{\\mathbf{C}}_{gb}"),len(list(self.graph.edges)),len(list(self.graph.edges)))
        
    def getInverseCapacitanceMatrix(self, mode="node", sym_lev="lowest", as_equ=False):
        self._check_args(mode=mode,sym_lev=sym_lev)
        if as_equ:
            sym = self.getInverseCapacitanceMatrix(mode=mode,sym_lev="highest")
            mat = self.getInverseCapacitanceMatrix(mode=mode,sym_lev="lowest")
            return sy.Eq(sym,mat)
        
        if sym_lev == "lowest":
            node_list = self.parent_circuit.getNodeList()
            node_indices = [node_list.index(n) for n in self.getNodeList()]
            return self.parent_circuit.getInverseCapacitanceMatrix()[node_indices[0]:node_indices[-1]+1,node_indices[0]:node_indices[-1]+1]
        elif sym_lev == "highest":
            return (self.getCapacitanceMatrix(mode=mode,sym_lev="highest")+self.getGateCapacitanceMatrix(mode=mode,sym_lev="highest"))**(-1)
    
    def getChargeVector(self, mode="node", sym_lev="lowest", as_equ=False):
        self._check_args(mode=mode,sym_lev=sym_lev)
        if as_equ:
            sym = self.getChargeVector(mode=mode,sym_lev="highest")
            mat = self.getChargeVector(mode=mode,sym_lev="lowest")
            return sy.Eq(sym,mat)
        
        if sym_lev == "lowest":
            if mode == "node":
                return sy.Matrix([self.parent_circuit.node_dofs[n][1] for n in list(self.graph.nodes)[1:]])
            elif mode == "branch":
                return sy.Matrix([self.parent_circuit.branch_dofs[e][1] for e in list(self.graph.edges)])
        elif sym_lev == "highest":
            if mode == "node":
                return sy.MatrixSymbol(sy.symbols("\\vec{\\mathbf{Q}}_{n}"),len(list(self.graph.nodes))-1,1)
            elif mode == "branch":
                return sy.MatrixSymbol(sy.symbols("\\vec{\\mathbf{Q}}_{b}"),len(list(self.graph.edges)),1)
    
    def getChargeBiasVector(self, mode="node", form="charge", sym_lev="lowest", as_equ=False):
        self._check_args(mode=mode,sym_lev=sym_lev,ch_form=form)
        if as_equ:
            sym = self.getChargeBiasVector(form=form,sym_lev="highest")
            mat = self.getChargeBiasVector(form=form,sym_lev="lowest")
            return sy.Eq(sym,mat)
        
        if sym_lev == "lowest":
            if form == "charge":
                bias_vec = list(np.zeros(len(list(self.graph.nodes))-1))
                nodes = list(self.graph.nodes)
                nodes.remove(0)
                for i,node in enumerate(nodes):
                    bias_vec[i] = self.parent_circuit.charge_bias[node]
            elif form == "phase": # THIS WILL BREAK, FIXME
                bias_vec = list(np.zeros(len(list(self.graph.edges))))
                for i,node in enumerate(list(self.graph.edges)):
                    if node in self.getClosureBranches():
                        bias_vec[i] = sy.symbols("%s_{%i%ie}" % (self.redcharge_prefix,edge[0],edge[1]))
            if mode == "node":
                return sy.Matrix(bias_vec)
            elif mode == "branch":
                return self.getNodeToBranchMatrix()*sy.Matrix(bias_vec)
        elif sym_lev == "highest":
            if mode == "node":
                if form == "charge":
                    return sy.MatrixSymbol(sy.symbols("\\vec{\\mathbf{Q}}_{ne}"),len(list(self.graph.nodes))-1,1)
                elif form == "phase":
                    return sy.MatrixSymbol(sy.symbols("\\vec{\\mathbf{q}}_{ne}"),len(list(self.graph.nodes))-1,1)
            elif mode == "branch":
                if form == "charge":
                    return sy.MatrixSymbol(sy.symbols("\\vec{\\mathbf{Q}}_{be}"),len(list(self.graph.edges)),1)
                elif form == "phase":
                    return sy.MatrixSymbol(sy.symbols("\\vec{\\mathbf{q}}_{be}"),len(list(self.graph.edges)),1)
    
    def getInductanceMatrix(self, mode="node", sym_lev="lowest", as_equ=False):
        self._check_args(mode=mode,sym_lev=sym_lev)
        if as_equ:
            sym = self.getInductanceMatrix(mode=mode,sym_lev="highest")
            mat = self.getInductanceMatrix(mode=mode,sym_lev="lowest")
            return sy.Eq(sym,mat)
        
        if sym_lev == "lowest":
            node_list = self.parent_circuit.getNodeList()
            node_indices = [node_list.index(n) for n in self.getNodeList()]
            
            M = self.parent_circuit.getInductanceMatrix()[node_indices[0]:node_indices[-1]+1,node_indices[0]:node_indices[-1]+1]
            return M
            
            # FIXME
            #if mode == "node":
            #    return self.getBranchToNodeMatrix()*M*self.getNodeToBranchMatrix()
            #elif mode == "branch":
            #    return M
        elif sym_lev == "highest":
            if mode == "node":
                return sy.MatrixSymbol(sy.symbols("\\hat{\\mathbf{L}}_{n}"),len(list(self.graph.nodes))-1,len(list(self.graph.nodes))-1)
            elif mode == "branch":
                return sy.MatrixSymbol(sy.symbols("\\hat{\\mathbf{L}}_{b}"),len(list(self.graph.edges)),len(list(self.graph.edges)))
    
    def getInverseInductanceMatrix(self, mode="node", sym_lev="lowest", as_equ=False):
        self._check_args(mode=mode,sym_lev=sym_lev)
        if as_equ:
            sym = self.getInverseInductanceMatrix(mode=mode,sym_lev="highest")
            mat = self.getInverseInductanceMatrix(mode=mode,sym_lev="lowest")
            return sy.Eq(sym,mat)
        
        if sym_lev == "lowest":
            node_list = self.parent_circuit.getNodeList()
            node_indices = [node_list.index(n) for n in self.getNodeList()]
            
            return self.parent_circuit.getInverseInductanceMatrix()[node_indices[0]:node_indices[-1]+1,node_indices[0]:node_indices[-1]+1]
            
        elif sym_lev == "highest":
            return (self.getInductanceMatrix(mode=mode,sym_lev="highest"))**(-1)
    
    def getFluxVector(self, mode="node", sym_lev="lowest", as_equ=False):
        self._check_args(mode=mode,sym_lev=sym_lev)
        if as_equ:
            sym = self.getFluxVector(mode=mode,sym_lev="highest")
            mat = self.getFluxVector(mode=mode,sym_lev="lowest")
            return sy.Eq(sym,mat)
        
        if sym_lev == "lowest":
            if mode == "node":
                return sy.Matrix([self.parent_circuit.node_dofs[n][0] for n in list(self.graph.nodes)[1:]])
            elif mode == "branch":
                return sy.Matrix([self.parent_circuit.branch_dofs[e][0] for e in list(self.graph.edges)])
        elif sym_lev == "highest":
            if mode == "node":
                return sy.MatrixSymbol(sy.symbols("\\vec{\\mathbf{\\Phi}}_{n}"),len(list(self.graph.nodes))-1,1)
            elif mode == "branch":
                return sy.MatrixSymbol(sy.symbols("\\vec{\\mathbf{\\Phi}}_{b}"),len(list(self.graph.edges)),1)
    
    def getLeftDisplacementOpMatrix(self,adjoint=False,mode="node",sym_lev="lowest",as_equ=False,as_vec=False):
        self._check_args(mode=mode,sym_lev=sym_lev)
        if as_equ:
            sym = self.getLeftDisplacementOpMatrix(adjoint=adjoint,mode=mode,sym_lev="highest")
            mat = self.getLeftDisplacementOpMatrix(adjoint=adjoint,mode=mode,sym_lev="lowest")
            return sy.Eq(sym,mat)
        
        if sym_lev == "lowest":
            if mode == "node":
                vec = list(np.zeros(len(list(self.graph.edges))))
                edges = list(self.graph.edges)
                ind2 = 2 if not adjoint else 3
                ind1 = 3 if not adjoint else 2
                
                # Use left decomposed flux vector
                lv = self.getLeftDecompFluxVector()
                rv = self.getRightDecompFluxVector()
                for i in range(len(edges)):
                    l = lv[i]
                    r = rv[i]
                    if l == 0:
                        vec[i] = 1
                    else:
                        if r.args != ():
                            if l.args[0] == 1 and r.args[0] == -1:
                                vec[i] = self.parent_circuit.node_dofs[edges[i][0]][ind2]
                        else:
                            vec[i] = self.parent_circuit.node_dofs[edges[i][0]][ind2]
                if as_vec:
                    return sy.Matrix(vec)
                else:
                    return sy.diag(*vec)
            elif mode == "branch":
                
                pass
        elif sym_lev == "highest":
            if mode == "node":
                if not adjoint:
                    return sy.MatrixSymbol(sy.symbols("\\hat{\\mathbf{D}}_{nl}"),len(list(self.graph.edges)),len(list(self.graph.edges)))
                else:
                    return sy.MatrixSymbol(sy.symbols("\\hat{\\mathbf{D}}_{nl}^{\\dagger}"),len(list(self.graph.edges)),len(list(self.graph.edges)))
            elif mode == "branch":
                pass
    
    def getRightDisplacementOpVector(self,adjoint=False,mode="node",sym_lev="lowest",as_equ=False):
        self._check_args(mode=mode,sym_lev=sym_lev)
        if as_equ:
            sym = self.getRightDisplacementOpVector(adjoint=adjoint,mode=mode,sym_lev="highest")
            mat = self.getRightDisplacementOpVector(adjoint=adjoint,mode=mode,sym_lev="lowest")
            return sy.Eq(sym,mat)
        
        if sym_lev == "lowest":
            if mode == "node":
                vec = list(np.zeros(len(list(self.graph.edges))))
                edges = list(self.graph.edges)
                ind2 = 2 if not adjoint else 3
                ind1 = 3 if not adjoint else 2
                
                
                # Use right decomposed flux vector
                lv = self.getLeftDecompFluxVector()
                rv = self.getRightDecompFluxVector()
                for i in range(len(edges)):
                    l = lv[i]
                    r = rv[i]
                    if r == 0:
                        vec[i] = 1
                    else:
                        if l.args != ():
                            if l.args[0] == 1 and r.args[0] == -1:
                                vec[i] = self.parent_circuit.node_dofs[edges[i][1]][ind1]
                        else:
                            vec[i] = self.parent_circuit.node_dofs[edges[i][1]][ind2]
                return sy.Matrix(vec)
            elif mode == "branch":
                
                pass
        elif sym_lev == "highest":
            if mode == "node":
                if not adjoint:
                    return sy.MatrixSymbol(sy.symbols("\\vec{\\mathbf{D}}_{nr}"),len(list(self.graph.edges)),1)
                else:
                    return sy.MatrixSymbol(sy.symbols("\\vec{\\mathbf{D}}_{nr}^{\\dagger}"),len(list(self.graph.edges)),1)
            elif mode == "branch":
                pass
    
    

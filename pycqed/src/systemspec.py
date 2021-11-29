""" The :py:mod:`pycqed.src.systemspec` module defines the class :class:`SystemSpec` which is used to join circuits built using the :class:`CircuitSpec` class. At the top-level, the subcircuits are defined as black boxes with nodes represented as I/O pins. Subcircuits can be coupled directly, or with any circuit element that is available.
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

class SystemSpec:
    """
    """
    
    # Class variables for direction control of parallel components
    __dir_map = {'right':0,'up':1,'left':2,'down':3}
    __dir_arr = ['right','up','left','down']
    
    # Circuit types for system graph. Determines the colours used to identify subcircuits.
    __circuit_types = {'Qubit':"blue",'Coupler':"red",'Other':"black"}
    
    def __init__(self, system_name):
        self.sys_name = system_name
        self.drawing = schem.Drawing()
        self.system_graph = nx.DiGraph(circuit_name=self.sys_name) # Top level
        self.circuit_graph = None # Bottom level, use first added circuit graph instance.
        
        # Tracker variables
        self.curr_circuit = 0                   # Current circuit ID
        self.circuit_names = {}                 # Maps a circuit name to an ID
        self.circuit_ids = {}                   # Maps a circuit ID to a CircuitSpec instance
        self.circuit_idname = {}                # Maps a circuit ID to a name
        self.ncap = 0                           # Total number of capacitors
        self.nind = 0                           # Total number of inductors
        self.njjs = 0                           # Total number of JJs
        self.schem_code = []                    # Holds all elements necessary to regenerate the circuit as is
        self.schem_code_block = []              # Block of drawing code associated with edge (temporary)
        self.schem_code_blocks = {}             # Holds all blocks necessary to regenerate/edit circuit edges
        self.node_circ_coords = {}              # Holds the circuit diagram coordinates of the nodes
        self.code_elems = 0                     # Counts the number of SchemDraw elements
        self.code_lines = 0
    
    def showSystemStats(self):
        """
        """
        print ("Subcircuit Information:")
        print ("  Subcircuit Count: %i" % self.curr_circuit)
        print ("")
        print ("Total Component Count:")
        print ("  Capacitors: %i" % self.ncap)
        print ("  Inductors: %i" % self.nind)
        print ("  Josephson Junctions: %i" % self.njjs)
        #print ("")
        #print ("Graph Properties:")
        #print ("  Nodes: %i" % (len(list(self.graph.nodes))))
        #print ("  Edges: %i" % len(list(self.graph.edges)))
        #print ("  Spanning Branches: %i" % len(self.getSpanningBranches()))
        #print ("  Closure Branches: %i" % len(self.getClosureBranches()))
        #print ("")
        #print ("Circuit Properties:")
        #print ("  Irreducible Loops: %i" % len(self.getIrreducibleLoops()))
        #print ("  Superconducting Loops: %i" % len(self.getSuperconductingLoops()))
        #print ("  Flux Bias Lines: %i" % len([v for v in list(self.flux_bias.values()) if v != 0.0]))
        #print ("  Charge Bias Lines: %i" % len([v for v in list(self.charge_bias.values()) if v != 0.0]))
    
    ###################################################################################################################
    #       Circuit and System Graph Functions
    ###################################################################################################################
    
    #
    #
    #
    def drawSystemGraph(self,filename=None,output="svg",inline=False,**kwargs):
        """
        """
        
        fig,ax = plt.subplots(1,1,constrained_layout=True)
        nx.draw(self.system_graph, with_labels=True, font_weight='bold',pos=nx.circular_layout(self.system_graph),ax=ax,node_color="C0",node_size=1500,font_size=16,**kwargs)
        #edge_labels = nx.get_edge_attributes(self.system_graph,'etype')
        #nx.draw_networkx_edge_labels(self.system_graph, nx.circular_layout(self.system_graph), edge_labels=edge_labels,node_color="C0",node_size=1500,font_size=16,**kwargs)
        
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
    #       Circuit Drawing Functions
    ###################################################################################################################
    
    def addCircuit(self, circuit_spec, circuit_type="Other"):
        """
        """
        
        # Check if circuit name already exists
        
        # Save the circuit and details
        self.curr_circuit += 1
        self.circuit_names[circuit_spec.circ_name] = self.curr_circuit
        self.circuit_idname[self.curr_circuit] = circuit_spec.circ_name
        self.circuit_ids[self.curr_circuit] = circuit_spec
        
        # Update system stats
        self.ncap += circuit_spec.ncap
        self.nind += circuit_spec.nind
        self.njjs += circuit_spec.njjs
        
        # Get the nodes, to use as I/O pins
        node_list = self.circuit_ids[self.curr_circuit].getNodeList()
        
        # Create black box
        binputs = {'cnt':1,'labels':['0']} # Ground node
        rinputs = {'cnt':len(node_list),'labels':[str(i) for i in node_list]} # Other nodes
        box = e.blackbox(self.drawing.unit, self.drawing.unit, binputs=binputs, rinputs=rinputs, mainlabel=self.circuit_idname[self.curr_circuit])
        self.addSchemElem(box, color=self.__circuit_types[circuit_type])
        
        # Add system graph node
        self.system_graph.add_node(1,ntype=circuit_type)
        
    def addCapCoupling(self, comp_name, circ1, node1, circ2, node2, scaling=0.5, lead_len=1.0, comp_len=0.6, endline=None, arm_len=0.0):
        """
        """
        
        # Generate box and anchors list for each circ. Pins shown are only those specified. Anchors will need to be updated on successive calls
        
        pass
    
    def addIndCoupling(self, comp_name, circ1, branch1, circ2, branch2, scaling=0.5, lead_len=1.0, comp_len=0.6, endline=None, arm_len=0.0):
        """
        """
        pass
    
    def addSchemElem(self, elm_def, **kwargs):
        elem = self.drawing.add(elm_def,**kwargs)
        self.schem_code_block.append(["add",self.code_elems,elm_def,kwargs])
        self.code_elems += 1
        self.code_lines += 1
        return elem
    
    def pushSchemDraw(self):
        self.drawing.push()
        self.schem_code_block.append(["push"])
        self.code_lines += 1
    
    def popSchemDraw(self):
        self.drawing.pop()
        self.schem_code_block.append(["pop"])
        self.code_lines += 1
    
    def drawCircuit(self, filename=None, output="svg", inline=False):
        """
        """
        
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
    
    
    ###################################################################################################################
    #       Symbolic Math and Equation Functions
    ###################################################################################################################
    
    
    ###################################################################################################################
    #       Internal Functions
    ###################################################################################################################
    


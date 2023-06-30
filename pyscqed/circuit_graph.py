import networkx as nx
import graphviz as gv
import pydot as pd
import platform

# FIXME: Make this more intelligent
if platform.system() == 'Windows':
    import os
    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

class CircuitGraph:

    # Prefixes for the circuit elements
    _element_prefixes = [
        "C", # Capacitor
        "L", # Inductor
        "I", # Josephson junction
        "V", # Phase-slip nanowire
        "M"  # Mutual inductance
    ]

    _resonator_prefixes = [
        "f", # Resonant frequency
        "Z", # Resonator impedance
        "g"  # Coupling term
    ]

    # Different subgraphs
    _subgraphs = [
        "Circuit",
        "Conductive",
        "SCTree",
        "SCGraph",
        "Loop"
    ]

    def __init__(self, circuit_name=""):
        """ Abstract description of a superconducting circuit.
        """

        # Components associated with each branch
        self.components_map = {}

        # Charge bias associated with node
        self.charge_bias_nodes = {}

        # Flux bias associated with each loop
        self.flux_bias_edges = {}

        # All circuit loops map
        self.loops_map = {}

        # Edges shared between loops
        self.loop_adjacency_map = {}

        # Capacitively-coupled resonators
        self.resonators_cap = {}

        # Mutually coupled branches
        self.coupled_branches = {}

        # General undirected circuit graph
        self.circuit_graph = nx.MultiGraph(circuit_name=circuit_name)

    def addBranch(self, n1, n2, component):
        """ Adds a branch between two circuit nodes that contains a single component.
        """

        # Check there are no mutual inductors specified
        if component[0] == self._element_prefixes[4]:
            raise TypeError("Cannot add a mutual inductance to a circuit branch.")

        # Check the symbol is correct
        if component[0] not in self._element_prefixes:
            raise ValueError("Invalid component symbol '%s', it should be one of %s." % (component[0], repr(self._element_prefixes)))

        # Add the branch
        k = self.circuit_graph.add_edge(n1, n2, component=component, label=component)

        # Update
        self._update_components_map()
        self._update_graphs()
        self._update_couplings_map((n1, n2, k))

    def coupleBranchesInductively(self, inductor1, inductor2, component):
        """ Couples two branches inductively.
        """
        # Check the edges are inductive
        if component[0] != self._element_prefixes[4]:
            raise TypeError("Branch coupling component must be a mutual inductance.")
        edge1 = self.getComponentEdge(inductor1)
        edge2 = self.getComponentEdge(inductor2)
        if not self.isInductiveEdge(edge1):
            raise TypeError("The selected component %s is not an inductor." % inductor1)
        if not self.isInductiveEdge(edge2):
            raise TypeError("The selected component %s is not an inductor." % inductor2)
        alt_edge1 = (edge1[1], edge1[0], edge1[2])
        alt_edge2 = (edge2[1], edge2[0], edge2[2])
        edge1 = alt_edge1 if edge1 not in self.sc_spanning_tree_wc.edges else edge1
        edge2 = alt_edge2 if edge2 not in self.sc_spanning_tree_wc.edges else edge2
        self.coupled_branches[component] = (edge1, edge2)

    def coupleResonatorCapacitively(self, node, component):
        """ Couples a linear resonator capacitively.
        """
        if node not in self.circuit_graph.nodes:
            raise ValueError("Node %i not part of the circuit graph." % node)

        if node == 0:
            raise ValueError("Cannot couple a resonator to the ground node of the circuit")

        if self.resonators_cap[node] is not None:
            raise ValueError("Node %i already has a resonator coupled to it (only one per node supported currently)." % node)

        # Check there are no mutual inductors specified
        if component[0] != self._element_prefixes[0]:
            raise TypeError("The resonator coupling element must be a capacitor.")

        # Detect duplicates
        if component in self.components_map.values():
            raise ValueError("Component %s already exists. Change the name of the component." % component)

        # Create the resonator capacitor and inductor symbols
        Cr = "%s%ir" % (self._element_prefixes[0], node)
        Lr = "%s%ir" % (self._element_prefixes[1], node)

        # Create the resonator frequency and impedance symbols
        fr = "%s%ir" % (self._resonator_prefixes[0], node) # This is the bare frequency
        Zr = "%s%ir" % (self._resonator_prefixes[1], node)

        # Create the Hamiltonian coupling term symbols
        gC = "%s%ir" % (self._resonator_prefixes[2], node)

        # Create the loaded resonator and impedance symbols
        frd = "%s%irl" % (self._resonator_prefixes[0], node)
        Zrd = "%s%irl" % (self._resonator_prefixes[1], node)

        self.resonators_cap[node] = {
            "coupling": component,
            "fr": fr,
            "Zr": Zr,
            "Cr": Cr,
            "Lr": Lr,
            "gC": gC,
            "frl": frd,
            "Zrl": Zrd
        }

    def removeAllResonators(self):
        """ Removes all resonator descriptions from the circuit graph.
        """
        for node in self.circuit_graph.nodes:
            self.resonators_cap[node] = None

    def coupleResonatorInductively(self, edge, component, frequency, impedance=50.0):
        """ Couples a linear resonator inductively.
        """
        pass

    def addFluxBias(self, edge_component, suffix, mutual_inductance=None):
        """ Adds a flux bias term to the specified branch.
        """
        edge = self.getComponentEdge(edge_component)
        if not self.isEdgeInSuperconductingLoop(edge):
            raise TypeError("The component %s is not part of a superconducting loop." % edge_component)
        if type(suffix) is not str:
            raise TypeError("The 'suffix' parameter is not a string.")
        if len(suffix) > 8:
            raise ValueError("The 'suffix' parameter is too long.")
        alt_edge = (edge[1], edge[0], edge[2])
        if edge not in self.sc_spanning_tree_wc.edges and alt_edge not in self.sc_spanning_tree_wc.edges:
            raise TypeError("Edge %s is not in conductive circuit subgraph." % repr(edge))
        edge = alt_edge if edge not in self.sc_spanning_tree_wc.edges else edge

        if mutual_inductance is not None:
            # Ensure component is a mutual inductance
            if mutual_inductance[0] != self._element_prefixes[4]:
                raise TypeError("Flux bias coupling component must be a mutual inductance.")

            # We can't load a JJ branch
            if not self.isInductiveEdge(edge):
                raise TypeError("A mutual inductance must be coupled to an inductive branch.")

            # Detect duplicates
            if mutual_inductance in self.components_map.values():
                raise ValueError("Component %s already exists. Change the name of the component." % mutual_inductance)

            # Check the edge has an inductor
            if self.isInductiveEdge(edge) == False:
                raise TypeError("The selected edge %s is not inductive." % repr(edge))

        # Save it
        if edge in self.flux_bias_edges:
            raise ValueError("Component %s already has a flux bias term." % edge_component)
        if suffix in [v["suffix"] for v in self.flux_bias_edges.values()]:
            raise ValueError("The suffix %s is already in use." % suffix)
        self.flux_bias_edges[edge] = {
            "mutual_inductance": mutual_inductance,
            "suffix": suffix
        }

    def addChargeBias(self, node, suffix, coupling_capacitance=None):
        """ Adds a charge bias term to the specified node.
        """
        if node not in self.circuit_graph.nodes:
            raise ValueError("Node %i not part of the circuit graph." % node)
        if type(suffix) is not str:
            raise TypeError("The 'suffix' parameter is not a string.")
        if len(suffix) > 8:
            raise ValueError("The 'suffix' parameter is too long.")

        if coupling_capacitance is not None:
            # Ensure component is a capacitor
            if coupling_capacitance[0] != self._element_prefixes[0]:
                raise TypeError("Charge bias coupling component must be a capacitor.")

            # Detect duplicates
            if coupling_capacitance in self.components_map.values():
                raise ValueError("Component %s already exists. Change the name of the component." % coupling_capacitance)

        if node in self.charge_bias_nodes:
            raise ValueError("Node %i already has a charge bias term." % node)
        if suffix in [v["suffix"] for v in self.charge_bias_nodes.values()]:
            raise ValueError("The suffix %s is already in use." % suffix)
        self.charge_bias_nodes[node] = {
            "coupling_capacitance": coupling_capacitance,
            "suffix": suffix
        }

    def isEdgeInSuperconductingLoop(self, edge):
        """ Checks a branch is part of a superconducting loop.
        """
        if edge not in self.components_map:
            raise ValueError("The specified edge %s was not found." % repr(edge))

        for index, loop_edges in self.sc_loops.items():
            if edge in loop_edges:
                return True
        return False

    def isCapacitiveEdge(self, edge):
        """ Checks a branch contains a capacitor.
        """
        cstr = self.components_map[edge]
        if cstr[0] == self._element_prefixes[0]:
            return True
        return False

    def isInductiveEdge(self, edge):
        """ Checks a branch contains an inductor.
        """
        cstr = self.components_map[edge]
        if cstr[0] == self._element_prefixes[1]:
            return True
        return False

    def isJosephsonEdge(self, edge):
        """ Checks a branch contains a JJ.
        """
        cstr = self.components_map[edge]
        if cstr[0] == self._element_prefixes[2]:
            return True
        return False

    def isPhaseSlipEdge(self, edge):
        """ Checks a branch contains a phase-slip nano wire.
        """
        cstr = self.components_map[edge]
        if cstr[0] == self._element_prefixes[3]:
            return True
        return False

    def getCapacitiveEdges(self):
        """ Gets a mapping of all edges that contain a capacitor.
        """
        edges_map = {v: k for k, v in self.components_map.items()}
        ret = {}
        for c, edge in edges_map.items():
            if c[0] == self._element_prefixes[0]:
                ret[c] = edge
        return ret

    def getInductiveEdges(self):
        """ Gets a mapping of all edges that contain an inductor.
        """
        edges_map = {self.components_map[k]: k for k in self.sc_spanning_tree_wc.edges}
        ret = {}
        for c, edge in edges_map.items():
            if c[0] == self._element_prefixes[1]:
                ret[c] = edge
        return ret

    def getJosephsonEdges(self):
        """ Gets a mapping of all edges that contain a JJ.
        """
        edges_map = {self.components_map[k]: k for k in self.sc_spanning_tree_wc.edges}
        ret = {}
        for c, edge in edges_map.items():
            if c[0] == self._element_prefixes[2]:
                ret[c] = edge
        return ret

    def getPhaseSlipEdges(self):
        """ Gets a mapping of all edges that contain a phase-slip nanowire.
        """
        edges_map = {self.components_map[k]: k for k in self.sc_spanning_tree_wc.edges}
        ret = {}
        for c, edge in edges_map.items():
            if c[0] == self._element_prefixes[3]:
                ret[c] = edge
        return ret

    def getComponentEdge(self, component):
        """ Gets the edge associated with the specified component.
        """
        if component not in self.components_map.values():
            raise ValueError("Component %s does not exist." % component)

        # Capacitive edges are not directional
        if component[0] == self._element_prefixes[0]:
            edges_map = {v: k for k, v in self.components_map.items()}
            return edges_map[component]

        # Other edges are directional
        edges_map = {self.components_map[k]: k for k in self.sc_spanning_tree_wc.edges}
        return edges_map[component]

    def getEdgesSharedWithLoop(self, loop_key):
        if loop_key not in self.sc_loops.keys():
            raise ValueError("No loop key %i available." % loop_key)

        # Get the loops connected to this loop and save their edges
        loop_edges = set(self.sc_loops[loop_key])
        edges = set(self.sc_loops[loop_key])
        for key, loop in self.sc_loops.items():
            if key == loop_key:
                continue

            if not loop_edges.isdisjoint(set(loop)):
                edges |= set(loop)
        return edges

    #
    # DRAWING
    #

    def drawGraphViz(self, graph='Circuit', filename=None, format='svg'):
        if graph not in self._subgraphs:
            raise TypeError("Invalid subgraph type '%s'." % graph)

        if graph == "Circuit":
            G = self.circuit_graph
        elif graph == "Conductive":
            G = self.circuit_conductive_graph
        elif graph == "SCTree":
            G = self.sc_spanning_tree
        elif graph == "SCGraph":
            G = self.sc_spanning_tree_wc
        elif graph == "Loop":
            G = self.loop_graph

        # Get the pydot graph
        pd_graph = nx.nx_pydot.to_pydot(G)

        # Compile the graphviz source
        src = pd_graph.create(format='dot').decode('utf8')
        if filename is not None:
            with open(filename, "w") as fd:
                fd.write(src)
        return gv.Source(src)

    #
    # INTERNAL
    #
    def _update_graphs(self):
        self._get_conductive_graph()
        self._get_virtual_grounds()
        self._get_spanning_tree()
        self._get_sc_circuit()
        self._get_closure_branches()
        self._get_sc_loops()
        self._get_loop_graph()

    def _update_components_map(self):
        # Get all components keyed by edge
        self.components_map = nx.get_edge_attributes(self.circuit_graph, "component")

        # Detect duplicates
        tmp = list(set(self.components_map.values()))
        if len(tmp) != len(self.components_map.values()):
            raise ValueError("Duplicate component detected. Change the name of the component.")

        # Reverse the keys for convenience, use more memory rather than complicating later code
        tmp = {}
        for k, v in self.components_map.items():
            tmp[(k[1], k[0], k[2])] = v
        self.components_map.update(tmp)

    def _update_couplings_map(self, edge):
        n1, n2, k = edge
        if n1 not in self.resonators_cap.keys():
            self.resonators_cap[n1] = None
        if n2 not in self.resonators_cap.keys():
            self.resonators_cap[n2] = None

    def _get_conductive_graph(self):
        labels = nx.get_edge_attributes(self.circuit_graph, "label")
        self.circuit_conductive_graph = nx.MultiGraph()

        # Add all nodes
        self.circuit_conductive_graph.add_nodes_from(self.circuit_graph.nodes)

        # Ignore edges containing only capacitors
        for edge, component in nx.get_edge_attributes(self.circuit_graph, "component").items():
            if component[0] == self._element_prefixes[0]:
                continue
            self.circuit_conductive_graph.add_edge(edge[0], edge[1], key=edge[2],
                                                   component=component, label=labels[edge])

    def _get_virtual_grounds(self):
        labels = nx.get_edge_attributes(self.circuit_conductive_graph, "label")
        # Get connected graphs
        connected = [self.circuit_conductive_graph.subgraph(c).copy()\
                    for c in nx.connected_components(self.circuit_conductive_graph)]

        # Get virtual grounds
        self.virtual_grounds = {}
        for subG in connected:

            # If only one node, it is a virtual ground
            if len(subG.nodes) == 1:
                n = list(subG.nodes)[0]
                S = nx.MultiDiGraph()
                S.add_node(n)
                self.virtual_grounds[n] = ([], S)

            # Get spanning tree of subgraph
            G = nx.minimum_spanning_tree(subG)

            # Construct the spanning tree part of the sub
            S = nx.MultiDiGraph()
            for edge in G.edges:
                S.add_edge(edge[0], edge[1], key=edge[2], edge_type="S", label=labels[edge])

            # Get closure branches, i.e. those that do not appear in the spanning tree
            closure_edges = list(set(subG.edges) - set(S.edges))

            # Format closure branches
            for i, edge in enumerate(closure_edges):
                closure_edges[i] = (edge[1], edge[0], edge[2]) # reverse edge to preserve flow order

            # Find nodes with 0 in_degree
            for n, d in S.in_degree:
                if d == 0:
                    self.virtual_grounds[n] = (closure_edges, S)
                    break

    def _get_spanning_tree(self):
        S = nx.union_all([S for c, S in self.virtual_grounds.values()])
        labels = nx.get_edge_attributes(self.circuit_conductive_graph, "label")

        # As the node order may have been changed, reconstruct the graph
        self.sc_spanning_tree = nx.MultiDiGraph()
        self.sc_spanning_tree.add_nodes_from(self.circuit_conductive_graph.nodes)
        for edge in S.edges:
            self.sc_spanning_tree.add_edge(edge[0], edge[1], key=edge[2], edge_type="S", label=labels[edge])

    def _get_sc_circuit(self):
        labels = nx.get_edge_attributes(self.circuit_conductive_graph, "label")
        self.sc_spanning_tree_wc = self.sc_spanning_tree.copy()
        for c, S in self.virtual_grounds.values():
            for edge in c:
                edger = (edge[1], edge[0], edge[2])
                label = labels[edger] if edge not in labels.keys() else labels[edge]
                self.sc_spanning_tree_wc.add_edge(edge[0], edge[1], key=edge[2], edge_type="C", label=label)

    def _get_sc_loops(self):
        c = 0
        self.sc_loops = {}
        self.loop_closures = {}
        for source_node, spanning in self.virtual_grounds.items():
            closure_edges, S = spanning

            for edge in closure_edges:
                self.sc_loops[c] = []
                try:
                    p1 = sorted(nx.all_simple_edge_paths(S, source_node, edge[0]))[0]
                    p2 = sorted(nx.all_simple_edge_paths(S, source_node, edge[1]))[0]
                    self.sc_loops[c].extend(list(set(p1)^set(p2)))
                except IndexError:
                    self.sc_loops[c].extend(list(p1))

                self.sc_loops[c].append(edge)
                self.loop_closures[c] = edge
                c+=1

    def _get_closure_branches(self):
        self.closure_branches = []
        for closure_edges, S in self.virtual_grounds.values():
            self.closure_branches.extend(closure_edges)

    def _get_loop_graph(self):
        loop_graph_nodes = {}
        loop_graph_edges = {}
        multi_edges = set()
        counter = 0

        # Find the 2-node loops
        counter_dict = {}
        loop_c = 0
        for edge in self.circuit_graph.edges:
            key = (edge[0], edge[1])
            if key in counter_dict.keys():
                counter_dict[key].append(edge)
            else:
                counter_dict[key] = [edge]
        duplicates_dict = {k: v for k, v in counter_dict.items() if len(v) > 1}
        loop_graph_nodes = {}
        counter = 0

        for k, v in duplicates_dict.items():
            for i in range(len(v)-1):
                loop_graph_nodes[counter] = [v[i], v[i+1]]
                counter += 1

            # Keep a set of multi-edges
            multi_edges.update(v)

        # Reduce multigraph to simple graph by removing edge duplicates
        G = nx.Graph()
        G.add_edges_from(list(set([(e[0],e[1]) for e in self.circuit_graph.edges])))

        # Update the loops data with longer-node loops
        multi_edges_used = set()
        #for cycle in nx.minimum_cycle_basis(G):
        for cycle in nx.cycle_basis(G): # FIXME: This preserves node order but doesn't use minimum weight cycles.

            # Last edge nodes are reversed to preserve ordering of multigraph
            edges = [(cycle[i], cycle[i+1], 0) if i < len(cycle)-1 else (cycle[(i+1)%len(cycle)], cycle[i], 0) for i in range(len(cycle))]

            # Sort the edge nodes
            edges = [(e[0], e[1], e[2]) if e[0] < e[1] else (e[1], e[0], e[2]) for e in edges]

            # Check if a multi-edge has already appeared
            for i, edge in enumerate(edges):
                if edge in multi_edges_used:
                    # Increment the multi-edge index
                    edges[i] = (edge[0], edge[1], edge[2]+1)
                else:
                    # Add the edge to the list if it's a multi-edge
                    if edge in multi_edges:
                        multi_edges_used.add(edge)
                        multi_edges.remove(edge)

            loop_graph_nodes[counter] = edges
            counter += 1

        # Now find the edges in common between loops
        # Exploits the ordering of nodes in multigraph edges
        for i in range(counter-1):
            for j in range(i+1, counter, 1):
                l1 = loop_graph_nodes[i]
                l2 = loop_graph_nodes[j]

                edges = []
                for k in l1:
                    if k in l2:
                        edges.append(k)
                if len(edges) > 0:
                    loop_graph_edges[(i, j)] = edges

        # Save the loop data
        self.loops_map = loop_graph_nodes
        self.loop_adjacency_map = loop_graph_edges

        # Create the loop graph
        self.loop_graph = nx.Graph()
        for k, v in loop_graph_nodes.items():
            self.loop_graph.add_node(k, circuit_edges=v)

        for k, v in loop_graph_edges.items():
            self.loop_graph.add_edge(k[0], k[1], circuit_edges=v)

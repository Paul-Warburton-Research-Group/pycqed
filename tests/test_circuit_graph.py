"""Test the circuit graph API."""
import unittest
import itertools
import os

import numpy as np

from pyscqed.circuit_graph import CircuitGraph


def do_all_branch_type_tests_on_branch(testobj, G, edge, expected_type):
    fn = [
        G.isCapacitiveEdge,
        G.isInductiveEdge,
        G.isJosephsonEdge,
        G.isPhaseSlipEdge
    ]
    for i in range(4):
        if i == G._element_prefixes.index(expected_type):
            testobj.assertTrue(fn[i](edge))
        else:
            testobj.assertFalse(fn[i](edge))


def all_loops_are_unique(testobj, G):
    """There should be at least one distinct branch per loop."""
    available = {}
    for i, loop in G.sc_loops.items():
        curr = set(loop)
        for j, other_loop in G.sc_loops.items():
            if i == j:
                continue
            curr -= set(other_loop)
        testobj.assertTrue(len(curr) > 0)


def plot_all_graphs(G, prefix):
    for t in G._subgraphs:
        G.drawGraphViz(graph=t, filename=("%s_%s" % (prefix, t)))


class CircuitGraphTest(unittest.TestCase):
    """Test the circuit_graph.CircuitGraph class."""

    def setUp(self):
        """Run before tests"""

    def tearDown(self):
        """Run after tests"""

    def test_graph_branch_rules(self):
        # Test that branch types can be correctly identified
        for i in range(4):
            graph = CircuitGraph()
            graph.addBranch(0, 1, graph._element_prefixes[i])
            do_all_branch_type_tests_on_branch(self, graph, (0, 1, 0), graph._element_prefixes[i])

        # Cannot add invalid components
        self.assertRaises(ValueError, graph.addBranch, 0, 1, "J")

        # Test that the correct number of superconducting loops are detected
        for perm in itertools.permutations(["I", "L"], 2):
            graph = CircuitGraph()
            for item in perm:
                graph.addBranch(0, 1, item)
            self.assertTrue(len(graph.closure_branches) == 1)
            self.assertTrue(len(graph.sc_loops) == 1)
            all_loops_are_unique(self, graph)

        for perm in itertools.permutations(["I1", "I2", "L"], 3):
            graph = CircuitGraph()
            for item in perm:
                graph.addBranch(0, 1, item)
            self.assertTrue(len(graph.closure_branches) == 2)
            self.assertTrue(len(graph.sc_loops) == 2)
            all_loops_are_unique(self, graph)

        for perm in itertools.permutations(["I1", "I2", "I3", "L"], 4):
            graph = CircuitGraph()
            for item in perm:
                graph.addBranch(0, 1, item)
            self.assertTrue(len(graph.closure_branches) == 3)
            self.assertTrue(len(graph.sc_loops) == 3)
            all_loops_are_unique(self, graph)

        # Floating single loop
        graph = CircuitGraph()
        graph.addBranch(0, 1, "I1")
        graph.addBranch(1, 2, "L1")
        graph.addBranch(1, 2, "L2")
        self.assertTrue(len(graph.closure_branches) == 1)
        self.assertTrue(len(graph.sc_loops) == 1)
        all_loops_are_unique(self, graph)

        # No conducting loops
        graph = CircuitGraph()
        graph.addBranch(0, 1, "I")
        graph.addBranch(1, 2, "C")
        graph.addBranch(1, 2, "L")
        self.assertTrue(len(graph.closure_branches) == 0)
        self.assertTrue(len(graph.sc_loops) == 0)
        all_loops_are_unique(self, graph)

        graph = CircuitGraph()
        graph.addBranch(0, 1, "I")
        graph.addBranch(1, 2, "C")
        graph.addBranch(0, 2, "L")
        self.assertTrue(len(graph.closure_branches) == 0)
        self.assertTrue(len(graph.sc_loops) == 0)
        all_loops_are_unique(self, graph)

        # Large loop
        graph = CircuitGraph()
        N = 6
        for i in range(N):
            graph.addBranch(
                i, (i + 1) % N,
                "%s%i" % (graph._element_prefixes[np.random.randint(2) + 1], i)
            )
        self.assertTrue(len(graph.closure_branches) == 1)
        self.assertTrue(len(graph.sc_loops) == 1)
        all_loops_are_unique(self, graph)

        # Loops in series forming a larger loop
        graph = CircuitGraph()
        graph.addBranch(0, 1, "I1")
        graph.addBranch(0, 1, "C1")
        graph.addBranch(0, 1, "L1") # 1 loop on (0, 1, k)
        graph.addBranch(1, 2, "L2")
        graph.addBranch(1, 2, "I2")
        graph.addBranch(1, 2, "C2") # 1 loop on (1, 2, k)
        graph.addBranch(0, 2, "I3")
        graph.addBranch(0, 2, "I4")
        graph.addBranch(0, 2, "L3") # 2 loops on (0, 2, k)
        self.assertTrue(len(graph.closure_branches) == 5)
        self.assertTrue(len(graph.sc_loops) == 5)
        all_loops_are_unique(self, graph)

        # Larger loops sharing a branch (3 in this case)
        graph = CircuitGraph()
        graph.addBranch(0, 1, "I1")
        graph.addBranch(1, 2, "L1a")
        graph.addBranch(2, 0, "L1b")
        graph.addBranch(1, 3, "L2a")
        graph.addBranch(3, 0, "L2b")
        graph.addBranch(1, 4, "L3a")
        graph.addBranch(4, 0, "L3b")
        self.assertTrue(len(graph.closure_branches) == 3)
        self.assertTrue(len(graph.sc_loops) == 3)
        all_loops_are_unique(self, graph)

        # Larger loops sharing a loop (3 in this case)
        graph = CircuitGraph()
        graph.addBranch(0, 1, "I1")
        graph.addBranch(0, 1, "I2")
        graph.addBranch(1, 2, "L1a")
        graph.addBranch(2, 0, "L1b")
        graph.addBranch(1, 3, "L2a")
        graph.addBranch(3, 0, "L2b")
        graph.addBranch(1, 4, "L3a")
        graph.addBranch(4, 0, "L3b")
        self.assertTrue(len(graph.closure_branches) == 4)
        self.assertTrue(len(graph.sc_loops) == 4)
        all_loops_are_unique(self, graph)

    def test_component_listers(self):
        graph = CircuitGraph()
        graph.addBranch(0, 1, "I1")
        graph.addBranch(0, 1, "C1")
        graph.addBranch(0, 1, "I2")
        graph.addBranch(0, 1, "C2")
        graph.addBranch(1, 2, "L1a")
        graph.addBranch(2, 0, "L1b")
        graph.addBranch(1, 3, "L2a")
        graph.addBranch(3, 0, "L2b")
        graph.addBranch(1, 4, "L3a")
        graph.addBranch(4, 0, "L3b")

        inds = {"L1a", "L1b", "L2a", "L2b", "L3a", "L3b"}
        jjs = {"I1", "I2"}
        caps = {"C1", "C2"}

        self.assertTrue(set(graph.getInductiveEdges()) == inds)
        self.assertTrue(set(graph.getJosephsonEdges()) == jjs)
        self.assertTrue(set(graph.getCapacitiveEdges()) == caps)

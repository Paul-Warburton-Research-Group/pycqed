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

        # Test that in loops with JJs and Ls, the JJs are always the closure branches for all gauge choices
        for perm in itertools.permutations(["I", "L"], 2):
            graph = CircuitGraph()
            for item in perm:
                graph.addBranch(0, 1, item)
            self.assertTrue(len(graph.closure_branches) == 1)
            self.assertTrue(graph.isJosephsonEdge(graph.closure_branches[0]))

        for perm in itertools.permutations(["I1", "I2", "L"], 3):
            graph = CircuitGraph()
            for item in perm:
                graph.addBranch(0, 1, item)
            self.assertTrue(len(graph.closure_branches) == 2)
            self.assertTrue(all(graph.isJosephsonEdge(branch) for branch in graph.closure_branches))

        for perm in itertools.permutations(["I1", "I2", "I3", "L"], 4):
            graph = CircuitGraph()
            for item in perm:
                graph.addBranch(0, 1, item)
            self.assertTrue(len(graph.closure_branches) == 3)
            self.assertTrue(all(graph.isJosephsonEdge(branch) for branch in graph.closure_branches))

        # Test that a loop with inductors allow the closure branches to be inductive
        graph = CircuitGraph()
        graph.addBranch(0, 1, "I1")
        graph.addBranch(1, 2, "L1")
        graph.addBranch(1, 2, "L2")
        self.assertTrue(len(graph.closure_branches) == 1)
        self.assertTrue(graph.isInductiveEdge(graph.closure_branches[0]))

        # Test that a loop that is not superconducting has no closure branches
        graph = CircuitGraph()
        graph.addBranch(0, 1, "I")
        graph.addBranch(1, 2, "C")
        graph.addBranch(1, 2, "L")
        self.assertTrue(len(graph.closure_branches) == 0)

        graph = CircuitGraph()
        graph.addBranch(0, 1, "I")
        graph.addBranch(1, 2, "C")
        graph.addBranch(0, 2, "L")
        self.assertTrue(len(graph.closure_branches) == 0)

        # Test that a large superconducting loop has a single closure branch
        graph = CircuitGraph()
        N = 6
        for i in range(N):
            graph.addBranch(
                i, (i + 1) % N,
                "%s%i" % (graph._element_prefixes[np.random.randint(2) + 1], i)
            )
        self.assertTrue(len(graph.closure_branches) == 1)

"""Test the circuit graph API."""
import unittest
import os

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
        graph = CircuitGraph()
        graph.addBranch(0, 1, "L")
        graph.addBranch(0, 1, "I")
        self.assertTrue(graph.isJosephsonEdge(graph.closure_branches[0]))

        graph = CircuitGraph()
        graph.addBranch(0, 1, "I")
        graph.addBranch(0, 1, "L")
        self.assertTrue(graph.isJosephsonEdge(graph.closure_branches[0]))

        graph = CircuitGraph()
        graph.addBranch(0, 1, "L")
        graph.addBranch(0, 1, "I1")
        graph.addBranch(0, 1, "I2")
        self.assertTrue(len(graph.closure_branches) == 2)
        for branch in graph.closure_branches:
            self.assertTrue(graph.isJosephsonEdge(branch))

        graph = CircuitGraph()
        graph.addBranch(0, 1, "I1")
        graph.addBranch(0, 1, "L")
        graph.addBranch(0, 1, "I2")
        self.assertTrue(len(graph.closure_branches) == 2)
        for branch in graph.closure_branches:
            self.assertTrue(graph.isJosephsonEdge(branch))

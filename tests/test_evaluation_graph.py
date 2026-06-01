import pytest

import numpy as np

from pyscqed.evaluation_graph import EvaluationGraph


def node_test_fn1(input: int):
    return input

def node_test_fn2(arg: int) -> bool:
    return arg == 1

def node_test_fn3(arg: int) -> float:
    return 0.0 if arg == 1 else 10.0

def node_test_fn4(switch: bool) -> tuple[float, float]:
    if switch:
        return 0.0, 1.0
    return 1.0, 0.0


def test_evaluation_graph_series_nodes():
    graph = EvaluationGraph()
    graph.addNode("A", fn=node_test_fn1, outputs=["arg"])
    graph.addNode("B", fn=node_test_fn2, outputs=["switch"])
    graph.addNode("C", fn=node_test_fn4, outputs=["final1", "final2"])
    graph.addDependency("A", "B", preserve_source_outputs=True)
    graph.addDependency("B", "C", preserve_source_outputs=True)
    inputs = {
        "A": {
            "input": 1
        }
    }
    result = graph.evaluate(inputs=inputs)
    assert result.data["A"]["arg"] == 1
    assert result.data["B"]["switch"] == True
    assert result.data["C"]["final1"] == 0.0
    assert result.data["C"]["final2"] == 1.0

    inputs = {
        "A": {
            "input": 0
        }
    }
    result = graph.evaluate(inputs=inputs)
    assert result.data["A"]["arg"] == 0
    assert result.data["B"]["switch"] == False
    assert result.data["C"]["final1"] == 1.0
    assert result.data["C"]["final2"] == 0.0


def test_evaluation_graph_fanout():
    graph = EvaluationGraph()
    graph.addNode("A", fn=node_test_fn1, outputs=["arg"])
    graph.addNode("B", fn=node_test_fn2, outputs=["final1"])
    graph.addNode("C", fn=node_test_fn3, outputs=["final2"])
    graph.addDependency("A", "B", preserve_source_outputs=True)
    graph.addDependency("A", "C", preserve_source_outputs=True)
    inputs = {
        "A": {
            "input": 1
        }
    }
    result = graph.evaluate(inputs=inputs)
    assert result.data["A"]["arg"] == 1
    assert result.data["B"]["final1"] == True
    assert result.data["C"]["final2"] == 0.0


def test_evaluation_dag():
    graph = EvaluationGraph()
    graph.addNode("A", fn=node_test_fn1, outputs=["arg"])
    graph.addNode("B", fn=node_test_fn2, outputs=["final1"])
    graph.addDependency("A", "B")
    graph.addDependency("B", "A")
    inputs = {
        "A": {
            "input": 1
        }
    }
    with pytest.raises(TypeError, match="The evaluation graph structure is not a DAG."):
        graph.evaluate(inputs=inputs)


def test_evaluation_output_preservation():
    graph = EvaluationGraph()
    graph.addNode("B", fn=node_test_fn2, outputs=["switch"])
    graph.addNode("C", fn=node_test_fn4, outputs=["final1", "final2"])
    graph.addDependency("B", "C")
    inputs = {
        "B": {
            "arg": 1
        }
    }

    result = graph.evaluate(inputs=inputs)
    assert "B" not in result.data
    assert result.data["C"]["final1"] == 0.0
    assert result.data["C"]["final2"] == 1.0


def test_evaluation_static_inputs():
    graph = EvaluationGraph()
    graph.addNode("A", fn=node_test_fn1, outputs=["arg"], static_inputs={"input":1})
    graph.addNode("B", fn=node_test_fn2, outputs=["final1"])
    graph.addDependency("A", "B", preserve_source_outputs=True)
    inputs = graph.getDefaultInputs()
    result = graph.evaluate(inputs=inputs)
    assert result.data["A"]["arg"] == 1
    assert result.data["B"]["final1"] == True

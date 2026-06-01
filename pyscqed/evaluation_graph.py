"""Defines the DAG structure for evaluations."""
from typing import Callable, Any, TypeAlias
import networkx as nx

from .result import EvaluationResult, NumericalResult, EvalKeys


NodeIOData: TypeAlias = dict[str, dict[str, Any]]


def _define_outputs(fn: Callable, keys: list[str]) -> Callable:
    """Decorator that maps function outputs to user defined keys."""
    def inner(*args, **kwargs) -> dict[str, Any]:
        outputs = fn(*args, **kwargs)
        if not isinstance(outputs, tuple):
            if len(keys) != 1:
                raise TypeError("A singular function output must map to a single key.")
            return {keys[0]: outputs}

        formatted = {}
        for k, v in zip(keys, outputs):
            if isinstance(v, NumericalResult):
                v.source = fn.__name__
            formatted[k] = v
        return formatted
    return inner


class EvaluationGraph:
    def __init__(self):
        self._graph = nx.DiGraph()
        self._silent_nodes: set[str] = set()
        self._output_keys: dict[str, list[str]] = {}
        self._static_inputs: NodeIOData = {}

    def addNode(self, name: str, fn: Callable, outputs: list[str], static_inputs: dict[str, Any] | None = None):
        self._graph.add_node(name, node_fn=fn, node_outputs=outputs)
        for output in outputs:
            if name not in self._output_keys:
                self._output_keys[name] = [output]
            else:
                self._output_keys[name].append(output)

        self._static_inputs[name] = {}
        if static_inputs is not None:
            self._static_inputs[name] = static_inputs

    def addDependency(self, source_node: str, target_node: str, preserve_source_outputs: bool = False):
        if source_node not in self._graph:
            raise ValueError(f"{source_node} source node does not exist.")
        if target_node not in self._graph:
            raise ValueError(f"{target_node} target node does not exist.")
        self._graph.add_edge(source_node, target_node)
        if not preserve_source_outputs:
            self._silent_nodes.add(source_node)
            del self._output_keys[source_node]

    def evaluate(self, inputs: NodeIOData) -> EvaluationResult:
        # Check the graph is a dag
        if not nx.is_directed_acyclic_graph(self._graph):
            raise TypeError("The evaluation graph structure is not a DAG.")

        # Get the start nodes
        next_nodes = self._get_start_nodes()
        next_inputs = inputs
        data = {}
        while len(next_nodes) > 0:
            data.update(self._get_node_outputs(next_nodes, next_inputs))
            next_nodes, next_inputs = self._get_next_nodes_and_inputs(next_nodes, data)

        # TODO: The silent node data should ideally be dropped in the loop
        for node in self._silent_nodes:
            del data[node]
        return EvaluationResult(data=data, source=self._get_name())

    def getDefaultInputs(self) -> NodeIOData:
        return {node: {} for node in self._get_start_nodes()}

    def getEvaluationKeys(self) -> EvalKeys:
        keys = []
        for name, outputs in self._output_keys.items():
            for output in outputs:
                keys.append((name, output))
        return keys

    @classmethod
    def _get_name(cls) -> str:
        return cls.__name__

    def _get_start_nodes(self) -> list[str]:
        return [node for node in self._graph.nodes if self._graph.in_degree(node) == 0]

    def _get_node_outputs(self, nodes: list[str], inputs: NodeIOData) -> NodeIOData:
        data = {}
        for node in nodes:
            original_callable = self._graph.nodes[node]["node_fn"]
            user_defined_outputs = self._graph.nodes[node]["node_outputs"]
            wrapped_callable = _define_outputs(
                original_callable,
                user_defined_outputs
            )
            data[node] = wrapped_callable(**inputs[node], **self._static_inputs[node])
        return data

    def _get_next_nodes_and_inputs(self, start_nodes: list[str], outputs: NodeIOData) -> tuple[list[str], NodeIOData]:
        next_nodes = []
        next_inputs = {}
        for node in start_nodes:
            node_iter = self._graph.successors(node)
            for inode in node_iter:
                next_nodes.append(inode)
            
                # The outputs of the previous function become the inputs of the next
                next_inputs[inode] = outputs[node]
        return next_nodes, next_inputs

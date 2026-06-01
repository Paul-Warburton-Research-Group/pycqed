import pytest

import numpy as np

from pyscqed.circuit_graph import CircuitGraph
from pyscqed.symbolic_system import SymbolicSystem
from pyscqed.numerical_system import NumericalSystem
from pyscqed.evaluation_graph import EvaluationGraph


def get_single_node_graph() -> CircuitGraph:
    graph = CircuitGraph()
    graph.addBranch(0, 1, "I")
    graph.addBranch(0, 1, "C")
    graph.addBranch(0, 1, "L")
    return graph


def get_numerical_system(graph: CircuitGraph) -> NumericalSystem:
    # NOTE: The system simulated here is not configured for generating representative physical results,
    # they are only there to check numerical consistency
    symbolic = SymbolicSystem(graph)
    hamil = NumericalSystem(symbolic)
    hamil.configureOperator(1, 40, "charge")
    return hamil


def test_sweep_one_dimension():
    hamil = get_numerical_system(get_single_node_graph())
    hamil.setParameterValues(
        "C", 20.0, # In fF
        "I", 40e-3, # In uA
        "L", 50.0  # In pH
    )

    expected_spectrum_sweep = [
        [59.95010688, 45.30354992, 36.57241861],
        [219.58210001, 175.64339355, 149.45048272],
        [379.20831265, 305.97937504, 262.32564643],
        [538.82879099, 436.31151958, 375.19792611],
        [698.44358094, 566.63985224, 488.06733808]
    ]

    trace = np.linspace(20.0, 40.0, 3)
    sweep = hamil.newSweepConfig()
    sweep.add("C", trace)

    result_disk = hamil.runSweep(sweep)
    spectrum = result_disk.getNumericalOutput("C")
    assert np.allclose(spectrum.T, expected_spectrum_sweep, rtol=0, atol=1e-6)
    traces = result_disk.getInputPoints("C")
    assert np.array_equal(traces, trace)

    result_mem = hamil.runSweep(sweep, use_disk=False)
    spectrum = result_mem.getNumericalOutput("C")
    assert np.allclose(spectrum.T, expected_spectrum_sweep, rtol=0, atol=1e-6)
    traces = result_disk.getInputPoints("C")
    assert np.array_equal(traces, trace)


def test_sweep_two_dimensions():
    hamil = get_numerical_system(get_single_node_graph())
    hamil.setParameterValues(
        "C", 20.0, # In fF
        "I", 40e-3, # In uA
        "L", 50.0  # In pH 
    )

    expected_spectrum_sweep1 = [
        [59.95010688, 45.30354992, 36.57241861],
        [219.58210001, 175.64339355, 149.45048272],
        [379.20831265, 305.97937504, 262.32564643],
        [538.82879099, 436.31151958, 375.19792611],
        [698.44358094, 566.63985224, 488.06733808]
    ]
    expected_spectrum_sweep2 = [
        [53.03927077, 39.66091887, 31.68577904],
        [198.84901805, 158.71511714, 134.79027614],
        [344.65184476, 277.7646906, 237.89129967],
        [490.44781127, 396.80967219, 340.98887104],
        [636.23697759, 515.85009468, 444.0830116 ]
    ]

    sweep = hamil.newSweepConfig()
    trace1 = np.linspace(20.0, 40.0, 3)
    trace2 = np.linspace(50.0, 60.0, 2)
    sweep.add("C", trace1)
    sweep.add("L", trace2)
    result = hamil.runSweep(sweep)

    with pytest.raises(
        ValueError,
        match="Insufficient independent and static variables to retrieve sweep result data."
    ):
        result.getOutputs("C")

    with pytest.raises(
        ValueError,
        match="Independent variable \"I\" not present in sweep."
    ):
        result.getOutputs("I", {"L": 50.0})

    spectrum1 = result.getNumericalOutput("C", {"L": 50.0})
    spectrum2 = result.getNumericalOutput("C", {"L": 60.0})
    assert np.allclose(spectrum1.T, expected_spectrum_sweep1, rtol=0, atol=1e-6)
    assert np.allclose(spectrum2.T, expected_spectrum_sweep2, rtol=0, atol=1e-6)
    traces1 = result.getInputPoints("C")
    assert np.array_equal(traces1, trace1)

    # Test multi-dimensional sweep retrieval
    CL = result.getNumericalOutput(["C", "L"])
    assert np.allclose(CL[:, 0].T, expected_spectrum_sweep1, rtol=0, atol=1e-6)
    assert np.allclose(CL[:, 1].T, expected_spectrum_sweep2, rtol=0, atol=1e-6)
    traces = result.getInputMesh(["C", "L"])
    assert np.array_equal(traces["C"][:, 0], trace1)
    assert np.array_equal(traces["L"][0, :], trace2)

    # Test that the independent variable input order correctly formats the output
    CL = result.getNumericalOutput(["L", "C"])
    assert np.allclose(CL[0].T, expected_spectrum_sweep1, rtol=0, atol=1e-6)
    assert np.allclose(CL[1].T, expected_spectrum_sweep2, rtol=0, atol=1e-6)
    traces = result.getInputMesh(["L", "C"])
    assert np.array_equal(traces["C"][0, :], trace1)
    assert np.array_equal(traces["L"][:, 0], trace2)

    result_mem = hamil.runSweep(sweep, use_disk=False)
    CL = result_mem.getNumericalOutput(["L", "C"])
    assert np.allclose(CL[0].T, expected_spectrum_sweep1, rtol=0, atol=1e-6)
    assert np.allclose(CL[1].T, expected_spectrum_sweep2, rtol=0, atol=1e-6)
    traces = result.getInputMesh(["L", "C"])
    assert np.array_equal(traces["C"][0, :], trace1)
    assert np.array_equal(traces["L"][:, 0], trace2)


def test_sweep_arb_eval_graph():
    hamil = get_numerical_system(get_single_node_graph())
    hamil.setParameterValues(
        "C", 20.0, # In fF
        "I", 40e-3, # In uA
        "L", 50.0  # In pH
    )

    input_array = np.arange(6).reshape((2, 3))

    def generator() -> np.ndarray:
        return input_array

    def manipulator(array: np.ndarray) -> np.ndarray:
        return array.T

    def final(array: np.ndarray) -> int:
        return array.size

    eval_graph = EvaluationGraph()
    eval_graph.addNode("Generator", fn=generator, outputs=["array"])
    eval_graph.addNode("Manipulator", fn=manipulator, outputs=["array"])
    eval_graph.addNode("Final", fn=final, outputs=["size"])
    eval_graph.addDependency("Generator", "Manipulator", preserve_source_outputs=True)
    eval_graph.addDependency("Manipulator", "Final", preserve_source_outputs=True)

    trace = np.linspace(20.0, 40.0, 2)
    sweep = hamil.newSweepConfig()
    sweep.setEvaluationGraph(eval_graph)
    sweep.add("C", trace)
    result = hamil.runSweep(sweep)

    # All evaluation results by default
    all_results = result.getOutputs("C")

    # Individual evaluation results
    size_result = result.getNumericalOutput("C", eval_output=("Final", "size"))
    assert np.array_equal(size_result, np.array([6., 6.]))
    assert np.array_equal(size_result, all_results[("Final", "size")])

    gen_result = result.getNumericalOutput("C", eval_output=("Generator", "array"))
    assert np.array_equal(gen_result, np.array([input_array, input_array]))
    assert np.array_equal(gen_result, all_results[("Generator", "array")])

    man_result = result.getNumericalOutput("C", eval_output=("Manipulator", "array"))
    assert np.array_equal(man_result, np.array([input_array.T, input_array.T]))
    assert np.array_equal(man_result, all_results[("Manipulator", "array")])


def test_sweep_axis_reordering():
    hamil = get_numerical_system(get_single_node_graph())
    hamil.setParameterValues(
        "C", 20.0, # In fF
        "I", 40e-3, # In uA
        "L", 50.0  # In pH
    )

    def generator() -> list[float]:
        return [hamil.getParameterValue("C"), hamil.getParameterValue("L")]

    eval_graph = EvaluationGraph()
    eval_graph.addNode("Generator", fn=generator, outputs=["array"])

    # Sweep one dimension, no reordering
    sweep = hamil.newSweepConfig()
    sweep.setEvaluationGraph(eval_graph)
    sweep.add("C", np.linspace(20.0, 40.0, 2))
    result = hamil.runSweep(sweep)
    all_results = result.getOutputs("C")
    assert all_results[('Generator', 'array')][0] == [20.0, 50.0]
    assert all_results[('Generator', 'array')][1] == [40.0, 50.0]

    # Sweep two dimensions, but request a trace that reorders the L axis from 1 to 0
    sweep = hamil.newSweepConfig()
    sweep.setEvaluationGraph(eval_graph)
    sweep.add("C", np.linspace(20.0, 40.0, 2))
    sweep.add("L", np.linspace(50.0, 60.0, 2))
    result = hamil.runSweep(sweep)
    all_results = result.getOutputs("L", {"C": 20})
    assert all_results[('Generator', 'array')][0] == [20.0, 50.0]
    assert all_results[('Generator', 'array')][1] == [20.0, 60.0]
    all_results = result.getOutputs("L", {"C": 40})
    assert all_results[('Generator', 'array')][0] == [40.0, 50.0]
    assert all_results[('Generator', 'array')][1] == [40.0, 60.0]

    # Original axes
    all_results = result.getOutputs(["C", "L"])
    assert all_results[('Generator', 'array')][0, 0] == [20.0, 50.0]
    assert all_results[('Generator', 'array')][1, 0] == [40.0, 50.0]

    # Swapped axes
    all_results = result.getOutputs(["L", "C"])
    assert all_results[('Generator', 'array')][0, 0] == [20.0, 50.0]
    assert all_results[('Generator', 'array')][1, 0] == [20.0, 60.0]


def create_test_numerical_system() -> NumericalSystem:
    graph = CircuitGraph()
    graph.addBranch(0, 1, "I")
    graph.addBranch(0, 1, "C")
    graph.addBranch(0, 1, "L")

    hamil = NumericalSystem(SymbolicSystem(graph))
    hamil.configureOperator(1, 40, "charge")
    hamil.setParameterValues("C", 20.0, "I", 40e-3, "L", 50.0)
    return hamil


def test_get_numerical_output_non_numpy_array():
    hamil = create_test_numerical_system()

    def make_list() -> list:
        return [object(), object()]

    eval_graph = EvaluationGraph()
    eval_graph.addNode("Producer", fn=make_list, outputs=["items"])

    sweep_config = hamil.newSweepConfig()
    sweep_config.setEvaluationGraph(eval_graph)
    sweep_config.add("C", np.linspace(20.0, 40.0, 3))
    sweep = hamil.runSweep(sweep_config)

    # This must not raise ValueError
    items = sweep.getNumericalOutput("C", eval_output=("Producer", "items"))
    assert items.shape == (3,)
    assert isinstance(items[0], list)
    assert len(items[0]) == 2


def test_get_numerical_output_eigenvalues_with_vectors():
    hamil = create_test_numerical_system()
    hamil.setDiagConfig(get_vectors=True, eigvalues=5)

    eval_graph = EvaluationGraph()
    eval_graph.addNode("Hamiltonian", fn=hamil.getHamiltonian, outputs=["qobj"])
    eval_graph.addNode("Spectrum", fn=hamil.diagonalize, outputs=["energies", "vectors"])
    eval_graph.addDependency("Hamiltonian", "Spectrum")

    sweep_config = hamil.newSweepConfig()
    sweep_config.setEvaluationGraph(eval_graph)
    sweep_config.add("C", np.linspace(20.0, 40.0, 3))
    sweep = hamil.runSweep(sweep_config)

    energies = sweep.getNumericalOutput("C", eval_output=("Spectrum", "energies"))
    assert isinstance(energies, np.ndarray)
    assert energies.shape == (3, 5)

    vectors = sweep.getNumericalOutput("C", eval_output=("Spectrum", "vectors"))
    assert isinstance(vectors, np.ndarray)
    assert vectors.shape == (3, 5)

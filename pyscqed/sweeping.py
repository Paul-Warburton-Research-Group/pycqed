"""Defines the sweeping API"""
import os

import numpy as np
import sympy as sy

from abc import abstractmethod
from typing import TypeAlias, Generator, Any

from .parameters import ParamCollection
from .evaluation_graph import EvaluationGraph
from .util import pickleRead
from .result import SweepNumericalResult, NumericalResult, EvaluationResult


EvaluationOutput: TypeAlias = tuple[str, str]
SweepValue: TypeAlias = np.float64 | np.int64
SweepPoint: TypeAlias = dict[str, SweepValue]
SweepVector: TypeAlias = np.ndarray[SweepValue]
SweepSubstitution: TypeAlias = dict[sy.Symbol, SweepValue]
ALLOWED_DTYPES = {"int64", "float64"}


class SweepConfig:
    def __init__(self, collection: ParamCollection):
        self._collection = collection
        self._sweep_data: dict[str, SweepVector] = {}
        self._evaluation_graph: EvaluationGraph | None = None

    def add(self, name: str, values: list[float] | SweepVector):
        """Adds a sweep dimension with the specified values."""
        if name in self._sweep_data:
            raise ValueError(f"Parameter \"{name}\" is already in the sweep.")
        if name not in self._collection.getParameterNamesList():
            raise ValueError(f"Parameter \"{name}\" is not in the parameter collection.")
        internal_values = np.array(values)
        if internal_values.ndim != 1:
            raise ValueError(f"Parameter \"{name}\" sweep values must be one-dimensional.")
        if internal_values.dtype.name not in ALLOWED_DTYPES:
            raise TypeError(f"Parameter \"{name}\" sweep values must be of types {ALLOWED_DTYPES}.")
        self._sweep_data[name] = internal_values

    def get(self, name: str) -> SweepVector:
        """Retrieves a sweep dimension."""
        return self._sweep_data[name]

    def setEvaluationGraph(self, graph: EvaluationGraph):
        """Sets the evaluation graph for this sweep."""
        self._evaluation_graph = graph

    def getEvaluationGraph(self) -> EvaluationGraph | None:
        """Gets the evaluation graph."""
        return self._evaluation_graph

    def getDimensionCount(self) -> int:
        """Gets the number of dimensions in the sweep."""
        return len(self._sweep_data)

    def getTotalCount(self) -> int:
        """Gets the total number of sweep points."""
        return int(np.prod([len(value) for value in self._sweep_data.values()]))

    def getSweepShape(self) -> tuple[int]:
        """Gets the shape of the sweep dimensions."""
        return tuple([len(value) for value in self._sweep_data.values()])

    def getSweptSymbols(self) -> SweepSubstitution:
        """Returns a symbol value mapping for symbols that will be swept by the current sweep. These will be
        substituted during the sweep runtime.
        """
        # Need to ensure that we substitute the parameterised parameter
        not_for_presub = set()
        for name in self._sweep_data:
            not_for_presub.add(name)
            params = self._collection.getParameterisationsInvolving(name)
            for param in params:
                successors = self._collection.getParameterisationSuccessors(param)
                not_for_presub.add(param)
                for k, v in successors.items():
                    not_for_presub |= set(v)
        actual_sub_names = list(not_for_presub)
        return self._collection.getSymbolValues(*actual_sub_names)

    def getStaticSymbols(self) -> SweepSubstitution:
        """Returns a symbol value mapping for symbols that will not be swept. These are static symbols that
        can be substituted before the sweep runtime.
        """
        # For each parameter we want to find it's parametric dependencies, and then exclude those.
        not_for_presub = set()
        for name in self._sweep_data:
            not_for_presub.add(name)
            params = self._collection.getParameterisationsInvolving(name)
            for param in params:
                successors = self._collection.getParameterisationSuccessors(param)
                not_for_presub.add(param)
                for k, v in successors.items():
                    not_for_presub |= set(v)
        
        non_sweep = list(set(self._collection.getParameterNamesList()) - not_for_presub)
        return self._collection.getSymbolValues(*non_sweep)

    def getGenerator(self) -> Generator[SweepPoint, None, None]:
        """Create a generator for sweeping all points."""
        point_count = self.getTotalCount()
        grid = {
            name: values.flatten() for name, values in 
            zip(
                self._sweep_data.keys(),
                np.meshgrid(*self._sweep_data.values(), indexing="ij", copy=False)
            )
        }
        for index in range(point_count):
            yield {name: values[index] for name, values in grid.items()}

    def getSweepAxes(self) -> dict[str, int]:
        """Get a parameter axis mapping"""
        return {param: axis for axis, param in enumerate(self._sweep_data)}


class SweepResult:
    def __init__(self, sweep_config: SweepConfig, data: np.ndarray):
        self.sweep_config = sweep_config
        self.parameter_axes = sweep_config.getSweepAxes()
        self.data: np.ndarray = data

    def getOutputs(
        self,
        independent_variables: str | list[str],
        static_variables: dict[str, SweepValue] | None = None,
        eval_outputs: list[EvaluationOutput] | None = None
    ) -> SweepNumericalResult:
        """Get the result of a sweep in a specific format, with the accompanying input vectors. The independent
        variables specify which traces to obtain, and the static variables specify the values of the other swept
        variables to take the trace along.

        The order of the independent variables affects the order of the result array axes.
        """
        if static_variables is None:
            static_variables = {}
        independent_vars = self._validate_independent_variables(independent_variables)

        # Check there are enough inputs to retrieve a sweep
        sweep_dimensions = self.sweep_config.getDimensionCount()
        if len(static_variables) + len(independent_vars) != sweep_dimensions:
            raise ValueError("Insufficient independent and static variables to retrieve sweep result data.")

        # Check the independent variables exist
        for independent_variable in independent_vars:
            if independent_variable not in self.parameter_axes:
                raise ValueError(f"Independent variable \"{independent_variable}\" not present in sweep.")

        # Reshape the data and reorder the independent variable axes
        reshaped_data = self._reshape_data(independent_vars)

        # Construct the slice specification for independent variables
        # NOTE: By default, all axes have full slice specifications, its the static variables that determine
        # which slices are constrained
        data_shape = list(reshaped_data.shape)
        slices = [slice(None)] * len(data_shape)

        # Construct the slice specification for static variables
        for param, value in static_variables.items():
            sweep_vector = self.sweep_config.get(param)
            value_index = np.argmin(np.abs(sweep_vector - value))
            slices[self.parameter_axes[param]] = value_index

        # Get the evaluation outputs
        evaluation_outputs = None
        if eval_outputs is None:
            evaluation_outputs = self.sweep_config.getEvaluationGraph().getEvaluationKeys()
        else:
            evaluation_outputs = eval_outputs

        # Obtain the final result
        return self._slice_data(slices, reshaped_data, evaluation_outputs, independent_vars)

    def getNumericalOutput(
        self,
        independent_variables: str | list[str],
        static_variables: dict[str, SweepValue] | None = None,
        eval_output: EvaluationOutput = ("Spectrum", "E")
    ) -> np.ndarray:
        """Get a raw (numpy) numerical result from this sweep result. This call can only retrieve a single
        evaluation output.
        """
        result = self.getOutputs(independent_variables, static_variables, [eval_output])
        return result[eval_output]

    def getInputPoints(self, variable: str) -> np.ndarray:
        """Get a single variable sweep vector."""
        return self.sweep_config.get(variable)

    def getInputMesh(self, variables: str | list[str]) -> dict[str, np.ndarray]:
        """Get a multidimensional mesh of the sweep input values."""
        independent_vars = self._validate_independent_variables(variables)
        return self._construct_input_mesh(independent_vars)

    def _validate_independent_variables(self, variables: str | list[str]) -> list[str]:
        independent_vars: list[str]
        if isinstance(variables, str):
            independent_vars = [variables]
        elif isinstance(variables, list):
            independent_vars = variables
        else:
            raise TypeError("independent_variables must be a list[str] or str.")
        return independent_vars

    def _reshape_data(self, independent_vars: list[str]) -> np.ndarray:
        sweep_shape = list(self.sweep_config.getSweepShape())
        uncollapsed = self.data.reshape(*sweep_shape)
        new_axes = [self.parameter_axes[independent_var] for independent_var in independent_vars]
        old_axes = sorted(new_axes)
        return np.moveaxis(uncollapsed, old_axes, new_axes)

    @abstractmethod
    def _retrieve_data(self, source: Any) -> EvaluationResult:
        pass

    def _construct_input_mesh(self, independent_variables: list[str]) -> dict[str, np.ndarray]:
        input_vectors = [self.sweep_config.get(name) for name in independent_variables]
        mesh = np.meshgrid(*input_vectors, indexing="ij", copy=False)
        return {name: grid for name, grid in zip(independent_variables, mesh)}

    def _slice_data(
        self,
        slices: list[slice],
        reshaped_data: np.ndarray,
        eval_outputs: list[EvaluationOutput],
        independent_vars: list[str]
    ) -> SweepNumericalResult:
        sliced_data = reshaped_data[*slices]

        # Determine the data shape in the files
        # They should all be the same
        index = [0] * len(sliced_data.shape)
        eval_result = self._retrieve_data(sliced_data[*index])
        eval_source = eval_result.source
        eval_keys = eval_result.getKeys()
        if not all(eval_key in eval_keys for eval_key in eval_outputs):
            raise ValueError("A specified evaluation key cannot be found in the result data.")

        # Initialize and populate the final data array
        # TODO: The output type handling here is ugly, should probably always require a NumericalResult type
        loaded_arrays = {}
        for eval_key in eval_outputs:
            node = eval_key[0]
            output = eval_key[1]
            raw_data = eval_result.data[node][output]
            if isinstance(raw_data, NumericalResult):
                file_data_shape = list(raw_data.data.shape)
                shape_spec = list(sliced_data.shape) + file_data_shape
                loaded_arrays[eval_key] = np.zeros(shape_spec, dtype=raw_data.data.dtype)
            elif isinstance(raw_data, np.ndarray):
                file_data_shape = list(raw_data.shape)
                shape_spec = list(sliced_data.shape) + file_data_shape
                loaded_arrays[eval_key] = np.zeros(shape_spec)
            else:
                object_type = type(raw_data)
                shape_spec = list(sliced_data.shape)
                loaded_arrays[eval_key] = np.empty(shape_spec, dtype=np.dtype(object_type))

        for index, data_source in np.ndenumerate(sliced_data):
            eval_result = self._retrieve_data(data_source)
            for node, output in eval_outputs:
                raw_data = eval_result.data[node][output]
                if isinstance(raw_data, NumericalResult):
                    loaded_arrays[(node, output)][index] = raw_data.data
                else:
                    loaded_arrays[(node, output)][index] = raw_data

        return SweepNumericalResult(
            data=loaded_arrays,
            source=eval_source
        )


class SweepResultFromMemory(SweepResult):
    def __init__(self, sweep_config: SweepConfig, data: list[EvaluationResult]):
        super().__init__(sweep_config, np.array(data))

    def _retrieve_data(self, source: Any) -> EvaluationResult:
        return source


class SweepResultFromDisk(SweepResult):
    def __init__(self, sweep_config: SweepConfig, files: list[str | bytes | os.PathLike]):
        data = np.array([str(file) for file in files])
        super().__init__(sweep_config, data)

    def _retrieve_data(self, source: Any) -> EvaluationResult:
        return pickleRead(source)

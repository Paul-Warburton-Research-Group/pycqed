"""Lowest level result data structures"""
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np


EvalKeys: TypeAlias = list[tuple[str, str]]


@dataclass
class NumericalResult:
    data: np.ndarray
    source: str | None = None


class EigenvalueResult(NumericalResult):
    pass


class EigenvectorResult(NumericalResult):
    pass


class FunctionResult(NumericalResult):
    pass


@dataclass
class EvaluationResult:
    data: dict[str, dict[str, NumericalResult]]
    source: str

    def getKeys(self) -> EvalKeys:
        keys = []
        for node in self.data:
            for output in self.data[node]:
                keys.append((node, output))
        return keys


@dataclass
class SweepNumericalResult:
    data: dict[EvalKeys, np.ndarray]
    source: str

    def __getitem__(self, subscript):
        return self.data[subscript]

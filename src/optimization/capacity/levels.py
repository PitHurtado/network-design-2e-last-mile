"""Strategies that choose a satellite's capacity levels from its peak-fleet distribution.

A strategy maps the peak fleet of one satellite (one value per scenario: the maximum over
the periods of the ceiled fleet the CA assigns it) to the positive levels it can install.
Level 0 (do not install) is always added by the analysis. A new method is a new subclass
with a `name`; it becomes selectable with `--levels <name>`.
"""

import math
from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

LEVEL_METHODS: dict[str, type["LevelMethod"]] = {}


def percentile_ceil(values: np.ndarray, q: float) -> int:
    return int(math.ceil(np.percentile(values, q)))


def round_up_even(x: float) -> int:
    c = math.ceil(x)
    return c + (c % 2)


class LevelMethod(ABC):
    name: ClassVar[str]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        LEVEL_METHODS[cls.name] = cls

    @abstractmethod
    def levels(self, peaks: np.ndarray) -> list[int]:
        """Positive capacity levels, ascending."""


class PercentilesA(LevelMethod):
    """Full useful range: from 2 below the P25 (at least 2) to 2 above the P95, in steps of 2."""

    name = "percentiles-a"

    def levels(self, peaks):
        low = max(2, round_up_even(percentile_ceil(peaks, 25)) - 2)
        high = round_up_even(percentile_ceil(peaks, 95)) + 2
        return list(range(low, high + 1, 2))


class PercentilesB(LevelMethod):
    """Three essential levels: covering the P50, the P90 and the largest observed peak."""

    name = "percentiles-b"

    def levels(self, peaks):
        return sorted(
            {
                round_up_even(percentile_ceil(peaks, 50)),
                round_up_even(percentile_ceil(peaks, 90)),
                round_up_even(float(np.max(peaks))),
            }
        )


class FixedGrid(LevelMethod):
    """The same grid for every satellite, 2..12 in steps of 2 (what input_facilities.xlsx had)."""

    name = "fixed-grid"

    def __init__(self, maximum: int = 12, step: int = 2):
        self.maximum, self.step = maximum, step

    def levels(self, peaks):
        return list(range(self.step, self.maximum + 1, self.step))


def level_method(name: str) -> LevelMethod:
    if name not in LEVEL_METHODS:
        raise ValueError(f"Unknown level method {name!r}; expected one of {sorted(LEVEL_METHODS)}.")
    return LEVEL_METHODS[name]()

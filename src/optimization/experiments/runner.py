"""`ExperimentRunner`: the build -> solve sequence every experiment shares, written once."""

from dataclasses import dataclass, field
from typing import Any

from src.optimization.instance import Instance, InstanceBuilder, InstanceSpec
from src.optimization.models.base import BaseSAAModel
from src.optimization.models.flex import FlexSAAModel


@dataclass(frozen=True)
class RunSpec:
    """One solve: which instance, which model, which Gurobi parameters."""

    instance: InstanceSpec
    solver: dict = field(default_factory=dict)
    model: type[BaseSAAModel] = FlexSAAModel
    model_kwargs: dict = field(default_factory=dict)


@dataclass
class SolvedRun:
    """A built instance, its solved model and `BaseSAAModel.solve()`'s record."""

    instance: Instance
    model: Any
    solve: dict


class ExperimentRunner:
    """Builds the instance, the model, sets the solver parameters and solves.

    `solver_overrides` is applied on top of every spec's parameters; it is how a whole
    batch is made deterministic (`Threads`, `Seed`, `WorkLimit`) without editing specs.
    """

    def __init__(self, builder: InstanceBuilder, solver_overrides: dict | None = None):
        self.builder = builder
        self.solver_overrides = dict(solver_overrides or {})

    @classmethod
    def for_version(cls, version: str, solver_overrides: dict | None = None) -> "ExperimentRunner":
        return cls(InstanceBuilder.for_version(version), solver_overrides)

    def build(self, spec: RunSpec):
        instance = self.builder.build(spec.instance)
        model = spec.model(instance, **spec.model_kwargs)
        model.set_params({**spec.solver, **self.solver_overrides})
        model.build()
        return instance, model

    def solve(self, spec: RunSpec) -> SolvedRun:
        instance, model = self.build(spec)
        return SolvedRun(instance, model, model.solve())

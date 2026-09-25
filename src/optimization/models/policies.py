"""Post-installation operating policies of `FlexSAAModel`: how Z[i,q,t,n] may relate to Y[i,q].

Each policy adds, for one facility-period-scenario, the constraints linking the operated
level to the installed one. A new policy is a new subclass with a `name`; it becomes
selectable by that name (`TypeOfFlexibility` lists the ones the experiments sweep).
"""

from abc import ABC, abstractmethod
from typing import ClassVar

from gurobipy import quicksum  # pylint: disable=E0611

from src.core.constants import TypeOfFlexibility

POLICIES: dict[str, type["OperationPolicy"]] = {}


class OperationPolicy(ABC):
    name: ClassVar[str]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        POLICIES[cls.name] = cls

    @abstractmethod
    def link(self, model, facility_id: str, levels: dict[int, float], period: int, scenario_id: str) -> None:
        """Add the Z-Y constraints of one facility, period and scenario to `model` (a FlexSAAModel)."""


class FixedOperation(OperationPolicy):
    """Always operate at the installed capacity: Z == Y."""

    name = TypeOfFlexibility.FIXED_OPERATION.value

    def link(self, model, facility_id, levels, period, scenario_id):
        for level in levels:
            model.model.addConstr(
                model.vars.Z[(facility_id, level, period, scenario_id)] == model.vars.Y[(facility_id, level)],
                name=f"R_fixed_{facility_id}_q{level}_t{period}_n{scenario_id}",
            )


class OnOffInstalled(OperationPolicy):
    """Either off or at the installed capacity: Z <= Y on every positive level."""

    name = TypeOfFlexibility.ON_OFF_INSTALLED.value

    def link(self, model, facility_id, levels, period, scenario_id):
        for level, capacity in levels.items():
            if capacity > 0:
                model.model.addConstr(
                    model.vars.Z[(facility_id, level, period, scenario_id)] <= model.vars.Y[(facility_id, level)],
                    name=f"R_onoff_{facility_id}_q{level}_t{period}_n{scenario_id}",
                )


class UpToInstalled(OperationPolicy):
    """Any level up to the installed one, including zero: no level above the installed capacity."""

    name = TypeOfFlexibility.UP_TO_INSTALLED.value

    def link(self, model, facility_id, levels, period, scenario_id):
        for installed_level, installed_capacity in levels.items():
            higher = [level for level, capacity in levels.items() if capacity > installed_capacity]
            if higher:
                model.model.addConstr(
                    quicksum(model.vars.Z[(facility_id, level, period, scenario_id)] for level in higher)
                    <= 1 - model.vars.Y[(facility_id, installed_level)],
                    name=f"R_upto_{facility_id}_q{installed_level}_t{period}_n{scenario_id}",
                )


def policy_for(name: str) -> OperationPolicy:
    if name not in POLICIES:
        raise ValueError(f"Unknown operating policy {name!r}; expected one of {sorted(POLICIES)}.")
    return POLICIES[name]()

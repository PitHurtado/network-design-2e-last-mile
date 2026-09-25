"""Capacitated SAA model: choose one installed capacity level per satellite.

Adds to the uncapacitated model the here-and-now installation `Y[i,q]` (level 0 means
not installed), its one-time cost, and a capacity that bounds the fleet a satellite can
dispatch in every period and scenario by the installed level. The capacity is fixed once
installed: there is no operating decision and no operating cost (that is `FlexSAAModel`).

Levels, capacities and installation costs come from `input_facilities.xlsx`. Cost keys are
indexed strictly, where `OLD/src/models/capacitated_saa_model.py` priced a missing key at 0.
"""

from dataclasses import dataclass

from gurobipy import GRB, quicksum  # pylint: disable=E0611

from src.optimization.models.base import Block, ModelFeatures
from src.optimization.models.uncapacitated import UncapacitatedSAAModel


@dataclass(frozen=True)
class CapacitatedFeatures(ModelFeatures):
    install_levels: bool = True
    capacity: bool = True


class CapacitatedSAAModel(UncapacitatedSAAModel):
    """Installation `Y[i,q]` plus fleet capacity on the installed level."""

    NAME = "capacitated"
    Features = CapacitatedFeatures
    DEFAULT_FEATURES = CapacitatedFeatures()
    BLOCKS = (
        Block("install", "var", "_vars_install", flag="install_levels", solution=("installation", "_solution_install")),
        Block(
            "installation",
            "obj",
            "_obj_installation",
            flag="install_levels",
            averaged=False,
            field="cost_installation",
            cost_key="installation_cost",
        ),
        Block("one_install_level", "constr", "_constr_one_install_level", flag="install_levels"),
        Block("fix_installation", "constr", "_constr_fix_installation", flag="install_levels", enabled_if="fixes_installation"),
        Block("capacity", "constr", "_constr_capacity", flag="capacity"),
    )

    def __init__(self, instance, features=None, fixed_installation: dict[str, float] | None = None):
        super().__init__(instance, features)
        self.levels = {
            facility_id: {int(level): float(capacity) for level, capacity in facility.capacity.items()}
            for facility_id, facility in self.instance.facilities.items()
        }
        self.fixed_installation = self._validate_fixed_installation(fixed_installation)

    def _validate_features(self) -> None:
        super()._validate_features()
        if self.features.capacity and not self.features.install_levels:
            raise ValueError("capacity=True requires install_levels=True: the capacity constraint is written over Y levels.")

    @property
    def fixes_installation(self) -> bool:
        """An evaluation model: Y is given, only the recourse is optimized."""
        return self.fixed_installation is not None

    def _validate_fixed_installation(self, fixed_installation: dict[str, float] | None) -> dict[str, int] | None:
        """Map persisted capacities to their exact level, rejecting incompatible Y values."""
        if fixed_installation is None:
            return None
        if set(fixed_installation) != set(self.levels):
            raise ValueError("fixed_installation must contain exactly the facilities in the evaluation instance.")
        selected = {}
        for facility_id, capacity in fixed_installation.items():
            matches = [level for level, value in self.levels[facility_id].items() if abs(value - float(capacity)) < 1e-9]
            if len(matches) != 1:
                raise ValueError(f"Capacity {capacity} for facility {facility_id} is not an available installation level.")
            selected[facility_id] = matches[0]
        return selected

    def _vars_install(self) -> None:
        self.vars.Y = {
            (facility_id, level): self.model.addVar(vtype=GRB.BINARY, name=f"Y_i{facility_id}_q{level}")
            for facility_id, levels in self.levels.items()
            for level in levels
        }

    def _obj_installation(self):
        return quicksum(
            float(self.instance.facilities[facility_id].cost_installation[str(level)]) * self.vars.Y[(facility_id, level)]
            for facility_id, levels in self.levels.items()
            for level, capacity in levels.items()
            if capacity > 0
        )

    def _constr_one_install_level(self) -> None:
        for facility_id, levels in self.levels.items():
            self.model.addConstr(
                quicksum(self.vars.Y[(facility_id, level)] for level in levels) == 1,
                name=f"R_install_{facility_id}",
            )

    def _constr_fix_installation(self) -> None:
        for facility_id, levels in self.levels.items():
            selected = self.fixed_installation[facility_id]
            for level in levels:
                self.model.addConstr(
                    self.vars.Y[(facility_id, level)] == int(level == selected),
                    name=f"R_fixed_install_{facility_id}_q{level}",
                )

    def _installed_capacity(self, facility_id: str, period: int, scenario_id: str):
        """Capacity available to a facility in one period and scenario: here, the installed level."""
        return quicksum(capacity * self.vars.Y[(facility_id, level)] for level, capacity in self.levels[facility_id].items())

    def _constr_capacity(self) -> None:
        for scenario_id, scenario in self.instance.scenarios.items():
            fleet = scenario.get_fleet_size("facility")
            for facility_id in self.instance.facilities:
                for period in range(self.instance.periods):
                    self.model.addConstr(
                        quicksum(
                            self.vars.X[(facility_id, pixel_id, period, scenario_id)]
                            * round(fleet[(facility_id, pixel_id, "small", period, scenario_id)], 1)
                            for pixel_id in scenario.pixels
                        )
                        <= self._installed_capacity(facility_id, period, scenario_id),
                        name=f"R_capacity_{facility_id}_t{period}_n{scenario_id}",
                    )

    def _solution_install(self) -> list[dict]:
        installation = []
        for facility_id, levels in self.levels.items():
            selected = next(level for level in levels if self.vars.Y[(facility_id, level)].X > 0.5)
            installation.append({"facility": facility_id, "installed": levels[selected] > 0, "capacity": levels[selected]})
        return installation

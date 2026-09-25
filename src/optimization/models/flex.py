"""Capacitated SAA model with three post-installation flexibility policies."""

from gurobipy import GRB, quicksum  # pylint: disable=E0611

from src.core.constants import TypeOfFlexibility
from src.optimization.models.base import BaseSAAModel, ModelFeatures


class FlexSAAModel(BaseSAAModel):
    """Choose one installed capacity and operate it under an explicit flexibility policy.

    Installation ``Y[i,q]`` is here-and-now. Operational capacity ``Z[i,q,t,n]`` is
    scenario and period dependent: fixed must equal Y, on/off can be zero or Y, and
    up-to-installed can be any lower level including zero.
    """

    NAME = "flex"
    DEFAULT_FEATURES = ModelFeatures(install_levels=True, flex_operation=True, capacity=True)

    def __init__(self, instance, features=None, fixed_installation: dict[str, float] | None = None):
        self.flexibility = TypeOfFlexibility(instance.config.type_of_flexibility)
        if instance.config.is_continuous_var_x:
            raise ValueError("FlexSAAModel requires binary assignment variables X and W; set is_continuous_var_x=False.")
        super().__init__(instance, features)
        self.levels = {
            facility_id: {int(level): float(capacity) for level, capacity in facility.capacity.items()}
            for facility_id, facility in self.instance.facilities.items()
        }
        self.fixed_installation = self._validate_fixed_installation(fixed_installation)

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

    def _vars_operation(self) -> None:
        self.vars.Z = {
            (facility_id, level, period, scenario_id): self.model.addVar(
                vtype=GRB.BINARY, name=f"Z_i{facility_id}_q{level}_t{period}_n{scenario_id}"
            )
            for facility_id, levels in self.levels.items()
            for level in levels
            for period in range(self.instance.periods)
            for scenario_id in self.instance.scenarios
        }

    def _obj_installation(self):
        return quicksum(
            float(self.instance.facilities[facility_id].cost_installation[str(level)]) * self.vars.Y[(facility_id, level)]
            for facility_id, levels in self.levels.items()
            for level, capacity in levels.items()
            if capacity > 0
        )

    def _operating_cost(self, facility_id: str, level: int, period: int) -> float:
        values = self.instance.facilities[facility_id].cost_operation[str(level)]
        return float(sum(values) / len(values)) if self.instance.periods == 1 else float(values[period])

    def _obj_operation(self):
        return quicksum(
            self._operating_cost(facility_id, level, period) * self.vars.Z[(facility_id, level, period, scenario_id)]
            for facility_id, levels in self.levels.items()
            for level, capacity in levels.items()
            if capacity > 0
            for period in range(self.instance.periods)
            for scenario_id in self.instance.scenarios
        )

    def _constr_one_install_level(self) -> None:
        for facility_id, levels in self.levels.items():
            self.model.addConstr(
                quicksum(self.vars.Y[(facility_id, level)] for level in levels) == 1,
                name=f"R_install_{facility_id}",
            )
            if self.fixed_installation is not None:
                selected = self.fixed_installation[facility_id]
                for level in levels:
                    self.model.addConstr(
                        self.vars.Y[(facility_id, level)] == int(level == selected),
                        name=f"R_fixed_install_{facility_id}_q{level}",
                    )

    def _constr_one_operation_level(self) -> None:
        for facility_id, levels in self.levels.items():
            for scenario_id in self.instance.scenarios:
                for period in range(self.instance.periods):
                    self.model.addConstr(
                        quicksum(self.vars.Z[(facility_id, level, period, scenario_id)] for level in levels) == 1,
                        name=f"R_operation_{facility_id}_t{period}_n{scenario_id}",
                    )

    def _constr_operation_le_install(self) -> None:
        for facility_id, levels in self.levels.items():
            for scenario_id in self.instance.scenarios:
                for period in range(self.instance.periods):
                    if self.flexibility is TypeOfFlexibility.FIXED_OPERATION:
                        for level in levels:
                            self.model.addConstr(
                                self.vars.Z[(facility_id, level, period, scenario_id)] == self.vars.Y[(facility_id, level)],
                                name=f"R_fixed_{facility_id}_q{level}_t{period}_n{scenario_id}",
                            )
                    elif self.flexibility is TypeOfFlexibility.ON_OFF_INSTALLED:
                        for level, capacity in levels.items():
                            if capacity > 0:
                                self.model.addConstr(
                                    self.vars.Z[(facility_id, level, period, scenario_id)] <= self.vars.Y[(facility_id, level)],
                                    name=f"R_onoff_{facility_id}_q{level}_t{period}_n{scenario_id}",
                                )
                    else:
                        for installed_level, installed_capacity in levels.items():
                            higher = [level for level, capacity in levels.items() if capacity > installed_capacity]
                            if higher:
                                self.model.addConstr(
                                    quicksum(self.vars.Z[(facility_id, level, period, scenario_id)] for level in higher)
                                    <= 1 - self.vars.Y[(facility_id, installed_level)],
                                    name=f"R_upto_{facility_id}_q{installed_level}_t{period}_n{scenario_id}",
                                )

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
                        <= quicksum(
                            capacity * self.vars.Z[(facility_id, level, period, scenario_id)]
                            for level, capacity in self.levels[facility_id].items()
                        ),
                        name=f"R_capacity_{facility_id}_t{period}_n{scenario_id}",
                    )

    def decisions(self) -> dict:
        """Serializable installation and recourse decisions after a feasible solve."""
        installation = []
        for facility_id, levels in self.levels.items():
            selected = next(level for level in levels if self.vars.Y[(facility_id, level)].X > 0.5)
            installation.append({"facility": facility_id, "installed": levels[selected] > 0, "capacity": levels[selected]})
        operation = [
            {"facility": facility_id, "capacity": capacity, "period": period, "scenario": scenario_id}
            for facility_id, levels in self.levels.items()
            for scenario_id in self.instance.scenarios
            for period in range(self.instance.periods)
            for level, capacity in levels.items()
            if self.vars.Z[(facility_id, level, period, scenario_id)].X > 0.5
        ]
        return {"installation": installation, "operation": operation}

    def scenario_costs(self) -> list[dict]:
        """Return unaveraged second-stage costs for every scenario in a solved evaluation.

        Installation is deliberately excluded: it is a first-stage, one-time cost and
        is reported separately so a distribution never counts it 100 times by mistake.
        """
        installation = self._obj_installation().getValue()
        rows = []
        for scenario_id, scenario in self.instance.scenarios.items():
            operation = quicksum(
                self._operating_cost(facility_id, level, period) * self.vars.Z[(facility_id, level, period, scenario_id)]
                for facility_id, levels in self.levels.items()
                for level, capacity in levels.items()
                if capacity > 0
                for period in range(self.instance.periods)
            ).getValue()
            routing_facilities = quicksum(
                scenario.get_cost_serving("facility")[(facility_id, pixel_id, "small", period, scenario_id)]
                * self.vars.X[(facility_id, pixel_id, period, scenario_id)]
                for facility_id in self.instance.facilities
                for pixel_id in scenario.pixels
                for period in range(self.instance.periods)
            ).getValue()
            routing_dc = quicksum(
                scenario.get_cost_serving("dc")[(pixel_id, "large", period, scenario_id)]
                * self.vars.W[(pixel_id, period, scenario_id)]
                for pixel_id in scenario.pixels
                for period in range(self.instance.periods)
            ).getValue()
            second_stage = operation + routing_facilities + routing_dc
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "installation_cost": round(installation, 3),
                    "operation_cost": round(operation, 3),
                    "routing_facilities_cost": round(routing_facilities, 3),
                    "routing_dc_cost": round(routing_dc, 3),
                    "second_stage_cost": round(second_stage, 3),
                    "total_cost": round(installation + second_stage, 3),
                }
            )
        return rows

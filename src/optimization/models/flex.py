"""Flexible capacitated SAA model: operate the installed capacity under an explicit policy.

Adds to the capacitated model the operated level `Z[i,q,t,n]`, chosen per period and
scenario, its operating cost, and an `OperationPolicy` linking Z to the installed Y
(`policies.py`: fixed, on/off, up to installed). The capacity is then written over Z.
"""

from dataclasses import dataclass

from gurobipy import GRB, quicksum  # pylint: disable=E0611

from src.optimization.models.base import Block
from src.optimization.models.capacitated import CapacitatedFeatures, CapacitatedSAAModel
from src.optimization.models.policies import OperationPolicy, policy_for


@dataclass(frozen=True)
class FlexFeatures(CapacitatedFeatures):
    flex_operation: bool = True


class FlexSAAModel(CapacitatedSAAModel):
    """Choose one installed capacity and operate it under an explicit flexibility policy.

    Installation ``Y[i,q]`` is here-and-now. Operational capacity ``Z[i,q,t,n]`` is
    scenario and period dependent: fixed must equal Y, on/off can be zero or Y, and
    up-to-installed can be any lower level including zero.
    """

    NAME = "flex"
    USES_POLICY = True
    Features = FlexFeatures
    DEFAULT_FEATURES = FlexFeatures()
    BLOCKS = (
        Block("operation", "var", "_vars_operation", flag="flex_operation", solution=("operation", "_solution_operation")),
        Block(
            "operation_cost",
            "obj",
            "_obj_operation",
            flag="flex_operation",
            averaged=True,
            field="cost_operation",
            cost_key="operation_cost",
        ),
        Block("one_operation_level", "constr", "_constr_one_operation_level", flag="flex_operation", before="capacity"),
        Block("operation_le_install", "constr", "_constr_operation_le_install", flag="flex_operation", before="capacity"),
    )

    def __init__(self, instance, features=None, fixed_installation=None, policy: OperationPolicy | None = None):
        self.policy = policy or policy_for(instance.config.type_of_flexibility)
        if instance.config.is_continuous_var_x:
            raise ValueError("FlexSAAModel requires binary assignment variables X and W; set is_continuous_var_x=False.")
        super().__init__(instance, features, fixed_installation)

    def _validate_features(self) -> None:
        super()._validate_features()
        if self.features.flex_operation and not self.features.install_levels:
            raise ValueError("flex_operation=True requires install_levels=True: Z is bounded by the installed level Y.")
        if self.features.capacity and not self.features.flex_operation:
            raise ValueError("FlexSAAModel writes capacity over Z; without flex_operation use CapacitatedSAAModel.")

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

    def _operating_cost(self, facility_id: str, level: int, period: int) -> float:
        values = self.instance.facilities[facility_id].cost_operation[str(level)]
        return float(sum(values) / len(values)) if self.instance.periods == 1 else float(values[period])

    def _obj_operation(self) -> dict:
        return {
            scenario_id: quicksum(
                self._operating_cost(facility_id, level, period) * self.vars.Z[(facility_id, level, period, scenario_id)]
                for facility_id, levels in self.levels.items()
                for level, capacity in levels.items()
                if capacity > 0
                for period in range(self.instance.periods)
            )
            for scenario_id in self.instance.scenarios
        }

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
                    self.policy.link(self, facility_id, levels, period, scenario_id)

    def _installed_capacity(self, facility_id: str, period: int, scenario_id: str):
        """The operated level, not the installed one, bounds the fleet."""
        return quicksum(
            capacity * self.vars.Z[(facility_id, level, period, scenario_id)]
            for level, capacity in self.levels[facility_id].items()
        )

    def _solution_operation(self) -> list[dict]:
        return [
            {"facility": facility_id, "capacity": capacity, "period": period, "scenario": scenario_id}
            for facility_id, levels in self.levels.items()
            for scenario_id in self.instance.scenarios
            for period in range(self.instance.periods)
            for level, capacity in levels.items()
            if self.vars.Z[(facility_id, level, period, scenario_id)].X > 0.5
        ]

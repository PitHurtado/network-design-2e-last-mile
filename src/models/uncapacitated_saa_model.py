"""Module of class Uncapacitated SAA Model."""

import time
from dataclasses import dataclass
from typing import Dict

import gurobipy as gb
from gurobipy import GRB, quicksum  # pylint: disable=E0611

from src.utils.custom_logger import get_logger
from src.utils.instance import Instance
from src.utils.scenario import Scenario

logger = get_logger("UncapacitatedSAAModel")


@dataclass
class VariablesModel:
    """Class for variables in the model."""

    # W[j,t,n]: pixel j is served from the DC in period t in scenario n
    W: dict[tuple[str, int, str], gb.Var]

    # X[i,j,t,n]: pixel j is served from facility i in period t in scenario n
    X: dict[tuple[str, str, int, str], gb.Var]


@dataclass
class ObjectiveModel:
    """Class for objective components in the model."""

    cost_served_from_dc: gb.LinExpr = None
    cost_served_from_facilities: gb.LinExpr = None
    cost_total: gb.LinExpr = None


class UncapacitatedSAAModel:
    """Uncapacitated SAA Model — distribution costs only, no installation or operation costs."""

    def __init__(self, instance: Instance):
        self.model = gb.Model(name="Uncapacitated")

        self.instance: Instance = instance
        self.is_continuous_x: bool = instance.config.is_continuous_var_x

        self.vars = VariablesModel(W={}, X={})
        self.obj = ObjectiveModel()
        self.results = {}

    def build(self) -> None:
        """Build the model."""
        logger.info("Building model")
        self.__add_variables(self.instance.facilities, self.instance.scenarios)
        self.__add_objective(self.instance.facilities, self.instance.scenarios)
        self.__add_constraints(self.instance.facilities, self.instance.scenarios)

        self.model.update()
        logger.info("Model built")

    def __add_variables(self, facilities: Dict, scenarios: Dict[str, Scenario]) -> None:
        """Add variables to model."""
        logger.info("Adding variables to model")
        type_variable = GRB.CONTINUOUS if self.is_continuous_x else GRB.BINARY
        logger.info(f"Using {'continuous' if self.is_continuous_x else 'binary'} variables for X[i,j,t,n]")

        self.vars.X = dict(
            [
                (
                    (i, k, t, n),
                    self.model.addVar(vtype=type_variable, name=f"X_i{i}_k{k}_t{t}_n{n}", lb=0, ub=1),
                )
                for i in facilities.keys()
                for n, scenario in scenarios.items()
                for k in scenario.pixels.keys()
                for t in range(self.instance.periods)
            ]
        )
        logger.info(f"Number of variables X: {len(self.vars.X)}")

        self.vars.W = dict(
            [
                (
                    (k, t, n),
                    self.model.addVar(vtype=type_variable, name=f"W_k{k}_t{t}_n{n}", lb=0, ub=1),
                )
                for n, scenario in scenarios.items()
                for k in scenario.pixels.keys()
                for t in range(self.instance.periods)
            ]
        )
        logger.info(f"Number of variables W: {len(self.vars.W)}")

        self.model._X = self.vars.X
        self.model._W = self.vars.W

    def __add_objective(self, facilities: Dict, scenarios: Dict[str, Scenario]) -> None:
        """Add objective to model."""
        logger.info("Adding objective to model")

        self.obj.cost_served_from_facilities = quicksum(
            [
                round(scenario.get_cost_serving("facility")[(i, k, "small", t, n)], 5) * self.vars.X[(i, k, t, n)]
                for i in facilities.keys()
                for n, scenario in scenarios.items()
                for k in scenario.pixels.keys()
                for t in range(self.instance.periods)
            ]
        )

        self.obj.cost_served_from_dc = quicksum(
            [
                round(scenario.get_cost_serving("dc")[(k, "large", t, n)], 5) * self.vars.W[(k, t, n)]
                for n, scenario in scenarios.items()
                for k in scenario.pixels.keys()
                for t in range(self.instance.periods)
            ]
        )

        self.obj.cost_total = (1 / len(scenarios)) * (
            self.obj.cost_served_from_facilities + self.obj.cost_served_from_dc
        )
        self.model._cost_total = self.obj.cost_total
        self.model.setObjective(self.obj.cost_total, GRB.MINIMIZE)

    def __add_constraints(self, facilities: Dict, scenarios: Dict[str, Scenario]) -> None:
        """Add constraints to model."""
        logger.info("Adding constraints to model")
        self.__add_constr_A_5(facilities, scenarios)

    def __add_constr_A_5(self, facilities: Dict, scenarios: Dict[str, Scenario]) -> None:
        logger.info("--------- Adding constraints A.5 - Demand satisfaction")
        for n, scenario in scenarios.items():
            for t in range(self.instance.periods):
                for k in scenario.pixels.keys():
                    name_constraint = f"R_demand_k{k}_t{t}_n{n}"
                    self.model.addConstr(
                        quicksum([self.vars.X[(i, k, t, n)] for i in facilities.keys()])
                        + quicksum([self.vars.W[(k, t, n)]])
                        >= 1,
                        name=name_constraint,
                    )

    def solve(self):
        """Solve the model."""
        logger.info("Solving model")
        start_time = time.time()
        self.model.optimize()
        logger.info("Model solved")
        try:
            mip_gap = round(100 * self.model.MIPGap, 3)
        except Exception:
            mip_gap = 0.0  # LP has no MIPGap
        self.results = {
            "actual_run_time": round(time.time() - start_time, 3),
            "optimality_gap": mip_gap,
            "objective_value": round(self.model._cost_total.getValue(), 3),
            "best_bound_value": round(self.model.ObjBound, 3),
        }
        return self.results

    def set_params(self, params: Dict[str, int]) -> None:
        """Set parameters to model."""
        logger.info(f"Setting parameters: {params}")
        for key, item in params.items():
            self.model.setParam(key, item)

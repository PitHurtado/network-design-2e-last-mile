"""Module of class Extended SAA Model."""

import time
from dataclasses import dataclass
from typing import Dict

import gurobipy as gb
from gurobipy import GRB, quicksum  # pylint: disable=E0611

from src.utils.classes import Facility
from src.utils.custom_logger import get_logger
from utils.instance import Instance
from utils.scenario import Scenario

logger = get_logger("ExtendedSAAModel")

FLEX_CAPACITY = "flex-capacity"
FIXED_CAPACITY = "fixed-capacity"


@dataclass
class VariablesModel:
    """Class for variables in the model."""

    # Z[i,q,t,n]: facility i operates at capacity q in period t in scenario n
    Z: dict[tuple[str, str, int, str], gb.Var]

    # Y[i,q]: facility i is installed with capacity q
    Y: dict[tuple[str, str], gb.Var]

    # W[j,t,n]: pixel j is served from the DC in period t in scenario n
    W: dict[tuple[str, int, str], gb.Var]

    # X[i,j,t,n]: pixel j is served from facility i in period t in scenario n
    X: dict[tuple[str, str, int, str], gb.Var]


@dataclass
class ObjectiveModel:
    """Class for objective components in the model."""

    cost_installation_facilities: gb.LinExpr = None
    cost_served_from_dc: gb.LinExpr = None
    cost_served_from_facilities: gb.LinExpr = None
    cost_operating_facilities: gb.LinExpr = None
    cost_total: gb.LinExpr = None


class ExtendedSAAModel:
    """Extended SAA Model."""

    def __init__(self, instance: Instance):
        self.model = gb.Model(name="Extended")

        # Params from instance and scenarios
        self.instance: Instance = instance

        # Params from instance-config
        self.type_of_flexibility: str = instance.config.type_of_flexibility
        self.is_continuous_x: bool = instance.config.is_continuous_var_x

        # Variables
        self.vars = VariablesModel(Z={}, Y={}, W={}, X={})

        # Objective
        self.obj = ObjectiveModel()

        # Results
        self.results = {}

    # def __calculate_basecase_alpha(self) -> None:
    #     """Calculate basecase alpha."""
    #     logger.info("[EXTENDED SAA] Calculate basecase alpha")
    #     # Initialize total cost and total count of pixels
    #     total_dc_serving_costs = 0
    #     total_pixels_dc = 0

    #     current_dc_serving_costs = 0

    #     # Iterate over all scenarios
    #     for n, scenario in self.scenarios.items():
    #         num_pixels = len(scenario.pixels)
    #         total_pixels_dc += num_pixels  # Accumulate the total number of pixels across all scenarios

    #         # Sum up the costs for each pixel and time period
    #         for t in range(self.periods):
    #             total_dc_serving_costs = 0
    #             for k in scenario.pixels.keys():
    #                 total_dc_serving_costs += round(scenario.get_cost_serving("dc")[(k, t)]["total"], 0)

    #             current_dc_serving_costs += total_dc_serving_costs/num_pixels

    #     current_dc_serving_costs = current_dc_serving_costs/ (len(self.scenarios) * self.periods)

    #     # Calculate the average cost over all scenarios, time periods, and pixels
    #     # The total pixels need to be multiplied by the number of periods only during the final averaging step
    #     # SELLY: current_dc_serving_costs = total_dc_serving_costs / (len(self.scenarios) * self.periods * total_pixels_dc)  # noqa: E501

    #     # Initialize total cost and total count of pixels
    #     total_satellites_serving_costs = 0
    #     total_pixels_s = 0  # This will track the total number of pixel-period-satellite combinations

    #     # Iterate over all scenarios
    #     current_satellites_serving_costs = 0
    #     for n, scenario in self.scenarios.items():
    #         num_pixels = len(scenario.pixels)
    #         total_pixels_s += num_pixels  # Accumulate the total number of pixels across all scenarios

    #         # Sum up the costs for each satellite, pixel, and time period
    #         for t in range(self.periods):
    #             total_satellites_serving_costs = 0
    #             for s in self.satellites.keys():
    #                 for k in scenario.pixels.keys():
    #                     total_satellites_serving_costs += round(
    #                         scenario.get_cost_serving("satellite")[(s, k, t)]["total"], 0)

    #             current_satellites_serving_costs += total_satellites_serving_costs/(len(self.satellites) * num_pixels)

    #     current_satellites_serving_costs = current_satellites_serving_costs /(len(self.scenarios)* self.periods)

    #     # Calculate the average cost over all scenarios, satellites, time periods, and pixels
    #     # SELLY: current_satellites_serving_costs = total_satellites_serving_costs / (
    #             # len(self.scenarios) * len(self.satellites) * self.periods * total_pixels_s)

    #     logger.info(f"[EXTENDED SAA] avg cost to serve 1 pixel in 1 time period from dc is {current_dc_serving_costs}") # noqa: E501
    #     logger.info(f"[EXTENDED SAA] avg cost to serve 1 pixel in 1 time period from a satellite is {current_satellites_serving_costs}") # noqa: E501
    #     basecase_alpha = current_dc_serving_costs / current_satellites_serving_costs
    #     logger.info(f"[EXTENDED SAA] basecase_alpha {basecase_alpha}")

    def build(self) -> None:
        """Build the model."""
        logger.info("Building model")
        self.__add_variables(self.instance.facilities, self.instance.scenarios)
        self.__add_objective(self.instance.facilities, self.instance.scenarios)
        self.__add_constraints(self.instance.facilities, self.instance.scenarios)

        self.model.update()
        logger.info("Model built")

    def __add_variables(self, facilities: Dict[str, Facility], scenarios: Dict[str, Scenario]) -> None:
        """Add variables to model."""
        logger.info("Adding variables to model")
        # 1. add variable Z: binary variable to decide if a facility is operating in a period with a given capacity
        if self.type_of_flexibility == FLEX_CAPACITY:
            self.vars.Z = dict(
                [
                    (
                        (i, q, t, n),
                        self.model.addVar(vtype=GRB.BINARY, name=f"Z_i{i}_q{q}_t{t}_n{n}", lb=0, ub=1),
                    )
                    for i, facility in facilities.items()
                    for q, _ in facility.capacity.items()
                    for t in range(self.instance.periods)
                    for n in scenarios.keys()
                ]
            )
        logger.info(f"Number of variables Z: {len(self.vars.Z)}")

        # 2. add variable X: binary variable to decide if a facility is used to serve a pixel
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

        # 3. add variable W: binary variable to decide if a pixel is served from dc
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

        # 4. add variable Y: binary variable to decide if a facility is open or not
        self.vars.Y = dict(
            [
                (
                    (i, q),
                    self.model.addVar(vtype=GRB.BINARY, name=f"Y_i{i}_q{q}"),
                )
                for i, facility in facilities.items()
                for q, _ in facility.capacity.items()
            ]
        )
        logger.info(f"Number of variables Y: {len(self.vars.Y)}")

        self.model._X = self.vars.X
        self.model._Y = self.vars.Y
        self.model._W = self.vars.W
        self.model._Z = self.vars.Z

    def __add_objective(
        self,
        facilities: Dict[str, Facility],
        scenarios: Dict[str, Scenario],
    ) -> None:
        """Add objective to model."""
        # 1. add cost installation facilities
        logger.info("Adding objective to model")
        self.obj.cost_installation_facilities = quicksum(
            [
                round(facility.cost_fixed[q], 0) * self.vars.Y[(i, q)]
                for i, facility in facilities.items()
                for q, capacity in facility.capacity.items()
                if capacity > 0
            ]
        )

        # 2. add cost operating facilities
        if self.type_of_flexibility == FIXED_CAPACITY:
            self.obj.cost_operating_facilities = quicksum(
                [
                    round(facility.cost_operation[q][t], 0) * self.vars.Y[(i, q)]
                    for i, facility in facilities.items()
                    for q, capacity in facility.capacity.items()
                    if capacity > 0
                    for t in range(self.instance.periods)
                    for n in scenarios.keys()
                ]
            )
        elif self.type_of_flexibility == FLEX_CAPACITY:
            self.obj.cost_operating_facilities = quicksum(
                [
                    round(facility.cost_operation[q][t], 0) * self.vars.Z[(i, q, t, n)]
                    for i, facility in facilities.items()
                    for q, capacity in facility.capacity.items()
                    if capacity > 0
                    for t in range(self.instance.periods)
                    for n in scenarios.keys()
                ]
            )

        # 3. add cost served from facilities
        self.obj.cost_served_from_facilities = quicksum(
            [
                round(scenario.get_cost_serving("facility")[(i, k, t)]["total"], 0) * self.vars.X[(i, k, t, n)]
                for i in facilities.keys()
                for n, scenario in scenarios.items()
                for k in scenario.pixels.keys()
                for t in range(self.instance.periods)
            ]
        )

        # 4. add cost served from dc
        self.obj.cost_served_from_dc = quicksum(
            [
                round(scenario.get_cost_serving("dc")[(k, t)]["total"], 0) * self.vars.W[(k, t, n)]
                for n, scenario in scenarios.items()
                for k in scenario.pixels.keys()
                for t in range(self.instance.periods)
            ]
        )

        self.obj.cost_total = self.obj.cost_installation_facilities + (1 / len(scenarios)) * (
            self.obj.cost_operating_facilities + self.obj.cost_served_from_dc + self.obj.cost_served_from_facilities
        )
        self.model._cost_total = self.obj.cost_total
        self.model.setObjective(self.obj.cost_total, GRB.MINIMIZE)

    def __add_constraints(
        self,
        facilities: Dict[str, Facility],
        scenarios: Dict[str, Scenario],
    ) -> None:
        """Add constraints to model."""
        logger.info("Adding constraints to model")
        self.__add_constr_A_1(facilities)

        if self.type_of_flexibility == FLEX_CAPACITY:
            self.__add_constr_A_2(facilities, scenarios)
            self.__add_constr_A_3(facilities, scenarios)
            # TODO check this constraints (3) ?
            # if self.type_of_flexibility == 3:
            #     self.__add_constr_deactivate_z(≈, scenarios)

        self.__add_constr_A_4(facilities, scenarios)
        self.__add_constr_A_5(facilities, scenarios)
        # self.__add_valid_inequalities(satellites, scenarios)

    def __add_constr_A_1(self, facilities: Dict[str, Facility]) -> None:
        logger.info("--------- Adding constraints A.1 - Installing facilities")
        for i, facility in facilities.items():
            name_constraint = f"R_Open_i{i}"
            self.model.addConstr(
                quicksum([self.vars.Y[(i, q)] for q in facility.capacity.keys()]) == 1,
                name=name_constraint,
            )

    def __add_constr_A_2(self, facilities: Dict[str, Facility], scenarios: Dict[str, Scenario]) -> None:
        logger.info("--------- Adding constraints A.2 - Operating facilities")
        for i, facility in facilities.items():
            for n in scenarios.keys():
                for t in range(self.instance.periods):
                    name_constraint = f"R_activation_i{i}_t{t}_n{n}"
                    self.model.addConstr(
                        quicksum([self.vars.Z[(i, q, t, n)] for q, _ in facility.capacity.items()]) == 1,
                        name=name_constraint,
                    )

    def __add_constr_A_3(self, facilities: Dict[str, Facility], scenarios: Dict[str, Scenario]) -> None:
        """Add Constraints A.3."""
        logger.info("--------- Adding constraints A.3 - maximum operating facilities")
        for t in range(self.instance.periods):
            for n in scenarios.keys():
                for i, facility in facilities.items():
                    max_capacity = max(facility.capacity.values())
                    for q, capacity in facility.capacity.items():
                        if capacity < max_capacity:
                            name_constraint = f"R_Operating_i{i}_q{q}_t{t}_n{n}"
                            q_higher_values = [
                                q_higher
                                for q_higher, q_higher_capacity in facility.capacity.items()
                                if q_higher_capacity > capacity
                            ]
                            self.model.addConstr(
                                quicksum([self.vars.Z[(i, q_higher, t, n)] for q_higher in q_higher_values])
                                <= 1 - self.vars.Y[(i, q)],
                                name=name_constraint,
                            )

    def __add_constr_A_4(
        self,
        facilities: Dict[str, Facility],
        scenarios: Dict[str, Scenario],
    ) -> None:
        logger.info("--------- Adding constraints A.4 - Capacity limit")
        for t in range(self.instance.periods):
            for i, facility in facilities.items():
                for n, scenario in scenarios.items():
                    fleet_size = scenario.get_fleet_size("facility")
                    name_constraint = f"R_capacity_i{i}_t{t}_n{n}"
                    if self.type_of_flexibility == FLEX_CAPACITY:
                        self.model.addConstr(
                            quicksum(
                                [
                                    self.vars.X[(i, k, t, n)] * round(fleet_size[(i, k, t)]["fleet_size"], 1)
                                    for k in scenario.pixels.keys()
                                ]
                            )
                            - quicksum(
                                [
                                    self.vars.Z[(i, q, t, n)] * capacity
                                    for q, capacity in facility.capacity.items()
                                    if capacity > 0
                                ]
                            )
                            <= 0,
                            name=name_constraint,
                        )
                    elif self.type_of_flexibility == FIXED_CAPACITY:
                        self.model.addConstr(
                            quicksum(
                                [
                                    self.vars.X[(i, k, t, n)] * round(fleet_size[(i, k, t)]["fleet_size"], 1)
                                    for k in scenario.pixels.keys()
                                ]
                            )
                            - quicksum(
                                [
                                    self.vars.Y[(i, q)] * capacity
                                    for q, capacity in facility.capacity.items()
                                    if capacity > 0  # noqa: E501
                                ]
                            )
                            <= 0,
                            name=name_constraint,
                        )

    def __add_constr_A_5(self, facilities: Dict[str, Facility], scenarios: Dict[str, Scenario]):
        logger.info("--------- Adding constraints A.5 - Demand satisfaction")
        for n, scenario in scenarios.items():
            for t in range(self.instance.periods):
                for k in scenario.pixels.keys():
                    name_constraint = f"R_demand_k{k}_t{t}_n{n}"
                    self.model.addConstr(
                        quicksum([self.vars.X[(i, k, n, t)] for i in facilities.keys()])
                        + quicksum([self.vars.W[(k, n, t)]])  # noqa: E501
                        >= 1,
                        name=name_constraint,
                    )

    # def __add_constr_deactivate_z(
    #         self, satellites: Dict[str, Satellite], scenarios: Dict[str, Scenario]
    # ) -> None:
    #     """Add Constraints Deactivate z"""
    #     logger.info("   Add constraints A.3.1 - Deactivate Z variables")
    #     for t in range(self.periods):
    #         for n in scenarios.keys():
    #             for s, satellite in satellites.items():
    #                 for q_y_key, q_y_value in satellite.capacity.items():
    #                     if q_y_value > 0:
    #                         for q_z_key, q_z_value in satellite.capacity.items():
    #                             if q_y_key > q_z_key > 0:
    #                                 nameConstraint = f"Z_Deactivated_s{s}_q{q_z_key}"
    #                                 self.model.addConstr(self.Z[(s, q_z_key, n, t)] <= 1 - self.Y[(s, q_y_key)],
    #                                                      name=nameConstraint,
    #                                                      )

    # def __add_valid_inequalities(
    #         self, satellites: Dict[str, Satellite], scenarios: Dict[str, Scenario]
    # ):
    #     logger.info("   Adding valid inequalities")
    #     for n, scenario in scenarios.items():
    #         for t in range(self.periods):
    #             for s, satellite in satellites.items():
    #                 for k in scenario.pixels.keys():
    #                     nameConstraint = f"R_valid_inequality_s{s}_t{t}"
    #                     self.model.addConstr(
    #                         self.X[(s, k, n, t)]
    #                         <= quicksum(
    #                             [
    #                                 self.Y[(s, q)]
    #                                 for q, capacity in satellite.capacity.items()
    #                                 if capacity > 0
    #                             ]
    #                         ),
    #                         name=nameConstraint,
    #                         )

    def solve(self):
        """Solve the model."""
        logger.info("Solving model")
        start_time = time.time()
        self.model.optimize()
        logger.info("Model solved")
        self.results = {
            "actual_run_time": round(time.time() - start_time, 3),
            "optimality_gap": round(100 * self.model.MIPGap, 3),
            "objective_value": round(self.model._cost_total.getValue(), 3),
            "best_bound_value": round(self.model.ObjBound, 3),
        }
        return self.results

    def set_params(self, params: Dict[str, int]) -> None:
        """Set parameters to model."""
        logger.info(f"Setting parameters: {params}")
        for key, item in params.items():
            self.model.setParam(key, item)

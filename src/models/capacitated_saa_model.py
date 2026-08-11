"""Capacitated SAA Model — Proposal-A capacity levels + fixed opening costs.

Extends the uncapacitated model with:
  - Y[i,q]  binary: facility i installed at capacity level q (q=0 → closed)
  - Capacity constraint: fleet assigned to i ≤ installed vehicles
  - Objective: installation cost + (1/N) × routing cost
  - No operating costs in this version.
"""

import time
from dataclasses import dataclass, field
from typing import Dict

import gurobipy as gb
from gurobipy import GRB, quicksum  # pylint: disable=E0611

from src.utils.custom_logger import get_logger
from src.utils.instance import Instance
from src.utils.scenario import Scenario

logger = get_logger("CapacitatedSAAModel")

def install_cost(level: int, facility_ci: dict[int, float]) -> float:
    """Return the installation cost for a given capacity level.

    Uses the facility's own cost_installation data (from input_facilities.xlsx).
    For levels beyond the maximum known level, plateaus at the maximum cost.

    Parameters
    ----------
    level       : capacity level (number of vehicles)
    facility_ci : {level_int: cost} dict parsed from Facility.cost_installation
    """
    if level in facility_ci:
        return facility_ci[level]
    return float(max(facility_ci.values())) if facility_ci else 0.0


@dataclass
class VariablesModel:
    # X[i,j,t,n]: fraction of pixel j served from facility i in period t, scenario n
    X: dict = field(default_factory=dict)
    # W[j,t,n]:  fraction of pixel j served directly from DC
    W: dict = field(default_factory=dict)
    # Y[i,q]:    binary — facility i opened at capacity level q (q=0 → closed)
    Y: dict = field(default_factory=dict)


@dataclass
class ObjectiveModel:
    cost_installation:       gb.LinExpr = None
    cost_served_from_dc:     gb.LinExpr = None
    cost_served_from_facilities: gb.LinExpr = None
    cost_total:              gb.LinExpr = None


class CapacitatedSAAModel:
    """Capacitated SAA Model with per-satellite Proposal-A capacity levels."""

    def __init__(
        self,
        instance: Instance,
        capacity_levels: dict[str, list[int]],
    ):
        """
        Parameters
        ----------
        instance : Instance
            Loaded instance (facilities, scenarios, CA already run).
        capacity_levels : dict[str, list[int]]
            Proposal-A levels per facility, e.g.
            {"ABAROA": [6, 8, 10], "SOPOCACHI": [14, 16, 18, 20], ...}
            Level 0 (closed) is always added automatically.
        """
        self.model = gb.Model(name="Capacitated")
        self.instance = instance
        self.is_continuous_x: bool = instance.config.is_continuous_var_x
        self.capacity_levels = capacity_levels

        # Pre-build per-facility level → (vehicles, install_cost) mapping
        self._cap: dict[str, dict[int, tuple[int, float]]] = self._build_cap()

        self.vars = VariablesModel()
        self.obj  = ObjectiveModel()
        self.results: dict = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_cap(self) -> dict[str, dict[int, tuple[int, float]]]:
        """Build {facility: {level: (vehicles, install_cost)}} from Proposal-A levels.

        Installation costs come from Facility.cost_installation (input_facilities.xlsx).
        Keys in cost_installation are strings; converted to int here.
        Levels beyond the Excel maximum plateau at the highest known cost.
        """
        cap = {}
        for i, facility in self.instance.facilities.items():
            # Parse cost_installation: keys are strings in the Excel JSON field
            fac_ci: dict[int, float] = {
                int(k): float(v)
                for k, v in facility.cost_installation.items()
            }
            levels = self.capacity_levels.get(i, sorted(k for k in fac_ci if k > 0))
            cap[i] = {0: (0, 0.0)}   # level 0 = closed, no cost
            for q in levels:
                cap[i][q] = (q, install_cost(q, fac_ci))
        return cap

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def build(self) -> None:
        logger.info("Building CapacitatedSAAModel")
        self._add_variables()
        self._add_objective()
        self._add_constraints()
        self.model.update()
        logger.info("Model built")

    def solve(self) -> dict:
        logger.info("Solving")
        t0 = time.time()
        self.model.optimize()
        logger.info("Solve complete")
        try:
            gap = round(100 * self.model.MIPGap, 3)
        except Exception:
            gap = 0.0
        self.results = {
            "actual_run_time":   round(time.time() - t0, 3),
            "optimality_gap":    gap,
            "objective_value":   round(self.obj.cost_total.getValue(), 3),
            "best_bound_value":  round(self.model.ObjBound, 3),
        }
        return self.results

    def set_params(self, params: dict) -> None:
        for k, v in params.items():
            self.model.setParam(k, v)

    # ------------------------------------------------------------------
    # Variable creation
    # ------------------------------------------------------------------
    def _add_variables(self) -> None:
        facilities = self.instance.facilities
        scenarios  = self.instance.scenarios
        vtype = GRB.CONTINUOUS if self.is_continuous_x else GRB.BINARY
        logger.info(f"Variable type X/W: {'continuous' if self.is_continuous_x else 'binary'}")

        self.vars.X = {
            (i, k, t, n): self.model.addVar(
                vtype=vtype, name=f"X_{i}_{k}_t{t}_n{n}", lb=0, ub=1
            )
            for i in facilities
            for n, sc in scenarios.items()
            for k in sc.pixels
            for t in range(self.instance.periods)
        }
        logger.info(f"X variables: {len(self.vars.X)}")

        self.vars.W = {
            (k, t, n): self.model.addVar(
                vtype=vtype, name=f"W_{k}_t{t}_n{n}", lb=0, ub=1
            )
            for n, sc in scenarios.items()
            for k in sc.pixels
            for t in range(self.instance.periods)
        }
        logger.info(f"W variables: {len(self.vars.W)}")

        self.vars.Y = {
            (i, q): self.model.addVar(vtype=GRB.BINARY, name=f"Y_{i}_q{q}")
            for i in facilities
            for q in self._cap[i]
        }
        logger.info(f"Y variables: {len(self.vars.Y)}")

        self.model._X = self.vars.X
        self.model._W = self.vars.W
        self.model._Y = self.vars.Y

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------
    def _add_objective(self) -> None:
        facilities = self.instance.facilities
        scenarios  = self.instance.scenarios
        N = len(scenarios)

        # Fixed installation cost (one-time, not per-scenario)
        self.obj.cost_installation = quicksum(
            cost * self.vars.Y[(i, q)]
            for i in facilities
            for q, (vehicles, cost) in self._cap[i].items()
            if vehicles > 0        # q=0 has cost=0, skip for clarity
        )

        # Second-echelon routing cost (facility → pixel)
        self.obj.cost_served_from_facilities = quicksum(
            round(sc.get_cost_serving("facility").get((i, k, "small", t, n), 0), 5)
            * self.vars.X[(i, k, t, n)]
            for i in facilities
            for n, sc in scenarios.items()
            for k in sc.pixels
            for t in range(self.instance.periods)
        )

        # DC direct routing cost (DC → pixel)
        self.obj.cost_served_from_dc = quicksum(
            round(sc.get_cost_serving("dc").get((k, "large", t, n), 0), 5)
            * self.vars.W[(k, t, n)]
            for n, sc in scenarios.items()
            for k in sc.pixels
            for t in range(self.instance.periods)
        )

        self.obj.cost_total = (
            self.obj.cost_installation
            + (1 / N) * (
                self.obj.cost_served_from_facilities
                + self.obj.cost_served_from_dc
            )
        )
        self.model._cost_total = self.obj.cost_total
        self.model.setObjective(self.obj.cost_total, GRB.MINIMIZE)

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------
    def _add_constraints(self) -> None:
        self._constr_one_level()
        self._constr_capacity()
        self._constr_demand()

    def _constr_one_level(self) -> None:
        """A.1 — Exactly one capacity level (including 0 = closed) per facility."""
        logger.info("Adding A.1 — one capacity level per facility")
        for i in self.instance.facilities:
            self.model.addConstr(
                quicksum(self.vars.Y[(i, q)] for q in self._cap[i]) == 1,
                name=f"R_open_{i}",
            )

    def _constr_capacity(self) -> None:
        """A.4 — Fleet assigned ≤ installed vehicle capacity."""
        logger.info("Adding A.4 — capacity constraints")
        facilities = self.instance.facilities
        scenarios  = self.instance.scenarios

        for t in range(self.instance.periods):
            for i in facilities:
                for n, sc in scenarios.items():
                    fleet = sc.get_fleet_size("facility")
                    self.model.addConstr(
                        quicksum(
                            self.vars.X[(i, k, t, n)]
                            * round(fleet.get((i, k, "small", t, n), 0), 1)
                            for k in sc.pixels
                        )
                        <= quicksum(
                            vehicles * self.vars.Y[(i, q)]
                            for q, (vehicles, _) in self._cap[i].items()
                        ),
                        name=f"R_cap_{i}_t{t}_n{n}",
                    )

    def _constr_demand(self) -> None:
        """A.5 — Each pixel-period-scenario must be (fractionally) covered."""
        logger.info("Adding A.5 — demand satisfaction")
        facilities = self.instance.facilities
        scenarios  = self.instance.scenarios

        for n, sc in scenarios.items():
            for t in range(self.instance.periods):
                for k in sc.pixels:
                    self.model.addConstr(
                        quicksum(self.vars.X[(i, k, t, n)] for i in facilities)
                        + self.vars.W[(k, t, n)]
                        >= 1,
                        name=f"R_demand_{k}_t{t}_n{n}",
                    )

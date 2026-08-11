"""Capacitated SAA Model with Flexible Operational Decisions.

Extends CapacitatedSAAModel with:
  - Z[i,q,t,n]: binary — satellite i operates at level q in period t, scenario n
  - Operational cost: cost_operation[i][q][t] × Z[i,q,t,n]
  - Flexibility: satellite can operate at installed level or LOWER (or not at all)

Design decisions flow:
  Y[i,q]=1 → installed at level q (design, one-time)
  Z[i,q',t,n]=1 → operating at level q' in period t, scenario n
                   where q' ≤ q (can't exceed installed level)
"""

import time
from dataclasses import dataclass, field
from typing import Dict

import gurobipy as gb
from gurobipy import GRB, quicksum  # pylint: disable=E0611

from src.utils.custom_logger import get_logger
from src.utils.instance import Instance
from src.utils.scenario import Scenario

logger = get_logger("CapacitatedFlexModel")


@dataclass
class VariablesModel:
    X: dict = field(default_factory=dict)   # X[i,j,t,n]: pixel assignment
    W: dict = field(default_factory=dict)   # W[j,t,n]: DC assignment
    Y: dict = field(default_factory=dict)   # Y[i,q]: installation (binary)
    Z: dict = field(default_factory=dict)   # Z[i,q,t,n]: operational level (binary)


@dataclass
class ObjectiveModel:
    cost_installation:         gb.LinExpr = None
    cost_operation:            gb.LinExpr = None
    cost_served_from_dc:       gb.LinExpr = None
    cost_served_from_facilities: gb.LinExpr = None
    cost_total:                gb.LinExpr = None


class CapacitatedFlexModel:
    """Capacitated SAA with flex operational level per period/scenario."""

    def __init__(
        self,
        instance: Instance,
        capacity_levels:  dict[str, list[int]],
        cost_operation:   dict[str, dict[int, list[float]]],
    ):
        """
        Parameters
        ----------
        instance         : loaded Instance with CA already run
        capacity_levels  : {satellite: [level1, level2, ...]} — Proposal-A levels
        cost_operation   : {satellite: {level: [12 monthly costs]}}
                           level 0 must be present with all-zero costs
        """
        self.model     = gb.Model(name="CapacitatedFlex")
        self.instance  = instance
        self.is_cont_x = instance.config.is_continuous_var_x

        self._cap      = self._build_cap(capacity_levels)
        self._cost_op  = cost_operation   # {sat: {q: [t0,t1,...,t11]}}

        self.vars  = VariablesModel()
        self.obj   = ObjectiveModel()
        self.results: dict = {}

    # ------------------------------------------------------------------
    def _build_cap(self, levels: dict) -> dict[str, dict[int, tuple[int, float]]]:
        """Build {sat: {level: (vehicles, install_cost)}} from Excel data."""
        cap = {}
        for i, fac in self.instance.facilities.items():
            fac_ci = {int(k): float(v) for k, v in fac.cost_installation.items()}
            plateau = float(max(fac_ci.values())) if fac_ci else 45_417.0
            sat_levels = levels.get(i, sorted(k for k in fac_ci if k > 0))
            cap[i] = {0: (0, 0.0)}
            for q in sat_levels:
                cap[i][q] = (q, fac_ci.get(q, plateau))
        return cap

    # ------------------------------------------------------------------
    def build(self) -> None:
        logger.info("Building CapacitatedFlexModel")
        self._add_variables()
        self._add_objective()
        self._add_constraints()
        self.model.update()
        logger.info("Model built")

    def solve(self) -> dict:
        logger.info("Solving")
        t0 = time.time()
        self.model.optimize()
        try:
            gap = round(100 * self.model.MIPGap, 3)
        except Exception:
            gap = 0.0
        self.results = {
            "actual_run_time":  round(time.time() - t0, 3),
            "optimality_gap":   gap,
            "objective_value":  round(self.obj.cost_total.getValue(), 3),
            "best_bound_value": round(self.model.ObjBound, 3),
        }
        return self.results

    def set_params(self, params: dict) -> None:
        for k, v in params.items():
            self.model.setParam(k, v)

    # ------------------------------------------------------------------
    def _add_variables(self) -> None:
        facs = self.instance.facilities
        scens = self.instance.scenarios
        vtype = GRB.CONTINUOUS if self.is_cont_x else GRB.BINARY
        logger.info(f"X/W type: {'continuous' if self.is_cont_x else 'binary'}")

        # X[i,j,t,n]
        self.vars.X = {
            (i, k, t, n): self.model.addVar(vtype=vtype, lb=0, ub=1,
                                             name=f"X_{i}_{k}_t{t}_n{n}")
            for i in facs
            for n, sc in scens.items()
            for k in sc.pixels
            for t in range(self.instance.periods)
        }
        logger.info(f"X variables: {len(self.vars.X)}")

        # W[j,t,n]
        self.vars.W = {
            (k, t, n): self.model.addVar(vtype=vtype, lb=0, ub=1,
                                          name=f"W_{k}_t{t}_n{n}")
            for n, sc in scens.items()
            for k in sc.pixels
            for t in range(self.instance.periods)
        }
        logger.info(f"W variables: {len(self.vars.W)}")

        # Y[i,q]
        self.vars.Y = {
            (i, q): self.model.addVar(vtype=GRB.BINARY, name=f"Y_{i}_q{q}")
            for i in facs
            for q in self._cap[i]
        }
        logger.info(f"Y variables: {len(self.vars.Y)}")

        # Z[i,q,t,n]
        self.vars.Z = {
            (i, q, t, n): self.model.addVar(vtype=GRB.BINARY,
                                             name=f"Z_{i}_q{q}_t{t}_n{n}")
            for i in facs
            for q in self._cap[i]
            for t in range(self.instance.periods)
            for n in scens
        }
        logger.info(f"Z variables: {len(self.vars.Z)}")

        self.model._X = self.vars.X
        self.model._W = self.vars.W
        self.model._Y = self.vars.Y
        self.model._Z = self.vars.Z

    # ------------------------------------------------------------------
    def _add_objective(self) -> None:
        facs = self.instance.facilities
        scens = self.instance.scenarios
        N = len(scens)

        # Installation (one-time fixed cost)
        self.obj.cost_installation = quicksum(
            cost * self.vars.Y[(i, q)]
            for i in facs
            for q, (veh, cost) in self._cap[i].items()
            if veh > 0
        )

        # Operational cost: paid each period when satellite is operating
        self.obj.cost_operation = quicksum(
            self._cost_op.get(i, {}).get(q, [0]*12)[t] * self.vars.Z[(i, q, t, n)]
            for i in facs
            for q in self._cap[i]
            for t in range(self.instance.periods)
            for n in scens
            if self._cost_op.get(i, {}).get(q, [0]*12)[t] > 0
        )

        # Routing costs
        self.obj.cost_served_from_facilities = quicksum(
            round(sc.get_cost_serving("facility").get((i, k, "small", t, n), 0), 5)
            * self.vars.X[(i, k, t, n)]
            for i in facs
            for n, sc in scens.items()
            for k in sc.pixels
            for t in range(self.instance.periods)
        )
        self.obj.cost_served_from_dc = quicksum(
            round(sc.get_cost_serving("dc").get((k, "large", t, n), 0), 5)
            * self.vars.W[(k, t, n)]
            for n, sc in scens.items()
            for k in sc.pixels
            for t in range(self.instance.periods)
        )

        self.obj.cost_total = (
            self.obj.cost_installation
            + (1 / N) * (
                self.obj.cost_operation
                + self.obj.cost_served_from_facilities
                + self.obj.cost_served_from_dc
            )
        )
        self.model._cost_total = self.obj.cost_total
        self.model.setObjective(self.obj.cost_total, GRB.MINIMIZE)

    # ------------------------------------------------------------------
    def _add_constraints(self) -> None:
        self._constr_one_install_level()
        self._constr_one_op_level()
        self._constr_op_le_install()
        self._constr_capacity()
        self._constr_demand()

    def _constr_one_install_level(self) -> None:
        """A.1 — Exactly one installation level per satellite."""
        logger.info("A.1 — one installation level")
        for i in self.instance.facilities:
            self.model.addConstr(
                quicksum(self.vars.Y[(i, q)] for q in self._cap[i]) == 1,
                name=f"R_install_{i}",
            )

    def _constr_one_op_level(self) -> None:
        """A.2 — Exactly one operational level per satellite per period/scenario."""
        logger.info("A.2 — one operational level per period/scenario")
        for i in self.instance.facilities:
            for n in self.instance.scenarios:
                for t in range(self.instance.periods):
                    self.model.addConstr(
                        quicksum(self.vars.Z[(i, q, t, n)] for q in self._cap[i]) == 1,
                        name=f"R_op_{i}_t{t}_n{n}",
                    )

    def _constr_op_le_install(self) -> None:
        """A.3 — Can't operate above installed level.
        If Y[i,q]=1 (installed at q), then Z[i,q',t,n]=0 for all q' with vehicles(q') > vehicles(q).
        """
        logger.info("A.3 — operational level ≤ installed level")
        for i in self.instance.facilities:
            max_veh = max(veh for veh, _ in self._cap[i].values())
            for q_inst, (veh_inst, _) in self._cap[i].items():
                if veh_inst >= max_veh:
                    continue  # max level — no higher levels to restrict
                higher_qs = [q for q, (veh, _) in self._cap[i].items() if veh > veh_inst]
                for n in self.instance.scenarios:
                    for t in range(self.instance.periods):
                        self.model.addConstr(
                            quicksum(self.vars.Z[(i, q, t, n)] for q in higher_qs)
                            <= 1 - self.vars.Y[(i, q_inst)],
                            name=f"R_op_le_inst_{i}_q{q_inst}_t{t}_n{n}",
                        )

    def _constr_capacity(self) -> None:
        """A.4 — Fleet used ≤ operational capacity in each period/scenario."""
        logger.info("A.4 — capacity constraints")
        for t in range(self.instance.periods):
            for i in self.instance.facilities:
                for n, sc in self.instance.scenarios.items():
                    fleet = sc.get_fleet_size("facility")
                    self.model.addConstr(
                        quicksum(
                            self.vars.X[(i, k, t, n)]
                            * round(fleet.get((i, k, "small", t, n), 0), 1)
                            for k in sc.pixels
                        )
                        <= quicksum(
                            veh * self.vars.Z[(i, q, t, n)]
                            for q, (veh, _) in self._cap[i].items()
                        ),
                        name=f"R_cap_{i}_t{t}_n{n}",
                    )

    def _constr_demand(self) -> None:
        """A.5 — Demand satisfaction."""
        logger.info("A.5 — demand satisfaction")
        for n, sc in self.instance.scenarios.items():
            for t in range(self.instance.periods):
                for k in sc.pixels:
                    self.model.addConstr(
                        quicksum(self.vars.X[(i, k, t, n)]
                                 for i in self.instance.facilities)
                        + self.vars.W[(k, t, n)] >= 1,
                        name=f"R_dem_{k}_t{t}_n{n}",
                    )

"""Base two-echelon SAA model: the shared formulation plus a registry of toggleable blocks.

The four formulations of the study nest strictly — uncapacitated ⊂ capacitated ⊂ flex,
with extended sharing flex's variable set — and roughly 70% of their code was identical:
the assignment variables `X` and `W`, both routing terms of the objective, the demand
constraint, `solve()` and `set_params()`. All of that lives here once.

A variant is a thin subclass that declares `DEFAULT_FEATURES` and implements the optional
blocks it turns on. Nothing else.

Block catalogue
---------------
Variables               flag              added by
    assignment          (always)          base — X[i,k,t,n], W[k,t,n]
    install             install_levels    capacitated — Y[i,q]
    operation           flex_operation    flex — Z[i,q,t,n]

Objective               flag              averaged by 1/N
    routing_facilities  (always)          yes
    routing_dc          (always)          yes
    installation        install_levels    no   <- one-time cost, must NOT be averaged
    operation           flex_operation    yes

Constraints             flag
    demand              (always)          A.5, every pixel-period served
    one_install_level   install_levels
    one_operation_level flex_operation
    operation_le_install flex_operation
    capacity            capacity

Two notes that matter more than the machinery:

* `installation` is the only objective block that is *not* divided by the number of
  scenarios. That is why the cost components in a result JSON do not sum to `objective`.
  The registry carries `averaged` per block precisely so a new variant cannot get this
  wrong by omission.
* `continuous_x` is deliberately *not* a feature flag. It is read from
  `instance.config.is_continuous_var_x`, so there is exactly one place that decides
  whether X and W are continuous or binary.

Ablation without a new class
----------------------------
    from dataclasses import replace
    model = FlexSAAModel(instance, features=replace(FlexSAAModel.DEFAULT_FEATURES,
                                                    disabled_blocks=frozenset({"capacity"})))

`disabled_blocks` names blocks, not flags, so a single constraint can be dropped while
its variables stay. Unknown names raise instead of being ignored.
"""

import time
from dataclasses import asdict, dataclass, field
from typing import Dict, NamedTuple, Optional

import gurobipy as gb
from gurobipy import GRB, quicksum  # pylint: disable=E0611

from src.core.logging import get_logger
from src.optimization.instance import Instance

logger = get_logger("SAAModel")

STATUS_NAMES = {
    GRB.LOADED: "LOADED",
    GRB.OPTIMAL: "OPTIMAL",
    GRB.INFEASIBLE: "INFEASIBLE",
    GRB.INF_OR_UNBD: "INF_OR_UNBD",
    GRB.UNBOUNDED: "UNBOUNDED",
    GRB.CUTOFF: "CUTOFF",
    GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
    GRB.NODE_LIMIT: "NODE_LIMIT",
    GRB.TIME_LIMIT: "TIME_LIMIT",
    GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
    GRB.INTERRUPTED: "INTERRUPTED",
    GRB.NUMERIC: "NUMERIC",
    GRB.SUBOPTIMAL: "SUBOPTIMAL",
}


@dataclass(frozen=True)
class ModelFeatures:
    """Which optional blocks a model builds.

    `disabled_blocks` is the ablation escape hatch: it names blocks from the catalogue
    in `BaseSAAModel`, overriding the flags above.
    """

    install_levels: bool = False
    flex_operation: bool = False
    capacity: bool = False
    disabled_blocks: frozenset = frozenset()

    def as_dict(self) -> dict:
        """Serializable form, for recording alongside the results."""
        out = asdict(self)
        out["disabled_blocks"] = sorted(self.disabled_blocks)
        return out


class Block(NamedTuple):
    """One buildable piece of the model."""

    name: str
    method: str
    flag: Optional[str]  # None means always on
    averaged: bool = False  # objective blocks: divided by the number of scenarios
    field: Optional[str] = None  # objective blocks: where to store the expression


@dataclass
class VariablesModel:
    """Decision variables, named as in the formulation."""

    # X[i,k,t,n]: pixel k is served from facility i in period t in scenario n
    X: Dict[tuple, gb.Var] = field(default_factory=dict)
    # W[k,t,n]: pixel k is served from the DC in period t in scenario n
    W: Dict[tuple, gb.Var] = field(default_factory=dict)
    # Y[i,q]: facility i is installed at capacity level q
    Y: Dict[tuple, gb.Var] = field(default_factory=dict)
    # Z[i,q,t,n]: facility i operates at capacity level q in period t in scenario n
    Z: Dict[tuple, gb.Var] = field(default_factory=dict)


@dataclass
class ObjectiveModel:
    """Objective components. Unused ones stay `None`."""

    cost_served_from_facilities: gb.LinExpr = None
    cost_served_from_dc: gb.LinExpr = None
    cost_installation: gb.LinExpr = None
    cost_operation: gb.LinExpr = None
    cost_total: gb.LinExpr = None


class BaseSAAModel:
    """Shared skeleton of every two-echelon SAA formulation."""

    NAME = "base"
    DEFAULT_FEATURES = ModelFeatures()

    VAR_BLOCKS = (
        Block("assignment", "_vars_assignment", None),
        Block("install", "_vars_install", "install_levels"),
        Block("operation", "_vars_operation", "flex_operation"),
    )
    OBJ_BLOCKS = (
        Block("routing_facilities", "_obj_routing_facilities", None, averaged=True, field="cost_served_from_facilities"),
        Block("routing_dc", "_obj_routing_dc", None, averaged=True, field="cost_served_from_dc"),
        Block("installation", "_obj_installation", "install_levels", averaged=False, field="cost_installation"),
        Block("operation", "_obj_operation", "flex_operation", averaged=True, field="cost_operation"),
    )
    CONSTR_BLOCKS = (
        Block("demand", "_constr_demand", None),
        Block("one_install_level", "_constr_one_install_level", "install_levels"),
        Block("one_operation_level", "_constr_one_operation_level", "flex_operation"),
        Block("operation_le_install", "_constr_operation_le_install", "flex_operation"),
        Block("capacity", "_constr_capacity", "capacity"),
    )

    def __init__(self, instance: Instance, features: Optional[ModelFeatures] = None):
        self.model = gb.Model(name=self.NAME)
        self.instance = instance
        self.features = features if features is not None else self.DEFAULT_FEATURES
        # Validated before anything is read off the instance, so an incoherent feature
        # set reports itself instead of surfacing as an error further down.
        self._validate_features()

        # Single source of truth: the instance decides the variable domain, not a flag.
        self.is_continuous_x: bool = instance.config.is_continuous_var_x

        self.vars = VariablesModel()
        self.obj = ObjectiveModel()
        self.results: dict = {}

    # ── block resolution ──────────────────────────────────────────────────────

    @classmethod
    def all_block_names(cls) -> frozenset:
        """Every block name in the catalogue, for validating `disabled_blocks`."""
        return frozenset(b.name for group in (cls.VAR_BLOCKS, cls.OBJ_BLOCKS, cls.CONSTR_BLOCKS) for b in group)

    def _validate_features(self) -> None:
        """Reject feature sets that would build a silently wrong model."""
        unknown = set(self.features.disabled_blocks) - set(self.all_block_names())
        if unknown:
            raise ValueError(
                f"disabled_blocks names blocks that do not exist: {sorted(unknown)}. "
                f"Known blocks: {sorted(self.all_block_names())}"
            )
        if self.features.capacity and not self.features.install_levels:
            raise ValueError("capacity=True requires install_levels=True: the capacity constraint is written over Y/Z levels.")
        if self.features.flex_operation and not self.features.install_levels:
            raise ValueError("flex_operation=True requires install_levels=True: Z is bounded by the installed level Y.")

    def _is_enabled(self, block: Block) -> bool:
        if block.name in self.features.disabled_blocks:
            return False
        if block.flag is None:
            return True
        return bool(getattr(self.features, block.flag))

    def _run_blocks(self, blocks) -> list:
        """Run the enabled blocks in catalogue order and return the ones that ran."""
        executed = []
        for block in blocks:
            if not self._is_enabled(block):
                continue
            method = getattr(self, block.method, None)
            if method is None:
                raise NotImplementedError(
                    f"{type(self).__name__} enables block '{block.name}' but does not implement {block.method}()"
                )
            executed.append((block, method()))
        return executed

    # ── build ─────────────────────────────────────────────────────────────────

    def build(self) -> None:
        """Add variables, objective and constraints for the enabled blocks."""
        logger.info(f"Building {self.NAME} | features={self.features.as_dict()}")
        self._add_variables()
        self._add_objective()
        self._add_constraints()
        self.model.update()
        logger.info(f"Model built: {self.model.NumVars} variables, {self.model.NumConstrs} constraints")

    def _add_variables(self) -> None:
        logger.info(f"Using {'continuous' if self.is_continuous_x else 'binary'} variables for X[i,k,t,n] and W[k,t,n]")
        self._run_blocks(self.VAR_BLOCKS)
        # Exposed on the Gurobi model so downstream readers can recover the solution.
        self.model._X = self.vars.X  # pylint: disable=protected-access
        self.model._W = self.vars.W  # pylint: disable=protected-access
        self.model._Y = self.vars.Y  # pylint: disable=protected-access
        self.model._Z = self.vars.Z  # pylint: disable=protected-access

    def _add_objective(self) -> None:
        n_scenarios = len(self.instance.scenarios)
        averaged, deterministic = [], []
        for block, expr in self._run_blocks(self.OBJ_BLOCKS):
            setattr(self.obj, block.field, expr)
            (averaged if block.averaged else deterministic).append(expr)

        total = None
        if averaged:
            scenario_dependent = averaged[0]
            for expr in averaged[1:]:
                scenario_dependent = scenario_dependent + expr
            total = (self.instance.horizon_weight / n_scenarios) * scenario_dependent
        for expr in deterministic:
            total = expr if total is None else expr + total

        self.obj.cost_total = total
        self.model._cost_total = total  # pylint: disable=protected-access
        self.model.setObjective(total, GRB.MINIMIZE)

    def _add_constraints(self) -> None:
        self._run_blocks(self.CONSTR_BLOCKS)

    # ── always-on blocks ──────────────────────────────────────────────────────

    def _vars_assignment(self) -> None:
        """X[i,k,t,n] and W[k,t,n], the pixel assignment variables."""
        facilities = self.instance.facilities
        scenarios = self.instance.scenarios
        vtype = GRB.CONTINUOUS if self.is_continuous_x else GRB.BINARY

        self.vars.X = {
            (i, k, t, n): self.model.addVar(vtype=vtype, name=f"X_i{i}_k{k}_t{t}_n{n}", lb=0, ub=1)
            for i in facilities.keys()
            for n, scenario in scenarios.items()
            for k in scenario.pixels.keys()
            for t in range(self.instance.periods)
        }
        logger.info(f"Number of variables X: {len(self.vars.X)}")

        self.vars.W = {
            (k, t, n): self.model.addVar(vtype=vtype, name=f"W_k{k}_t{t}_n{n}", lb=0, ub=1)
            for n, scenario in scenarios.items()
            for k in scenario.pixels.keys()
            for t in range(self.instance.periods)
        }
        logger.info(f"Number of variables W: {len(self.vars.W)}")

    def _obj_routing_facilities(self) -> gb.LinExpr:
        """Second-echelon routing cost: satellites serving pixels with vans.

        Indexed directly, not with `.get(key, 0)`. The CA only writes cost keys for
        `demand > 0`, so a missing key is a broken scenario and must raise here rather
        than be priced at zero.
        """
        facilities = self.instance.facilities
        scenarios = self.instance.scenarios
        return quicksum(
            [
                round(scenario.get_cost_serving("facility")[(i, k, "small", t, n)], 5) * self.vars.X[(i, k, t, n)]
                for i in facilities.keys()
                for n, scenario in scenarios.items()
                for k in scenario.pixels.keys()
                for t in range(self.instance.periods)
            ]
        )

    def _obj_routing_dc(self) -> gb.LinExpr:
        """First-echelon routing cost: the DC serving pixels directly with large trucks."""
        scenarios = self.instance.scenarios
        return quicksum(
            [
                round(scenario.get_cost_serving("dc")[(k, "large", t, n)], 5) * self.vars.W[(k, t, n)]
                for n, scenario in scenarios.items()
                for k in scenario.pixels.keys()
                for t in range(self.instance.periods)
            ]
        )

    def _constr_demand(self) -> None:
        """A.5 — every pixel-period is served, from some facility or from the DC."""
        logger.info("--------- Adding constraints A.5 - Demand satisfaction")
        facilities = self.instance.facilities
        for n, scenario in self.instance.scenarios.items():
            for t in range(self.instance.periods):
                for k in scenario.pixels.keys():
                    self.model.addConstr(
                        quicksum([self.vars.X[(i, k, t, n)] for i in facilities.keys()]) + self.vars.W[(k, t, n)] >= 1,
                        name=f"R_demand_k{k}_t{t}_n{n}",
                    )

    # ── optional blocks: implemented by the variants ──────────────────────────
    # Each is declared here so the catalogue is complete and a missing implementation
    # fails loudly at build time. Reference implementations live in OLD/src/models/.

    def _vars_install(self) -> None:
        raise NotImplementedError("Y[i,q] belongs to the capacitated variant (see OLD/src/models/capacitated_saa_model.py)")

    def _vars_operation(self) -> None:
        raise NotImplementedError("Z[i,q,t,n] belongs to the flex variant (see OLD/src/models/capacitated_flex_model.py)")

    def _obj_installation(self) -> gb.LinExpr:
        raise NotImplementedError("installation cost belongs to the capacitated variant")

    def _obj_operation(self) -> gb.LinExpr:
        raise NotImplementedError("operation cost belongs to the flex variant")

    def _constr_one_install_level(self) -> None:
        raise NotImplementedError("one installed level per facility belongs to the capacitated variant")

    def _constr_one_operation_level(self) -> None:
        raise NotImplementedError("one operating level per facility-period belongs to the flex variant")

    def _constr_operation_le_install(self) -> None:
        raise NotImplementedError("Z <= Y belongs to the flex variant")

    def _constr_capacity(self) -> None:
        raise NotImplementedError("the capacity constraint belongs to the capacitated variant")

    # ── solve ─────────────────────────────────────────────────────────────────

    def set_params(self, params: Dict[str, float]) -> None:
        """Forward Gurobi parameters to the underlying model."""
        logger.info(f"Setting parameters: {params}")
        for key, item in params.items():
            self.model.setParam(key, item)

    def solve(self) -> dict:
        """Optimize and record the outcome, `Status` included.

        The pre-refactor models recorded an objective without inspecting `Status`, so a
        time-limit incumbent was indistinguishable from an optimum in the result JSON.
        Here `status` and `is_optimal` always travel with the objective, and a run that
        ended with no feasible solution records `objective_value = None` instead of
        raising deep inside a getter.
        """
        logger.info("Solving model")
        start_time = time.time()
        self.model.optimize()
        run_time = round(time.time() - start_time, 3)

        status = self.model.Status
        status_name = STATUS_NAMES.get(status, str(status))
        has_solution = self.model.SolCount > 0
        logger.info(f"Solve finished: status={status_name}, solutions={self.model.SolCount}")
        if not has_solution:
            logger.error(f"No feasible solution recorded (status={status_name}); the objective is not reportable.")
        elif status != GRB.OPTIMAL:
            logger.warning(f"Objective comes from a non-optimal solution (status={status_name}).")

        try:
            mip_gap = round(100 * self.model.MIPGap, 3)
        except Exception:  # noqa: BLE001 - an LP has no MIPGap
            mip_gap = 0.0
        try:
            best_bound = round(self.model.ObjBound, 3)
        except Exception:  # noqa: BLE001 - undefined for some statuses
            best_bound = None

        self.results = {
            "actual_run_time": run_time,
            "optimality_gap": mip_gap,
            "objective_value": round(self.obj.cost_total.getValue(), 3) if has_solution else None,
            "best_bound_value": best_bound,
            "status": status_name,
            "is_optimal": status == GRB.OPTIMAL,
            "model": self.NAME,
            "features": self.features.as_dict(),
        }
        return self.results


__all__ = ["BaseSAAModel", "ModelFeatures", "Block", "VariablesModel", "ObjectiveModel", "STATUS_NAMES"]

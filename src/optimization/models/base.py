"""Base two-echelon SAA model: the shared formulation plus a registry of toggleable blocks.

The formulations of the study nest strictly — uncapacitated ⊂ capacitated ⊂ flex — so
each variant is a subclass of the previous one that *adds* blocks and never restates
them:

    BaseSAAModel            X[i,k,t,n], W[k,t,n], both routing terms, demand (A.5)
    └ UncapacitatedSAAModel  nothing more: the lower bound of the family
      └ CapacitatedSAAModel  Y[i,q], installation cost, one level, capacity on Y, fixed Y
        └ FlexSAAModel       Z[i,q,t,n], operation cost, one level, policy Z~Y, capacity on Z

How a variant extends the family
--------------------------------
* `BLOCKS` lists only the blocks the class introduces. `__init_subclass__` appends them
  to the parent's catalogue (or inserts them `before=` a named block, which fixes the
  order Gurobi sees; with a work or time limit that order changes the result).
* Redefining a block's method changes its formulation: `FlexSAAModel._constr_capacity`
  replaces the capacity on Y by the capacity on Z without touching the catalogue.
* `Features` is the variant's dataclass of flags (a subclass of the parent's), and
  `DEFAULT_FEATURES` its instance. A block with a `flag` runs only if that flag is on;
  one with `enabled_if` only if that model property is truthy.
* Setting `NAME` registers the class in `MODELS`, so runs select it by name
  (`optimize flexibility --model capacitated`).

Two properties the registry enforces, because both used to be silent errors:

* each objective block declares whether it is `averaged` — divided by `N` and weighted by
  `horizon_weight` — and installation, the one-time cost, is not;
* `solve()` records `Status` and `is_optimal` next to the objective.

One source of cost truth
------------------------
An averaged objective block returns one expression per scenario. The objective is their
weighted sum, and `scenario_costs()` evaluates the very same expressions, so the
per-scenario costs of an evaluation cannot drift from the objective.

Ablation without a new class
----------------------------
    from dataclasses import replace
    model = FlexSAAModel(instance, features=replace(FlexSAAModel.DEFAULT_FEATURES,
                                                    disabled_blocks=frozenset({"capacity"})))

`disabled_blocks` names blocks, not flags, so a single constraint can be dropped while
its variables stay. Unknown names raise instead of being ignored.

`continuous_x` is deliberately *not* a flag: `instance.config.is_continuous_var_x` is the
one place that decides whether X and W are continuous or binary.
"""

import time
from dataclasses import asdict, dataclass, field
from typing import ClassVar, Dict, NamedTuple, Optional

import gurobipy as gb
from gurobipy import GRB, quicksum  # pylint: disable=E0611

from src.optimization.instance import Instance
from src.tools.logging import get_logger

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

# Every concrete model, by `NAME`; filled by `BaseSAAModel.__init_subclass__`.
MODELS: dict[str, type["BaseSAAModel"]] = {}


@dataclass(frozen=True)
class ModelFeatures:
    """Flags of a model; each variant subclasses this with the flags of its own blocks.

    `disabled_blocks` is the ablation escape hatch: it names blocks of the catalogue,
    overriding the flags.
    """

    disabled_blocks: frozenset = frozenset()

    def as_dict(self) -> dict:
        """Serializable form, for recording alongside the results."""
        out = asdict(self)
        out["disabled_blocks"] = sorted(self.disabled_blocks)
        return out


class Block(NamedTuple):
    """One buildable piece of the model."""

    name: str  # unique across the catalogue: `disabled_blocks` refers to it
    kind: str  # "var" | "obj" | "constr"
    method: str
    flag: Optional[str] = None  # feature that switches it on; None = always on
    averaged: bool = False  # objective blocks: per-scenario, weighted by horizon_weight / N
    field: Optional[str] = None  # objective blocks: `ObjectiveModel` attribute for the total
    cost_key: Optional[str] = None  # objective blocks: column in `scenario_costs()`
    enabled_if: Optional[str] = None  # model property that must be truthy
    before: Optional[str] = None  # insert before this block in the parent's catalogue
    solution: Optional[tuple[str, str]] = None  # var blocks: (decisions key, extractor method)


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
    # averaged block name -> {scenario id -> expression}
    by_scenario: Dict[str, Dict[str, gb.LinExpr]] = field(default_factory=dict)


class BaseSAAModel:
    """Shared skeleton of every two-echelon SAA formulation."""

    NAME: ClassVar[str] = "base"
    Features: ClassVar[type] = ModelFeatures
    DEFAULT_FEATURES: ClassVar[ModelFeatures] = ModelFeatures()
    CATALOGUE: ClassVar[tuple[Block, ...]] = (
        Block("assignment", "var", "_vars_assignment"),
        Block(
            "routing_facilities",
            "obj",
            "_obj_routing_facilities",
            averaged=True,
            field="cost_served_from_facilities",
            cost_key="routing_facilities_cost",
        ),
        Block("routing_dc", "obj", "_obj_routing_dc", averaged=True, field="cost_served_from_dc", cost_key="routing_dc_cost"),
        Block("demand", "constr", "_constr_demand"),
    )
    BLOCKS: ClassVar[tuple[Block, ...]] = ()
    # Whether the model takes an operating policy (`type_of_flexibility`); experiments
    # sweep policies only for models that do.
    USES_POLICY: ClassVar[bool] = False
    # Whether the model reads satellite levels and costs, i.e. needs a facilities artifact.
    USES_CAPACITY: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        catalogue = list(cls.__mro__[1].CATALOGUE)
        for block in cls.__dict__.get("BLOCKS", ()):
            names = [item.name for item in catalogue]
            if block.name in names:
                raise TypeError(f"{cls.__name__} redeclares block '{block.name}'; override its method instead.")
            if block.before is None:
                catalogue.append(block)
            else:
                if block.before not in names:
                    raise TypeError(f"{cls.__name__}: block '{block.name}' is placed before unknown block '{block.before}'.")
                catalogue.insert(names.index(block.before), block)
        cls.CATALOGUE = tuple(catalogue)
        if "NAME" in cls.__dict__:
            MODELS[cls.NAME] = cls

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
        return frozenset(block.name for block in cls.CATALOGUE)

    @classmethod
    def blocks(cls, kind: str) -> tuple[Block, ...]:
        return tuple(block for block in cls.CATALOGUE if block.kind == kind)

    def _validate_features(self) -> None:
        """Reject feature sets that would build a silently wrong model.

        Variants extend this with the coherence rules of their own flags.
        """
        if not isinstance(self.features, self.Features):
            raise TypeError(f"{type(self).__name__} expects {self.Features.__name__}, got {type(self.features).__name__}.")
        unknown = set(self.features.disabled_blocks) - set(self.all_block_names())
        if unknown:
            raise ValueError(
                f"disabled_blocks names blocks that do not exist: {sorted(unknown)}. "
                f"Known blocks: {sorted(self.all_block_names())}"
            )

    def _is_enabled(self, block: Block) -> bool:
        if block.name in self.features.disabled_blocks:
            return False
        if block.flag is not None and not getattr(self.features, block.flag):
            return False
        return block.enabled_if is None or bool(getattr(self, block.enabled_if))

    def _run_blocks(self, kind: str) -> list:
        """Run the enabled blocks of one kind in catalogue order; return (block, result) pairs."""
        executed = []
        for block in self.blocks(kind):
            if self._is_enabled(block):
                executed.append((block, getattr(self, block.method)()))
        return executed

    # ── build ─────────────────────────────────────────────────────────────────

    def build(self) -> None:
        """Add variables, objective and constraints for the enabled blocks."""
        logger.info(f"Building {self.NAME} | features={self.features.as_dict()}")
        self._add_variables()
        self._add_objective()
        self._run_blocks("constr")
        self.model.update()
        logger.info(f"Model built: {self.model.NumVars} variables, {self.model.NumConstrs} constraints")

    def _add_variables(self) -> None:
        logger.info(f"Using {'continuous' if self.is_continuous_x else 'binary'} variables for X[i,k,t,n] and W[k,t,n]")
        self._run_blocks("var")

    def _add_objective(self) -> None:
        n_scenarios = len(self.instance.scenarios)
        averaged, deterministic = [], []
        for block, value in self._run_blocks("obj"):
            if block.averaged:
                self.obj.by_scenario[block.name] = value
                value = quicksum(value.values())
            setattr(self.obj, block.field, value)
            (averaged if block.averaged else deterministic).append(value)

        total = None
        if averaged:
            scenario_dependent = averaged[0]
            for expr in averaged[1:]:
                scenario_dependent = scenario_dependent + expr
            total = (self.instance.horizon_weight / n_scenarios) * scenario_dependent
        for expr in deterministic:
            total = expr if total is None else expr + total

        self.obj.cost_total = total
        self.model.setObjective(total, GRB.MINIMIZE)

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

    def _obj_routing_facilities(self) -> dict[str, gb.LinExpr]:
        """Second-echelon routing cost per scenario: satellites serving pixels with vans.

        Indexed directly, not with `.get(key, 0)`. The CA only writes cost keys for
        `demand > 0`, so a missing key is a broken scenario and must raise here rather
        than be priced at zero.
        """
        return {
            n: quicksum(
                [
                    round(scenario.get_cost_serving("facility")[(i, k, "small", t, n)], 5) * self.vars.X[(i, k, t, n)]
                    for i in self.instance.facilities.keys()
                    for k in scenario.pixels.keys()
                    for t in range(self.instance.periods)
                ]
            )
            for n, scenario in self.instance.scenarios.items()
        }

    def _obj_routing_dc(self) -> dict[str, gb.LinExpr]:
        """First-echelon routing cost per scenario: the DC serving pixels directly with large trucks."""
        return {
            n: quicksum(
                [
                    round(scenario.get_cost_serving("dc")[(k, "large", t, n)], 5) * self.vars.W[(k, t, n)]
                    for k in scenario.pixels.keys()
                    for t in range(self.instance.periods)
                ]
            )
            for n, scenario in self.instance.scenarios.items()
        }

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

    # ── solve ─────────────────────────────────────────────────────────────────

    def set_params(self, params: Dict[str, float]) -> None:
        """Forward Gurobi parameters to the underlying model."""
        logger.info(f"Setting parameters: {params}")
        for key, item in params.items():
            self.model.setParam(key, item)

    def solve(self) -> dict:
        """Optimize and record the outcome, `Status` included.

        `status` and `is_optimal` always travel with the objective, and a run that ended
        with no feasible solution records `objective_value = None` instead of raising
        deep inside a getter.
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

    # ── reading a solution ────────────────────────────────────────────────────

    def decisions(self) -> dict:
        """Serializable first- and second-stage decisions, one entry per variable block that defines one."""
        return {
            key: getattr(self, extractor)()
            for block in self.blocks("var")
            if block.solution is not None and self._is_enabled(block)
            for key, extractor in (block.solution,)
        }

    def scenario_costs(self) -> list[dict]:
        """Unaveraged cost of every scenario, from the objective's own expressions.

        First-stage (non-averaged) cost is reported once per row but counted once, not
        per scenario: `installation + mean(second_stage)` is the objective. Second-stage
        terms carry `horizon_weight`, like the objective.
        """
        weight = self.instance.horizon_weight
        first_stage = sum(value.getValue() for block, value in self._objective_parts() if not block.averaged)
        averaged = [block for block, _ in self._objective_parts() if block.averaged]
        rows = []
        for scenario_id in self.instance.scenarios:
            components = {block.cost_key: weight * self.obj.by_scenario[block.name][scenario_id].getValue() for block in averaged}
            second_stage = sum(components.values())
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "installation_cost": round(first_stage, 3),
                    **{key: round(value, 3) for key, value in components.items()},
                    "second_stage_cost": round(second_stage, 3),
                    "total_cost": round(first_stage + second_stage, 3),
                }
            )
        return rows

    def _objective_parts(self) -> list[tuple[Block, gb.LinExpr]]:
        return [
            (block, getattr(self.obj, block.field)) for block in self.blocks("obj") if getattr(self.obj, block.field) is not None
        ]


__all__ = ["BaseSAAModel", "ModelFeatures", "Block", "VariablesModel", "ObjectiveModel", "STATUS_NAMES", "MODELS"]

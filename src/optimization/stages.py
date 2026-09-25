"""The versioned stages of the optimization side: satellite capacity tables and runs.

    facilities  cf-* -> f<N>   capacity.json (levels + costs per satellite), peak_fleet.csv, assignment.json
    runs        cr-* -> r<N>   one experiment over one scenario version (and one facilities table)

A run holds one experiment:

    flexibility/<v>/<flex>/<regime>/<case>/result.json                  optimize flexibility --scenarios v1 --facilities f1
    flexibility_evaluation/<v>/<flex>/<regime>/<case>/evaluation.json   optimize evaluate --run r1
    flexibility_validation_benchmark/<v>/<flex>/<regime>/rp_validation.json   optimize benchmark ...

Models that read satellite capacity (capacitated, flex) run only with a facilities
artifact; its id and digest enter the run manifest and every leaf key.

A capacity table is re-executed at promotion (the CA is deterministic). A time-limited MIP
is not, so a run is not: idempotence is by *leaf key* instead — the digest of everything
that defines a solve. A leaf whose key is already in an official run is copied from it.
"""

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from src.core.constants import (
    PATH_CUSTOMER_PIXEL_LAYER,
    PATH_DATA_DISTANCES_FACILITIES,
    PATH_DATA_DISTANCES_FACILITY_DELIVERY_ZONE,
    PATH_DATA_FACILITY,
    PATH_DATA_PIXEL,
    REGIMES,
    TypeOfFlexibility,
)
from src.core.contract import ScenarioLayout
from src.core.inputs import get_facilities
from src.optimization.capacity.analysis import CapacityConfig, run_analysis
from src.optimization.capacity.table import CAPACITY_FILE, CapacityTable
from src.optimization.capacity.tariffs import PATH_TARIFFS
from src.optimization.experiments import flexibility as flex_exp
from src.optimization.experiments import flexibility_evaluation as eval_exp
from src.optimization.experiments import validation_benchmark as bench_exp
from src.optimization.experiments.runner import ExperimentRunner
from src.optimization.experiments.store import ResultStore
from src.optimization.instance import InstanceBuilder
from src.optimization.models import model_class
from src.tools.artifacts import Artifact, ArtifactKind, ArtifactStore
from src.tools.io import read_json, sha256_json
from src.tools.logging import get_logger
from src.tools.manifest import Manifest, parent_ref, raw_input
from src.tools.validation import Check, Validator, check

logger = get_logger("Runs")

EXPERIMENTS = {
    "flexibility": flex_exp.RESULT_FILE,
    "flexibility_evaluation": eval_exp.EVALUATION_FILE,
    "flexibility_validation_benchmark": bench_exp.BENCHMARK_FILE,
}
ALL_FLEXIBILITIES = tuple(item.value for item in TypeOfFlexibility)
MIN_TOP_LEVEL_COVERAGE = 95.0  # % of capacity scenarios whose peak fleet the largest level covers


def _command(subcommand: str, argv, config: dict) -> dict:
    return {"program": "optimize", "subcommand": subcommand, "argv": list(argv or []), "config": config}


# ── facilities ────────────────────────────────────────────────────────────────


class FacilityStage:
    kind = ArtifactKind.FACILITIES

    def __init__(self, store: ArtifactStore | None = None):
        self.store = store or ArtifactStore()

    @staticmethod
    def produce(out: Path, scenarios: Artifact, config: CapacityConfig, tariffs_path: Path = PATH_TARIFFS) -> dict:
        return run_analysis(out, ScenarioLayout(scenarios.path), scenarios.id, config, tariffs_path)

    def analyze(self, config: CapacityConfig, argv=None) -> Artifact:
        scenarios = self.store.resolve(config.scenarios, ArtifactKind.SCENARIOS)
        artifact = self.store.new_candidate(self.kind)
        summary = self.produce(artifact.path, scenarios, config)
        Manifest(
            kind="facilities",
            id=artifact.id,
            command=_command("capacity analyze", argv, {**config.__dict__, "regimes": list(config.regimes)}),
            inputs={
                "raw": [
                    raw_input(PATH_TARIFFS),
                    raw_input(PATH_DATA_FACILITY),
                    raw_input(PATH_DATA_PIXEL),
                    raw_input(PATH_CUSTOMER_PIXEL_LAYER),
                    raw_input(PATH_DATA_DISTANCES_FACILITY_DELIVERY_ZONE),
                    raw_input(PATH_DATA_DISTANCES_FACILITIES),
                ],
                "parents": [parent_ref(scenarios)],
            },
            details={"scenarios": scenarios.id, **summary},
        ).save(artifact)
        return artifact

    def reproduce(self):
        def run(artifact: Artifact, manifest: Manifest, out: Path) -> None:  # pylint: disable=unused-argument
            config = dict(manifest.command["config"])
            config["regimes"] = tuple(config["regimes"])
            scenarios = self.store.resolve(config["scenarios"], ArtifactKind.SCENARIOS)
            self.produce(out, scenarios, CapacityConfig(**config))

        return run

    def report(self, artifact: Artifact) -> Path:
        from src.optimization.reports import build_capacity_report

        return build_capacity_report(artifact.path, artifact.reports_dir / "capacity.html", label=artifact.id)


class FacilityValidator(Validator):
    """Is a capacity table fit for the models: complete, coherent and covering the demand?"""

    def __init__(self, stage: FacilityStage):
        self.stage = stage

    def checks(self, artifact: Artifact) -> list[Check]:
        satellites = read_json(artifact.path / CAPACITY_FILE)["satellites"]
        expected = set(get_facilities())
        checks = [
            check(
                "todos los satélites de input_facilities.xlsx",
                expected == set(satellites),
                f"{len(satellites)} de {len(expected)}",
            )
        ]
        for facility_id, block in sorted(satellites.items()):
            levels = block["levels"]
            installation = [block["cost_installation"][str(q)] for q in levels]
            opex = [sum(block["cost_operation"][str(q)]) for q in levels]
            top = block["coverage_pct"][str(levels[-1])]
            checks += [
                check(
                    f"[{facility_id}] niveles 0 < q1 < q2 < … (pares)",
                    levels[0] == 0 and all(b > a for a, b in zip(levels, levels[1:])) and all(q % 2 == 0 for q in levels),
                    f"{levels}",
                ),
                check(
                    f"[{facility_id}] el nivel máximo cubre ≥ {MIN_TOP_LEVEL_COVERAGE:.0f}% de los picos",
                    top >= MIN_TOP_LEVEL_COVERAGE,
                    f"nivel {levels[-1]} cubre {top}% (P95 = {block['peak_fleet']['p95']:.1f})",
                    value=top,
                    threshold=MIN_TOP_LEVEL_COVERAGE,
                ),
                check(
                    f"[{facility_id}] costos no decrecen con el nivel",
                    all(b >= a for a, b in zip(installation, installation[1:])) and all(b >= a for a, b in zip(opex, opex[1:])),
                    f"instalación {installation[1:]}",
                ),
                check(
                    f"[{facility_id}] 12 costos de operación ≥ 0 por nivel, cero en el nivel 0",
                    all(len(block["cost_operation"][str(q)]) == 12 for q in levels)
                    and all(x >= 0 for q in levels for x in block["cost_operation"][str(q)])
                    and not any(block["cost_operation"]["0"]),
                    "",
                ),
                check(
                    f"[{facility_id}] tiene píxeles asignados",
                    block["n_pixels"] > 0,
                    f"{block['n_pixels']} píxeles",
                    warn_only=True,
                ),
            ]
        return checks

    def report(self, artifact: Artifact) -> Path:
        return self.stage.report(artifact)


# ── runs ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RunConfig:
    scenarios: str
    regimes: tuple[str, ...] = REGIMES
    flexibilities: tuple[str, ...] = ALL_FLEXIBILITIES
    cases: tuple[str, ...] = tuple(flex_exp.CASES)
    time_limit: float = 600.0
    mip_gap: float = 0.0
    solver_overrides: dict = field(default_factory=dict)
    model: str = "flex"
    facilities: str | None = None


def leaf_key(inputs: dict, experiment: str, solver: dict, flexibility: str, regime: str, case: str | None, model: str) -> str:
    """Digest of everything that defines one solve."""
    cls = model_class(model)
    return sha256_json(
        {
            **{kind: {"id": ref["id"], "content_sha256": ref["content_sha256"]} for kind, ref in inputs.items() if ref},
            "experiment": experiment,
            "model": cls.NAME,
            "features": cls.DEFAULT_FEATURES.as_dict(),
            "solver": solver,
            "flexibility": flexibility,
            "regime": regime,
            "case": case,
        }
    )


@dataclass
class RunInputs:
    """The artifacts a run reads: a scenario version and, for capacitated models, a capacity table."""

    scenarios: Artifact
    facilities: Artifact | None
    model: str

    @classmethod
    def resolve(cls, store: ArtifactStore, scenarios: str, facilities: str | None, model: str) -> "RunInputs":
        if model_class(model).USES_CAPACITY and facilities is None:
            raise ValueError(f"Model {model!r} reads satellite capacity: pass --facilities f<N> (a capacity table).")
        facilities_artifact = store.resolve(facilities, ArtifactKind.FACILITIES) if facilities else None
        return cls(store.resolve(scenarios, ArtifactKind.SCENARIOS), facilities_artifact, model)

    @property
    def parents(self) -> list[Artifact]:
        return [self.scenarios] + ([self.facilities] if self.facilities else [])

    @property
    def identity(self) -> dict:
        return {"scenarios": parent_ref(self.scenarios), "facilities": parent_ref(self.facilities) if self.facilities else None}

    def builder(self) -> InstanceBuilder:
        capacity = CapacityTable.load(self.facilities.path) if self.facilities else None
        return InstanceBuilder(
            ScenarioLayout(self.scenarios.path),
            self.scenarios.id,
            capacity=capacity,
            facilities_id=self.facilities.id if self.facilities else None,
        )


class RunStage:
    kind = ArtifactKind.RUNS

    def __init__(self, store: ArtifactStore | None = None):
        self.store = store or ArtifactStore()

    def _official_leaves(self) -> dict[str, tuple[Artifact, str]]:
        """Leaf key -> (official run, leaf path within it), across every official run."""
        found = {}
        for run_id in self.store.official_ids(self.kind):
            artifact = self.store.resolve(run_id, self.kind)
            for leaf in Manifest.load(artifact.manifest_path).details.get("leaves", []):
                found.setdefault(leaf["key"], (artifact, leaf["path"]))
        return found

    def _record(self, artifact: Artifact, experiment: str, inputs: RunInputs, solver: dict, reused: dict) -> list[dict]:
        version = inputs.scenarios.id
        store = ResultStore(artifact.path / experiment, EXPERIMENTS[experiment])
        leaves = []
        for path in store.iter_leaves(version):
            relative = path.relative_to(artifact.path).as_posix()
            parts = path.relative_to(store.root / version).parts
            flexibility, regime, case = parts[0], parts[1], parts[2] if len(parts) > 3 else None
            payload = read_json(path)
            solve = payload.get("solve") or {}
            leaves.append(
                {
                    "key": leaf_key(inputs.identity, experiment, solver, flexibility, regime, case, inputs.model),
                    "path": relative,
                    "status": payload.get("status") or solve.get("status"),
                    "is_optimal": solve.get("is_optimal"),
                    "reused_from": reused.get(relative),
                }
            )
        return leaves

    def _save(self, artifact, subcommand, argv, config: dict, inputs: RunInputs, experiment, solver, reused, extra_parents=()):
        Manifest(
            kind="runs",
            id=artifact.id,
            command=_command(subcommand, argv, config),
            inputs={"raw": [], "parents": [parent_ref(parent) for parent in [*inputs.parents, *extra_parents]]},
            seeds={"solver": {k: solver.get(k) for k in ("Seed", "Threads")}},
            details={
                "experiment": experiment,
                "model": inputs.model,
                "scenarios": inputs.scenarios.id,
                "facilities": inputs.facilities.id if inputs.facilities else None,
                "solver": solver,
                "leaves": self._record(artifact, experiment, inputs, solver, reused),
            },
        ).save(artifact)

    @staticmethod
    def _config_dict(config: RunConfig, cases=None) -> dict:
        return {
            **config.__dict__,
            "regimes": list(config.regimes),
            "flexibilities": list(config.flexibilities),
            "cases": list(config.cases if cases is None else cases),
        }

    def flexibility(self, config: RunConfig, argv=None, overwrite: bool = False) -> Artifact:
        inputs = RunInputs.resolve(self.store, config.scenarios, config.facilities, config.model)
        solver = {"TimeLimit": config.time_limit, "MIPGap": config.mip_gap, **config.solver_overrides}
        runner = ExperimentRunner(inputs.builder(), config.solver_overrides)
        artifact = self.store.new_candidate(self.kind)
        store = ResultStore(artifact.path / "flexibility", flex_exp.RESULT_FILE)
        official, reused = self._official_leaves(), {}
        for flexibility in flex_exp.policies_for(config.model, config.flexibilities):
            for regime in config.regimes:
                for case in config.cases:
                    run = flex_exp.ExperimentRun(inputs.scenarios.id, regime, flexibility, case, config.model)
                    target = store.leaf_path(run.version, flexibility, regime, case)
                    key = leaf_key(inputs.identity, "flexibility", solver, flexibility, regime, case, config.model)
                    if key in official and not overwrite:
                        source, relative = official[key]
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(source.path / relative, target)
                        reused[target.relative_to(artifact.path).as_posix()] = f"{source.id}/{relative}"
                        logger.info(f"{flexibility}/{regime}/{case}: reused from {source.id}")
                        continue
                    flex_exp.run_one(run, store, runner, config.time_limit, config.mip_gap, overwrite=True)
        self._save(artifact, "flexibility", argv, self._config_dict(config), inputs, "flexibility", solver, reused)
        return artifact

    def evaluate(self, source_ref: str, time_limit: float, argv=None, solver_overrides: dict | None = None) -> Artifact:
        source = self.store.resolve(source_ref, self.kind)
        source_manifest = Manifest.load(source.manifest_path)
        if source_manifest.details.get("experiment") != "flexibility":
            raise ValueError(f"{source} is not a flexibility run.")
        source_config = source_manifest.command["config"]
        model = source_config.get("model", "flex")
        # The recourse is evaluated with the very capacity table the source Y was chosen with.
        inputs = RunInputs.resolve(
            self.store, source_manifest.details["scenarios"], source_manifest.details.get("facilities"), model
        )
        solver = {"TimeLimit": time_limit, "MIPGap": 0.0, **(solver_overrides or {})}
        runner = ExperimentRunner(inputs.builder(), solver_overrides)
        artifact = self.store.new_candidate(self.kind)
        eval_exp.evaluate_experiment(
            inputs.scenarios.id,
            source_config["regimes"],
            flex_exp.policies_for(model, source_config["flexibilities"]),
            [case for case in eval_exp.SOLUTION_CASES if case in source_config["cases"]],
            time_limit,
            source=eval_exp.SourceRun(source.id, source.path),
            output_root=artifact.path / "flexibility_evaluation",
            runner=runner,
        )
        config = {"run": source.id, "time_limit": time_limit, "solver_overrides": dict(solver_overrides or {})}
        self._save(artifact, "evaluate", argv, config, inputs, "flexibility_evaluation", solver, {}, extra_parents=[source])
        return artifact

    def benchmark(self, config: RunConfig, argv=None) -> Artifact:
        inputs = RunInputs.resolve(self.store, config.scenarios, config.facilities, config.model)
        solver = {"TimeLimit": config.time_limit, "MIPGap": 0.0, **config.solver_overrides}
        runner = ExperimentRunner(inputs.builder(), config.solver_overrides)
        artifact = self.store.new_candidate(self.kind)
        for flexibility in flex_exp.policies_for(config.model, config.flexibilities):
            for regime in config.regimes:
                bench_exp.run_validation_benchmark(
                    inputs.scenarios.id,
                    regime,
                    flexibility,
                    config.time_limit,
                    overwrite=True,
                    output_root=artifact.path / "flexibility_validation_benchmark",
                    runner=runner,
                    model=config.model,
                )
        self._save(
            artifact,
            "benchmark",
            argv,
            self._config_dict(config, cases=[]),
            inputs,
            "flexibility_validation_benchmark",
            solver,
            {},
        )
        return artifact

    def report(self, artifact: Artifact, benchmark: Artifact | None = None) -> list[Path]:
        """HTML reports of a run under its `reports/` directory."""
        from src.optimization.reports import build_evaluation_preview, build_evaluation_report, build_flexibility_report

        details = Manifest.load(artifact.manifest_path).details
        version, experiment, written = details["scenarios"], details["experiment"], []
        if experiment == "flexibility":
            written.append(
                build_flexibility_report(
                    version, artifact.reports_dir / "flexibility_comparison.html", root=artifact.path / "flexibility"
                )
            )
        elif experiment == "flexibility_evaluation":
            root = artifact.path / "flexibility_evaluation"
            written.append(build_evaluation_preview(version, artifact.reports_dir / "vss_preview_partial.html", root=root))
            try:
                bench_root = benchmark.path / "flexibility_validation_benchmark" if benchmark else artifact.path / "_no_benchmark"
                written.append(build_evaluation_report(version, artifact.reports_dir / "vss_comparison.html", root, bench_root))
            except ValueError as error:
                logger.warning(f"Full VSS report skipped: {error}")
        return written


class RunValidator(Validator):
    """Every leaf solved (no ERROR); non-optimal incumbents flagged, not failed."""

    def checks(self, artifact: Artifact) -> list[Check]:
        manifest = Manifest.load(artifact.manifest_path)
        leaves = manifest.details.get("leaves", [])
        errors = [leaf["path"] for leaf in leaves if leaf["status"] == "ERROR"]
        not_optimal = [leaf["path"] for leaf in leaves if leaf["status"] != "ERROR" and leaf.get("is_optimal") is False]
        checks = [
            check("la corrida tiene hojas", bool(leaves), f"{len(leaves)} hojas"),
            check("ninguna hoja terminó en ERROR", not errors, f"{len(errors)} con error: {errors[:3]}"),
            check(
                "todas las hojas son óptimas certificadas",
                not not_optimal,
                f"{len(not_optimal)} incumbentes no óptimos (p.ej. TIME_LIMIT): {not_optimal[:3]}",
                warn_only=True,
            ),
        ]
        if model_class(manifest.details.get("model", "flex")).USES_CAPACITY:
            checks.append(check("usa una tabla de capacidad versionada (fN)", bool(manifest.details.get("facilities")), ""))
        return checks

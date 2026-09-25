"""Optimization runs as versioned artifacts: `cr-*` candidates promoted to `r<N>`.

A run holds one experiment over one scenario version:

    flexibility/<v>/<flex>/<regime>/<case>/result.json                  optimize flexibility --scenarios v1
    flexibility_evaluation/<v>/<flex>/<regime>/<case>/evaluation.json   optimize evaluate --run r1
    flexibility_validation_benchmark/<v>/<flex>/<regime>/rp_validation.json   optimize benchmark --scenarios v1

A time-limited MIP is not deterministic, so a run is not re-executed at promotion.
Idempotence is by *leaf key* instead: the digest of everything that defines a solve (the
scenario version and its content, model, solver settings, policy, regime, case). A leaf
whose key is already in an official run is copied from it, not solved again.
"""

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from src.core.constants import REGIMES, TypeOfFlexibility
from src.core.contract import ScenarioLayout
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
from src.tools.manifest import Manifest, parent_ref
from src.tools.validation import Check, Validator, check

logger = get_logger("Runs")

EXPERIMENTS = {
    "flexibility": flex_exp.RESULT_FILE,
    "flexibility_evaluation": eval_exp.EVALUATION_FILE,
    "flexibility_validation_benchmark": bench_exp.BENCHMARK_FILE,
}
ALL_FLEXIBILITIES = tuple(item.value for item in TypeOfFlexibility)


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


def _command(subcommand: str, argv, config: dict) -> dict:
    return {"program": "optimize", "subcommand": subcommand, "argv": list(argv or []), "config": config}


def leaf_key(
    scenarios: dict, experiment: str, solver: dict, flexibility: str, regime: str, case: str | None, model: str = "flex"
) -> str:
    """Digest of everything that defines one solve."""
    cls = model_class(model)
    return sha256_json(
        {
            "scenarios": {"id": scenarios["id"], "content_sha256": scenarios["content_sha256"]},
            "experiment": experiment,
            "model": cls.NAME,
            "features": cls.DEFAULT_FEATURES.as_dict(),
            "solver": solver,
            "flexibility": flexibility,
            "regime": regime,
            "case": case,
        }
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

    def _record(self, artifact: Artifact, experiment: str, scenarios: dict, solver: dict, reused: dict, model: str) -> list[dict]:
        store = ResultStore(artifact.path / experiment, EXPERIMENTS[experiment])
        leaves = []
        for path in store.iter_leaves(scenarios["id"]):
            relative = path.relative_to(artifact.path).as_posix()
            parts = path.relative_to(store.root / scenarios["id"]).parts
            flexibility, regime, case = parts[0], parts[1], parts[2] if len(parts) > 3 else None
            payload = read_json(path)
            solve = payload.get("solve") or {}
            leaves.append(
                {
                    "key": leaf_key(scenarios, experiment, solver, flexibility, regime, case, model),
                    "path": relative,
                    "status": payload.get("status") or solve.get("status"),
                    "is_optimal": solve.get("is_optimal"),
                    "reused_from": reused.get(relative),
                }
            )
        return leaves

    def _save(  # pylint: disable=too-many-arguments
        self, artifact, subcommand, argv, config: dict, parents: list[Artifact], experiment, scenarios, solver, reused, model
    ):
        Manifest(
            kind="runs",
            id=artifact.id,
            command=_command(subcommand, argv, config),
            inputs={"raw": [], "parents": [parent_ref(parent) for parent in parents]},
            seeds={"solver": {k: solver.get(k) for k in ("Seed", "Threads")}},
            details={
                "experiment": experiment,
                "model": model,
                "scenarios": scenarios["id"],
                "solver": solver,
                "leaves": self._record(artifact, experiment, scenarios, solver, reused, model),
            },
        ).save(artifact)

    def _scenarios(self, ref: str) -> tuple[Artifact, dict]:
        artifact = self.store.resolve(ref, ArtifactKind.SCENARIOS)
        identity = parent_ref(artifact)
        return artifact, identity

    def flexibility(self, config: RunConfig, argv=None, overwrite: bool = False) -> Artifact:
        scenarios_artifact, scenarios = self._scenarios(config.scenarios)
        solver = {"TimeLimit": config.time_limit, "MIPGap": config.mip_gap, **config.solver_overrides}
        runner = ExperimentRunner(
            InstanceBuilder(ScenarioLayout(scenarios_artifact.path), scenarios_artifact.id), config.solver_overrides
        )
        artifact = self.store.new_candidate(self.kind)
        store = ResultStore(artifact.path / "flexibility", flex_exp.RESULT_FILE)
        official, reused = self._official_leaves(), {}
        for flexibility in flex_exp.policies_for(config.model, config.flexibilities):
            for regime in config.regimes:
                for case in config.cases:
                    run = flex_exp.ExperimentRun(scenarios_artifact.id, regime, flexibility, case, config.model)
                    target = store.leaf_path(run.version, flexibility, regime, case)
                    key = leaf_key(scenarios, "flexibility", solver, flexibility, regime, case, config.model)
                    if key in official and not overwrite:
                        source, relative = official[key]
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(source.path / relative, target)
                        reused[target.relative_to(artifact.path).as_posix()] = f"{source.id}/{relative}"
                        logger.info(f"{flexibility}/{regime}/{case}: reused from {source.id}")
                        continue
                    flex_exp.run_one(run, store, runner, config.time_limit, config.mip_gap, overwrite=True)
        config_dict = {
            **config.__dict__,
            "regimes": list(config.regimes),
            "flexibilities": list(config.flexibilities),
            "cases": list(config.cases),
        }
        self._save(
            artifact,
            "flexibility",
            argv,
            config_dict,
            [scenarios_artifact],
            "flexibility",
            scenarios,
            solver,
            reused,
            config.model,
        )
        return artifact

    def evaluate(self, source_ref: str, time_limit: float, argv=None, solver_overrides: dict | None = None) -> Artifact:
        source = self.store.resolve(source_ref, self.kind)
        source_manifest = Manifest.load(source.manifest_path)
        if source_manifest.details.get("experiment") != "flexibility":
            raise ValueError(f"{source} is not a flexibility run.")
        scenarios_artifact, scenarios = self._scenarios(source_manifest.details["scenarios"])
        source_config = source_manifest.command["config"]
        solver = {"TimeLimit": time_limit, "MIPGap": 0.0, **(solver_overrides or {})}
        runner = ExperimentRunner(
            InstanceBuilder(ScenarioLayout(scenarios_artifact.path), scenarios_artifact.id), solver_overrides
        )
        artifact = self.store.new_candidate(self.kind)
        eval_exp.evaluate_experiment(
            scenarios_artifact.id,
            source_config["regimes"],
            flex_exp.policies_for(source_config.get("model", "flex"), source_config["flexibilities"]),
            [case for case in eval_exp.SOLUTION_CASES if case in source_config["cases"]],
            time_limit,
            source=eval_exp.SourceRun(source.id, source.path),
            output_root=artifact.path / "flexibility_evaluation",
            runner=runner,
        )
        config = {"run": source.id, "time_limit": time_limit, "solver_overrides": dict(solver_overrides or {})}
        self._save(
            artifact,
            "evaluate",
            argv,
            config,
            [scenarios_artifact, source],
            "flexibility_evaluation",
            scenarios,
            solver,
            {},
            source_config.get("model", "flex"),
        )
        return artifact

    def benchmark(self, config: RunConfig, argv=None) -> Artifact:
        scenarios_artifact, scenarios = self._scenarios(config.scenarios)
        solver = {"TimeLimit": config.time_limit, "MIPGap": 0.0, **config.solver_overrides}
        runner = ExperimentRunner(
            InstanceBuilder(ScenarioLayout(scenarios_artifact.path), scenarios_artifact.id), config.solver_overrides
        )
        artifact = self.store.new_candidate(self.kind)
        for flexibility in flex_exp.policies_for(config.model, config.flexibilities):
            for regime in config.regimes:
                bench_exp.run_validation_benchmark(
                    scenarios_artifact.id,
                    regime,
                    flexibility,
                    config.time_limit,
                    overwrite=True,
                    output_root=artifact.path / "flexibility_validation_benchmark",
                    runner=runner,
                    model=config.model,
                )
        config_dict = {
            **config.__dict__,
            "regimes": list(config.regimes),
            "flexibilities": list(config.flexibilities),
            "cases": [],
        }
        self._save(
            artifact,
            "benchmark",
            argv,
            config_dict,
            [scenarios_artifact],
            "flexibility_validation_benchmark",
            scenarios,
            solver,
            {},
            config.model,
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
        return [
            check("la corrida tiene hojas", bool(leaves), f"{len(leaves)} hojas"),
            check("ninguna hoja terminó en ERROR", not errors, f"{len(errors)} con error: {errors[:3]}"),
            check(
                "todas las hojas son óptimas certificadas",
                not not_optimal,
                f"{len(not_optimal)} incumbentes no óptimos (p.ej. TIME_LIMIT): {not_optimal[:3]}",
                warn_only=True,
            ),
        ]

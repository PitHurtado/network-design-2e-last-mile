"""The versioned stages of the scenario side: parameters, scenario versions, comparisons.

Each stage knows how to produce its artifact's content into a directory (`produce_*`),
how to re-execute that from the manifest for `promote` (`reproduce`), and how to validate
it. Candidates are created through an `ArtifactStore`; nothing here picks a path itself.

    params     cp-* -> p<N>   shape_params.json + the panel it was fitted on
    scenarios  cv-* -> v<N>   <regime>/<set>/scenario_*.json + set manifests
    comparison cc-*           exploratory paired sets per dependence method, never promoted
"""

import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.core.constants import (
    N_PERIODS,
    PATH_CUSTOMER_PIXEL_LAYER,
    PATH_DATA_PIXEL,
    PATH_PANEL_MONTHLY,
    PATH_PANEL_SOURCE,
    REGIME_TARGETS,
    REGIMES,
    SEED_BASE,
)
from src.core.contract import DEPENDENCE_METHODS, SCENARIO_SETS, ScenarioLayout
from src.core.inputs import get_pixels
from src.scenarios.fitting.fitter import FitResult, ParamsFitter  # noqa: F401 - FitResult re-exported
from src.scenarios.fitting.panel import load_panel
from src.scenarios.generation.generator import ScenarioGenerator
from src.scenarios.generation.seeds import SEED_SCHEME
from src.scenarios.generation.sets import ScenarioSetWriter, SetSpec
from src.scenarios.params import ShapeParams
from src.scenarios.validation.params import roundtrip_validation
from src.scenarios.validation.scenarios import contract_checks
from src.tools.artifacts import Artifact, ArtifactKind, ArtifactStore
from src.tools.io import read_json, sha256_file
from src.tools.manifest import Manifest, parent_ref, raw_input
from src.tools.validation import Check, Validator, check

PARAMS_FILE = "shape_params.json"
PANEL_FILE = "panel_monthly.csv"
PANEL_SOURCE_FILE = "panel_source.json"

# Acceptance thresholds of a params artifact, set against p1's measured values
# (round-trip rho error ~2%, weighted R² 0.884).
MAX_ROUNDTRIP_RHO_ERROR = 0.10
MIN_CORRELOGRAM_R2 = 0.80
MAX_REGIME_TARGET_ERROR = 0.01


def _command(subcommand: str, argv: list[str] | None, config: dict) -> dict:
    return {"program": "scenarios", "subcommand": subcommand, "argv": list(argv or []), "config": config}


# ── params ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FitConfig:
    """`params fit`: fit on the panel and calibrate the regimes on the first `n` validation streams.

    `validation_n` is kept only to reproduce artifacts made before the fit calibrated on the
    validation streams itself (they ran a second recalibration pass of that size); when set
    it is the calibration size.
    """

    n: int = 100
    bins: int = 12
    seed_base: int = SEED_BASE
    validation_n: int | None = None

    @property
    def calibration_n(self) -> int:
        return self.validation_n if self.validation_n is not None else self.n


class ParamsStage:
    kind = ArtifactKind.PARAMS

    @staticmethod
    def load(artifact: Artifact) -> tuple[ShapeParams, pd.DataFrame]:
        return ShapeParams.load(artifact.path / PARAMS_FILE), pd.read_csv(artifact.path / PANEL_FILE)

    @staticmethod
    def produce_fit(out: Path, config: FitConfig, panel_path: Path, panel_source: Path | None) -> FitResult:
        """Fit into `out`, which receives its own copy of the panel so the artifact is self-contained."""
        shutil.copyfile(panel_path, out / PANEL_FILE)
        if panel_source is not None and panel_source.exists():
            shutil.copyfile(panel_source, out / PANEL_SOURCE_FILE)
        fitter = ParamsFitter(bins=config.bins, n_calibration=config.calibration_n, seed_base=config.seed_base)
        result = fitter.fit(pd.read_csv(out / PANEL_FILE), set(get_pixels()))
        result.params.save(out / PARAMS_FILE)
        return result

    @staticmethod
    def produce_recalibrate(out: Path, parent: Artifact, validation_n: int, seed_base: int) -> tuple[ShapeParams, dict]:
        for name in (PANEL_FILE, PANEL_SOURCE_FILE):
            if (parent.path / name).exists():
                shutil.copyfile(parent.path / name, out / name)
        params, regimes = ParamsFitter(seed_base=seed_base).recalibrate(ShapeParams.load(parent.path / PARAMS_FILE), validation_n)
        params.save(out / PARAMS_FILE)
        return params, regimes

    @staticmethod
    def _details(params: ShapeParams, regimes: dict) -> dict:
        return {
            "regimes": params["regimes"],
            "spatial": {k: params["spatial"][k] for k in ("model", "rho_km", "nugget", "plateau", "weighted_r2")},
            "calibration_iterations": {regime: len(result["history"]) for regime, result in regimes.items()},
            "shape_params_sha256": params.sha256,
        }

    def fit(self, store: ArtifactStore, config: FitConfig, argv: list[str] | None = None) -> Artifact:
        panel = PATH_PANEL_MONTHLY
        if not panel.exists():
            load_panel()  # builds and caches it
        artifact = store.new_candidate(self.kind)
        result = self.produce_fit(artifact.path, config, panel, PATH_PANEL_SOURCE)
        seeds = {"seed_base": config.seed_base, "scheme": f"calibration on {SEED_SCHEME} (validation, n={config.calibration_n})"}
        Manifest(
            kind="params",
            id=artifact.id,
            command=_command("params fit", argv, {"mode": "fit", **config.__dict__}),
            inputs={
                "raw": [raw_input(PATH_DATA_PIXEL), raw_input(PATH_CUSTOMER_PIXEL_LAYER), raw_input(artifact.path / PANEL_FILE)],
                "parents": [],
            },
            seeds=seeds,
            details=self._details(result.params, result.regimes),
        ).save(artifact)
        return artifact

    def recalibrate(
        self, store: ArtifactStore, parent: Artifact, validation_n: int, seed_base: int, argv: list[str] | None = None
    ) -> Artifact:
        artifact = store.new_candidate(self.kind)
        params, regimes = self.produce_recalibrate(artifact.path, parent, validation_n, seed_base)
        Manifest(
            kind="params",
            id=artifact.id,
            command=_command(
                "params recalibrate",
                argv,
                {"mode": "recalibrate", "parent": parent.id, "validation_n": validation_n, "seed_base": seed_base},
            ),
            inputs={"raw": [raw_input(PATH_DATA_PIXEL), raw_input(PATH_CUSTOMER_PIXEL_LAYER)], "parents": [parent_ref(parent)]},
            seeds={"seed_base": seed_base, "scheme": f"{SEED_SCHEME} (validation, n={validation_n})"},
            details=self._details(params, regimes),
        ).save(artifact)
        return artifact

    def reproduce(self, store: ArtifactStore):
        """Reproducer for `Promoter`: re-run the recorded command into a scratch directory."""

        def run(artifact: Artifact, manifest: Manifest, out: Path) -> None:
            config = dict(manifest.command["config"])
            if config.pop("mode") == "fit":
                source = artifact.path / PANEL_SOURCE_FILE
                self.produce_fit(out, FitConfig(**config), artifact.path / PANEL_FILE, source if source.exists() else None)
            else:
                parent = store.resolve(config["parent"], ArtifactKind.PARAMS)
                self.produce_recalibrate(out, parent, config["validation_n"], config["seed_base"])

        return run


class ParamsValidator(Validator):
    """Is a params artifact fit to generate scenarios from?"""

    def checks(self, artifact: Artifact) -> list[Check]:
        params, panel = ParamsStage.load(artifact)
        grid = set(get_pixels())
        checks = [check("píxeles == input_pixels.xlsx", set(params.pixels) == grid, f"{len(params.pixels)} vs {len(grid)}")]
        scaling = params.regime_scaling
        total = scaling["stop_exponent"] + scaling["drop_exponent"]
        checks.append(check("exponentes de régimen suman 1", abs(total - 1) < 1e-9, f"{total:.6f}"))
        for regime in REGIMES:
            block = params.regime(regime)
            error = block["realized_period_demand"] / REGIME_TARGETS[regime] - 1
            checks.append(
                check(
                    f"[{regime}] objetivo alcanzado (±{MAX_REGIME_TARGET_ERROR:.0%})",
                    abs(error) < MAX_REGIME_TARGET_ERROR,
                    f"{block['realized_period_demand']:,.0f} vs {REGIME_TARGETS[regime]:,.0f}",
                    value=error,
                    threshold=MAX_REGIME_TARGET_ERROR,
                )
            )
        r2 = params.spatial["weighted_r2"]
        checks.append(
            check(
                "R² ponderado del correlograma",
                r2 >= MIN_CORRELOGRAM_R2,
                f"{r2:.3f}",
                warn_only=True,
                value=r2,
                threshold=MIN_CORRELOGRAM_R2,
            )
        )
        included = panel[~panel["excluded"]]
        roundtrip = roundtrip_validation(params, included.groupby(["year", "month"]).ngroups, len(params.pixels))
        rho, recovered = params.spatial["rho_km"], roundtrip["recovered"]["rho_km"]
        rho_error = abs(recovered / rho - 1) if rho else float("inf")
        checks.append(
            check(
                "round-trip recupera ρ",
                rho_error <= MAX_ROUNDTRIP_RHO_ERROR,
                f"{recovered:.3f} vs {rho:.3f} km ({rho_error:.1%})",
                value=rho_error,
                threshold=MAX_ROUNDTRIP_RHO_ERROR,
            )
        )
        plateau, recovered_plateau = params.spatial["plateau"], roundtrip["recovered"]["plateau"]
        checks.append(
            check(
                "round-trip recupera el plateau (limitación conocida: factor común homogéneo)",
                abs(recovered_plateau - plateau) < 0.5 * max(plateau, 1e-6),
                f"{recovered_plateau:.3f} vs {plateau:.3f}",
                warn_only=True,
            )
        )
        return checks


# ── scenarios ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GenerateConfig:
    params: str
    regimes: tuple[str, ...] = REGIMES
    optimization_n: int = 30
    validation_n: int = 100
    seed_base: int = SEED_BASE


class ScenarioStage:
    kind = ArtifactKind.SCENARIOS

    @staticmethod
    def params_of(store: ArtifactStore, artifact: Artifact) -> tuple[Artifact, ShapeParams, pd.DataFrame]:
        """The params artifact a scenario version was generated from, loaded."""
        params_id = Manifest.load(artifact.manifest_path).parent("params")["id"]
        params_artifact = store.resolve(params_id, ArtifactKind.PARAMS)
        return (params_artifact, *ParamsStage.load(params_artifact))

    @staticmethod
    def produce(out: Path, params: ShapeParams, config: GenerateConfig) -> dict:
        """Every set of every regime into `out`; returns `{regime: {set: {n, manifest_sha256}}}`."""
        generator = ScenarioGenerator.from_params(params)
        writer = ScenarioSetWriter(ScenarioLayout(out))
        sets = {}
        for regime in config.regimes:
            sets[regime] = {}
            for scenario_set, count in (
                ("optimization", config.optimization_n),
                ("validation", config.validation_n),
                ("expected", 1),
                ("annual_expected", 1),
            ):
                spec = SetSpec(regime, scenario_set, count, params.multiplier(regime), config.seed_base)
                summary = writer.generate(generator, spec, params.sha256)
                path = writer.write_manifest(
                    regime, scenario_set, {**summary, "target_period_demand": params.regime(regime)["target_period_demand"]}
                )
                sets[regime][scenario_set] = {"n": summary["n_scenarios"], "manifest_sha256": sha256_file(path)}
        return sets

    def generate(self, store: ArtifactStore, config: GenerateConfig, argv: list[str] | None = None) -> Artifact:
        params_artifact = store.resolve(config.params, ArtifactKind.PARAMS)
        params, _ = ParamsStage.load(params_artifact)
        artifact = store.new_candidate(self.kind)
        sets = self.produce(artifact.path, params, config)
        Manifest(
            kind="scenarios",
            id=artifact.id,
            command=_command("generate", argv, {**config.__dict__, "regimes": list(config.regimes)}),
            inputs={
                "raw": [raw_input(PATH_DATA_PIXEL), raw_input(PATH_CUSTOMER_PIXEL_LAYER)],
                "parents": [parent_ref(params_artifact)],
            },
            seeds={"seed_base": config.seed_base, "scheme": SEED_SCHEME},
            details={"params": params_artifact.id, "dependence": "spatial_joint", "periods": N_PERIODS, "sets": sets},
        ).save(artifact)
        return artifact

    def reproduce(self, store: ArtifactStore):
        def run(artifact: Artifact, manifest: Manifest, out: Path) -> None:  # pylint: disable=unused-argument
            config = dict(manifest.command["config"])
            config["regimes"] = tuple(config["regimes"])
            params, _ = ParamsStage.load(store.resolve(config["params"], ArtifactKind.PARAMS))
            self.produce(out, params, GenerateConfig(**config))

        return run


class ScenarioValidator(Validator):
    """Contract invariants of every regime's sets, plus the validation report."""

    def __init__(self, store: ArtifactStore):
        self.store = store

    def checks(self, artifact: Artifact) -> list[Check]:
        layout = ScenarioLayout(artifact.path)
        writer = ScenarioSetWriter(layout)
        regimes = Manifest.load(artifact.manifest_path).command["config"]["regimes"]
        generated = {regime: writer.load_long(regime, "validation") for regime in regimes}
        manifests = {regime: read_json(layout.set_manifest(regime, "validation")) for regime in regimes}
        checks = contract_checks(generated, manifests)
        for regime in regimes:
            for scenario_set in SCENARIO_SETS:
                listed = read_json(layout.set_manifest(regime, scenario_set))["scenario_ids"]
                on_disk = sorted(
                    p.name[len("scenario_") : -len(".json")] for p in layout.set_dir(regime, scenario_set).glob("scenario_*.json")
                )
                checks.append(
                    check(
                        f"[{regime}/{scenario_set}] ids del manifest == archivos", sorted(listed) == on_disk, f"{len(listed)} ids"
                    )
                )
        if set(REGIMES) <= set(regimes):
            means = [manifests[regime]["period_total_mean"] for regime in REGIMES]
            checks.append(check("orden low < normal < high", means == sorted(means), " < ".join(f"{m:,.0f}" for m in means)))
        return checks

    def report(self, artifact: Artifact) -> Path:
        from src.scenarios.reports import build_validation_report

        params_artifact, params, panel = ScenarioStage.params_of(self.store, artifact)
        regimes = tuple(Manifest.load(artifact.manifest_path).command["config"]["regimes"])
        created = Manifest.load(params_artifact.manifest_path).created_at[:10]
        path, _ = build_validation_report(
            ScenarioLayout(artifact.path),
            params,
            panel,
            artifact.reports_dir / "validation.html",
            regimes=regimes,
            params_label=f"{params_artifact.id} del {created}",
        )
        return path


# ── comparison ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CompareConfig:
    params: str
    regimes: tuple[str, ...] = ("normal",)
    n: int = 100
    seed_base: int = SEED_BASE


class ComparisonStage:
    kind = ArtifactKind.COMPARISONS

    @staticmethod
    def produce(out: Path, params: ShapeParams, panel: pd.DataFrame, config: CompareConfig) -> None:
        writer = ScenarioSetWriter(ScenarioLayout(out, comparison=True))
        for regime in config.regimes:
            for method in DEPENDENCE_METHODS:
                generator = ScenarioGenerator.from_params(
                    params, method=method, panel=panel if method == "historical_bootstrap" else None
                )
                spec = SetSpec(regime, "validation", config.n, params.multiplier(regime), config.seed_base, method=method)
                writer.write_manifest(
                    regime, "validation", writer.generate(generator, spec, params_sha256=params.sha256), method=method
                )

    def compare(self, store: ArtifactStore, config: CompareConfig, argv: list[str] | None = None) -> Artifact:
        from src.scenarios.reports import build_comparison_report

        params_artifact = store.resolve(config.params, ArtifactKind.PARAMS)
        params, panel = ParamsStage.load(params_artifact)
        artifact = store.new_candidate(self.kind)
        self.produce(artifact.path, params, panel, config)
        Manifest(
            kind="comparisons",
            id=artifact.id,
            command=_command("compare", argv, {**config.__dict__, "regimes": list(config.regimes)}),
            inputs={"raw": [raw_input(PATH_CUSTOMER_PIXEL_LAYER)], "parents": [parent_ref(params_artifact)]},
            seeds={"seed_base": config.seed_base, "scheme": "validation streams, paired across methods"},
            details={"params": params_artifact.id, "methods": list(DEPENDENCE_METHODS)},
        ).save(artifact)
        layout = ScenarioLayout(artifact.path, comparison=True)
        for regime in config.regimes:
            build_comparison_report(layout, params, panel, regime, artifact.reports_dir / regime, label=artifact.id)
        return artifact

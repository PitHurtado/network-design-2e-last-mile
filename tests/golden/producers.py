"""One producer per golden. Each returns a JSON-serializable dict of the outputs it pins.

This is the only file that changes when the refactor moves an API: producers call the
code, the stored expectations describe what it must keep producing.
"""

import json
import shutil
from pathlib import Path

import pandas as pd

from src.core.constants import SEED_BASE  # noqa: E402
from tests.golden.support import (
    FIXTURES,
    VERSION,
    Workspace,
    dump,
    figure_capture,
    normalize_ids,
    patched,
    sha_bytes,
    sha_json,
    sha_text,
    tree_digest,
    visible_text,
)

PARAMS = FIXTURES / "shape_params.json"
PANEL = FIXTURES / "panel_monthly.csv"
FIXTURE_RESULTS = FIXTURES / "results"
FIXTURE_RESULTS_VERSION = "v3"
REGIMES = ("low", "normal", "high")
FLEXIBILITIES = ("fixed_operation", "on_off_installed", "up_to_installed")

# Deterministic solver settings: one thread, fixed seed, and a *work* limit, which
# unlike a time limit stops at the same point on every run.
SOLVER = {"Threads": 1, "Seed": 0, "WorkLimit": 3.0}

MANIFEST_FIELDS = (
    "scenario_set",
    "method",
    "scenario_ids",
    "multiplier",
    "stop_factor",
    "drop_factor",
    "regime_scaling",
    "n_scenarios",
    "seed_base",
    "period_total_mean",
    "period_total_p10",
    "period_total_p90",
    "stop_floor_hits",
    "stop_cells_drawn",
    "stop_floor_share",
    "pixels",
    "periods",
    "target_period_demand",
)


def _load_params() -> dict:
    return json.loads(PARAMS.read_text())


def _params_summary(params: dict, name: str) -> dict:
    """Byte digest of the params as persisted, minus the date (it moves to the manifest)."""
    params = dict(params)
    params.pop("generated_on", None)
    text = json.dumps(params, indent=2)
    dump(f"{name}_shape_params.json", text)
    return {
        "sha": sha_text(text),
        "regimes": params["regimes"],
        "spatial": {k: params["spatial"][k] for k in ("model", "rho_km", "nugget", "plateau", "weighted_r2")},
        "base_period_demand": params["base_period_demand"],
        "calibrated_n_scenarios": params["calibrated_n_scenarios"],
    }


def _manifests(root: Path) -> dict:
    out = {}
    for path in sorted(root.rglob("manifest.json")):
        manifest = json.loads(normalize_ids(path.read_text()))
        out[normalize_ids(str(path.relative_to(root)))] = {k: manifest.get(k) for k in MANIFEST_FIELDS if k in manifest}
    return out


# ── G1 / G2: fitting ──────────────────────────────────────────────────────────


def g1_fit(ws: Workspace) -> dict:
    from src.scenarios.stages import FitConfig, ParamsStage

    out = ws.path("g1")
    ParamsStage.produce_fit(out, FitConfig(n=5), PANEL, None)
    return _params_summary(json.loads((out / "shape_params.json").read_text()), "g1")


def g2_recalibrate(ws: Workspace) -> dict:
    from src.scenarios.stages import ParamsStage
    from src.tools.artifacts import Artifact, ArtifactKind

    parent = ws.path("g2-parent")
    shutil.copy(PARAMS, parent / "shape_params.json")
    out = ws.path("g2")
    ParamsStage.produce_recalibrate(out, Artifact(ArtifactKind.PARAMS, "p0", parent), validation_n=5, seed_base=SEED_BASE)
    return _params_summary(json.loads((out / "shape_params.json").read_text()), "g2")


# ── G3 / G4: generation ───────────────────────────────────────────────────────


def generated_root(ws: Workspace) -> Path:
    """Scenario sets for every regime (n=3), generated once per session."""

    def build() -> Path:
        from src.scenarios.params import ShapeParams
        from src.scenarios.stages import GenerateConfig, ScenarioStage

        root = ws.path("scenarios", VERSION)
        ScenarioStage.produce(
            root, ShapeParams.load(PARAMS), GenerateConfig("pG", REGIMES, optimization_n=3, validation_n=3, capacity_n=3)
        )
        return root

    return ws.once("generated", build)


def comparison_root(ws: Workspace) -> Path:
    """Paired comparison sets for `normal`, three methods, n=3."""

    def build() -> Path:
        from src.scenarios.params import ShapeParams
        from src.scenarios.stages import CompareConfig, ComparisonStage

        root = ws.path("comparison", VERSION)
        ComparisonStage.produce(root, ShapeParams.load(PARAMS), pd.read_csv(PANEL), CompareConfig("pG", ("normal",), n=3))
        return root

    return ws.once("comparison", build)


def g3_generate(ws: Workspace) -> dict:
    root = generated_root(ws)
    return {"files": tree_digest(root), "manifests": _manifests(root)}


def g4_compare(ws: Workspace) -> dict:
    root = comparison_root(ws)
    return {"files": tree_digest(root), "manifests": _manifests(root)}


# ── G5: spatial ───────────────────────────────────────────────────────────────


def g5_spatial(ws: Workspace) -> dict:  # pylint: disable=unused-argument
    from src.scenarios.spatial import cholesky_factor, correlation_matrix, haversine_matrix, pixel_centroids, pixel_neighbor_pairs

    params = _load_params()
    pixels = params["pixels"]
    rings = {}
    for ring in (1, 2, 3):
        frame = pixel_neighbor_pairs(pixels, ring=ring)
        rings[str(ring)] = {"n": len(frame), "sha": sha_text(frame.to_csv(index=False))}
    centroids = pixel_centroids().set_index("id_pixel").reindex(pixels)
    distances = haversine_matrix(centroids["lon"].to_numpy(), centroids["lat"].to_numpy())
    spatial = params["spatial"]
    sigma = correlation_matrix(distances, spatial["rho_km"], spatial["nugget"], spatial["plateau"])
    return {
        "rings": rings,
        "centroids": sha_text(centroids.to_csv()),
        "distances": sha_bytes(distances.tobytes()),
        "correlation": sha_bytes(sigma.tobytes()),
        "cholesky": sha_bytes(cholesky_factor(sigma).tobytes()),
    }


# ── G6: reports ───────────────────────────────────────────────────────────────


def _report_digest(name: str, html_path: Path, figures: list[str], extra_files: list[Path] = ()) -> dict:
    text = normalize_ids(visible_text(html_path.read_text()))
    dump(f"{name}.txt", text)
    out = {"figures": figures, "text": sha_text(text)}
    for path in extra_files:
        content = normalize_ids(path.read_text())
        dump(f"{name}__{path.name}", content)
        out[path.name] = sha_text(content)
    return out


def g6_reports(ws: Workspace) -> dict:
    from src.core.constants import REGIMES as ALL_REGIMES

    generated = generated_root(ws)
    comparison = comparison_root(ws)
    out_dir = ws.path("reports")
    results = {}

    from src.core.contract import ScenarioLayout
    from src.scenarios.params import ShapeParams
    from src.scenarios.reports import build_comparison_report, build_explore_report, build_validation_report
    from src.scenarios.validation.params import aggregate_cv_impact

    params, panel = ShapeParams.load(PARAMS), pd.read_csv(PANEL)
    with figure_capture() as figures:
        path, n_failed = build_validation_report(
            ScenarioLayout(generated), params, panel, out_dir / "validation.html", ALL_REGIMES
        )
        results["validation"] = {**_report_digest("validation", path, list(figures)), "n_failed": n_failed}
        results["aggregate_cv"] = aggregate_cv_impact(_load_params())

    with figure_capture() as figures:
        path = build_explore_report(ScenarioLayout(generated), out_dir / "explore.html", ALL_REGIMES)
        results["explore"] = _report_digest("explore", path, list(figures))

    with figure_capture() as figures:
        cmp_dir = ws.path("reports", "comparison")
        layout = ScenarioLayout(comparison, comparison=True)
        path = build_comparison_report(layout, params, panel, "normal", cmp_dir, label=VERSION)
        extra = sorted(p for p in cmp_dir.iterdir() if p.suffix in (".csv", ".json"))
        results["comparison"] = _report_digest("comparison", path, list(figures), extra)

    with figure_capture() as figures:
        from src.optimization.reports import build_flexibility_report as build_flex

        flex_dir = ws.path("reports", "flexibility")
        path = build_flex(FIXTURE_RESULTS_VERSION, output_path=flex_dir / "flex.html", root=FIXTURE_RESULTS / "flexibility")
        results["flexibility"] = _report_digest("flexibility", path, list(figures), [flex_dir / "summary.json"])

    if True:  # noqa: SIM108 - one block per report, like the ones above
        from src.optimization.reports import build_evaluation_preview as build_partial_preview
        from src.optimization.reports import build_evaluation_report as build_vss

        root = FIXTURE_RESULTS / "flexibility_evaluation"
        with figure_capture() as figures:
            vss_dir = ws.path("reports", "vss")
            path = build_vss(
                FIXTURE_RESULTS_VERSION, vss_dir / "vss.html", root, FIXTURE_RESULTS / "flexibility_validation_benchmark"
            )
            summary = vss_dir / "vss_summary.json"
            summary.write_text(ws.relativize(summary.read_text()))
            results["vss"] = _report_digest("vss", path, list(figures), [summary])
        with figure_capture() as figures:
            path = build_partial_preview(FIXTURE_RESULTS_VERSION, output_path=vss_dir / "partial.html", root=root)
            results["vss_partial"] = _report_digest("vss_partial", path, list(figures))
    return results


# ── G7: continuous approximation ──────────────────────────────────────────────


def _canon(value):
    if isinstance(value, dict):
        return sorted([normalize_ids(repr(key)), _canon(item)] for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return [_canon(item) for item in value]
    if hasattr(value, "item"):  # numpy scalar
        return value.item()
    return value


def _instance_digest(instance) -> dict:
    out = {
        "periods": instance.periods,
        "horizon_weight": instance.horizon_weight,
        "scenario_ids": [normalize_ids(i) for i in instance.scenarios_ids],
        "facilities": sorted(instance.facilities),
    }
    for id_scenario, scenario in instance.scenarios.items():
        key = normalize_ids(id_scenario)
        out[key] = {
            "pixels": len(scenario.pixels),
            "costs": sha_json(_canon(scenario.costs)),
            "fleet_size": sha_json(_canon(scenario.fleet_size)),
            "parameters": sha_json(_canon(scenario.parameters)),
            "cost_facility_total": round(sum(scenario.costs["facility"].values()), 6),
            "cost_dc_total": round(sum(scenario.costs["dc"].values()), 6),
        }
    return out


INSTANCE_CASES = {
    "optimization_euclidean": {"N": 2, "scenario_set": "optimization", "use_euclidean_distance": True},
    "optimization_matrix": {"N": 2, "scenario_set": "optimization", "use_euclidean_distance": False},
    "expected": {"N": 1, "scenario_set": "expected", "use_euclidean_distance": True},
    "annual_expected": {"N": 1, "scenario_set": "annual_expected", "periods": 1, "use_euclidean_distance": True},
}


def build_instance(
    ws: Workspace, regime: str = "normal", continuous_x: bool = False, flexibility: str = "up_to_installed", **case
):
    from src.core.contract import ScenarioLayout
    from src.optimization.instance import InstanceBuilder, InstanceSpec

    spec = InstanceSpec(
        id_instance="golden",
        n_scenarios=case.pop("N"),
        regime=regime,
        is_continuous_var_x=continuous_x,
        type_of_flexibility=flexibility,
        **case,
    )
    return InstanceBuilder(ScenarioLayout(generated_root(ws)), version=VERSION).build(spec)


def g7_ca(ws: Workspace) -> dict:
    return {label: _instance_digest(build_instance(ws, **case)) for label, case in INSTANCE_CASES.items()}


# ── G8: solves ────────────────────────────────────────────────────────────────


def _clean_payload(payload: dict, ws: Workspace) -> dict:
    payload = json.loads(normalize_ids(ws.relativize(json.dumps(payload))))
    for key in ("solve", "source_solve"):
        if isinstance(payload.get(key), dict):
            payload[key].pop("actual_run_time", None)
    return payload


def g8_solve(ws: Workspace) -> dict:
    from src.core.contract import ScenarioLayout
    from src.optimization.experiments import flexibility as flexibility_module
    from src.optimization.experiments.flexibility_evaluation import SOLUTION_CASES, SourceRun, evaluate_experiment
    from src.optimization.experiments.runner import ExperimentRunner
    from src.optimization.instance import InstanceBuilder

    cases = {**flexibility_module.CASES, "optimization": {**flexibility_module.CASES["optimization"], "n_scenarios": 2}}
    results_root = ws.path("results")
    with patched(
        {
            "src.optimization.experiments.flexibility.CASES": cases,
            "src.optimization.experiments.flexibility_evaluation.VALIDATION_SCENARIOS": 3,
        }
    ):
        runner = ExperimentRunner(InstanceBuilder(ScenarioLayout(generated_root(ws)), version=VERSION), solver_overrides=SOLVER)
        runs = flexibility_module.run_experiment(
            VERSION,
            ["normal"],
            list(FLEXIBILITIES),
            list(cases),
            time_limit=600.0,
            mip_gap=0.0,
            output_root=results_root / "flexibility",
            runner=runner,
        )
        evaluations = evaluate_experiment(
            VERSION,
            ["normal"],
            list(FLEXIBILITIES),
            list(SOLUTION_CASES),
            time_limit=600.0,
            source=SourceRun("rG", results_root),
            output_root=results_root / "flexibility_evaluation",
            runner=runner,
        )
    out = {"runs": [_clean_payload(run, ws) for run in runs]}
    out["evaluations"] = [_clean_payload(item, ws) for item in evaluations]

    from src.optimization.models.uncapacitated import UncapacitatedSAAModel

    instance = build_instance(
        ws, continuous_x=True, flexibility="fixed_operation", N=2, scenario_set="optimization", use_euclidean_distance=True
    )
    model = UncapacitatedSAAModel(instance)
    model.set_params({"TimeLimit": 600, "MIPGap": 0.0, "OutputFlag": 0, **SOLVER})
    model.build()
    solve = model.solve()
    solve.pop("actual_run_time")
    out["uncapacitated"] = solve
    dump("g8.json", json.dumps(out, indent=2))
    return out


# ── G9: satellite capacity analysis ──────────────────────────────────────────


def g9_capacity(ws: Workspace) -> dict:
    from src.core.contract import ScenarioLayout
    from src.optimization.capacity.analysis import CapacityConfig, run_analysis

    out = ws.path("capacity")
    summary = run_analysis(out, ScenarioLayout(generated_root(ws)), VERSION, CapacityConfig("vG"))
    table = (out / "capacity.json").read_text()
    dump("g9_capacity.json", table)
    return {
        "capacity": sha_text(table),
        "peak_fleet": sha_text((out / "peak_fleet.csv").read_text()),
        "assignment": sha_text((out / "assignment.json").read_text()),
        "summary": summary["satellites"],
    }


PRODUCERS = {
    "g1_fit": g1_fit,
    "g2_recalibrate": g2_recalibrate,
    "g3_generate": g3_generate,
    "g4_compare": g4_compare,
    "g5_spatial": g5_spatial,
    "g6_reports": g6_reports,
    "g7_ca": g7_ca,
    "g8_solve": g8_solve,
    "g9_capacity": g9_capacity,
}

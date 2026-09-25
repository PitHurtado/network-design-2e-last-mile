"""One producer per golden. Each returns a JSON-serializable dict of the outputs it pins.

This is the only file that changes when the refactor moves an API: producers call the
code, the stored expectations describe what it must keep producing.
"""

import json
import shutil
from pathlib import Path

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
    out = ws.path("g1") / "shape_params.json"
    with patched(
        {
            "src.scenarios.cli.fit_params.PATH_SHAPE_PARAMS": out,
            "src.scenarios.fitting.panel.PATH_PANEL_MONTHLY": PANEL,
            "sys.argv": ["fit_params", "--n", "5"],
        }
    ):
        from src.scenarios.cli import fit_params

        fit_params.main()
    return _params_summary(json.loads(out.read_text()), "g1")


def g2_recalibrate(ws: Workspace) -> dict:
    out = ws.path("g2") / "shape_params.json"
    shutil.copy(PARAMS, out)
    with patched(
        {
            "src.scenarios.cli.recalibrate_regimes.PATH_SHAPE_PARAMS": out,
            "sys.argv": ["recalibrate_regimes", "--validation-n", "5"],
        }
    ):
        from src.scenarios.cli import recalibrate_regimes

        recalibrate_regimes.main()
    return _params_summary(json.loads(out.read_text()), "g2")


# ── G3 / G4: generation ───────────────────────────────────────────────────────


def generated_root(ws: Workspace) -> Path:
    """Scenario sets for every regime (n=3), generated once per session."""

    def build() -> Path:
        root = ws.path("scenarios")
        with patched(
            {
                "src.scenarios.cli.generate.PATH_SHAPE_PARAMS": PARAMS,
                "src.core.constants.PATH_GENERATED_SCENARIOS": root,
                "sys.argv": ["generate", "--all", "--version", VERSION, "--optimization-n", "3", "--validation-n", "3"],
            }
        ):
            from src.scenarios.cli import generate

            generate.main()
        return root

    return ws.once("generated", build)


def comparison_root(ws: Workspace) -> Path:
    """Paired comparison sets for `normal`, three methods, n=3."""

    def build() -> Path:
        from unittest import mock

        root = ws.path("comparison")
        with patched(
            {
                "src.scenarios.cli.compare.PATH_SHAPE_PARAMS": PARAMS,
                "src.scenarios.cli.compare.build_comparison_report": mock.MagicMock(return_value=Path("skipped")),
                "src.core.constants.PATH_COMPARISON_SCENARIOS": root,
                "src.scenarios.fitting.panel.PATH_PANEL_MONTHLY": PANEL,
                "sys.argv": ["compare", "--regime", "normal", "--version", VERSION, "--n", "3"],
            }
        ):
            from src.scenarios.cli import compare

            compare.main()
        return root

    return ws.once("comparison", build)


def g3_generate(ws: Workspace) -> dict:
    root = generated_root(ws) / VERSION
    return {"files": tree_digest(root), "manifests": _manifests(root)}


def g4_compare(ws: Workspace) -> dict:
    root = comparison_root(ws) / VERSION
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

    scenario_patches = {
        "src.scenarios.reports.PATH_SHAPE_PARAMS": PARAMS,
        "src.scenarios.fitting.panel.PATH_PANEL_MONTHLY": PANEL,
        "src.core.constants.PATH_GENERATED_SCENARIOS": generated,
    }
    with patched(scenario_patches), figure_capture() as figures:
        from src.scenarios.reports import build_validation_report
        from src.scenarios.validation.params import aggregate_cv_impact

        path, n_failed = build_validation_report(list(ALL_REGIMES), output_path=out_dir / "validation.html", version=VERSION)
        results["validation"] = {**_report_digest("validation", path, list(figures)), "n_failed": n_failed}
        results["aggregate_cv"] = aggregate_cv_impact(_load_params())

    with patched(scenario_patches), figure_capture() as figures:
        from src.scenarios.reports import build_explore_report

        path = build_explore_report(list(ALL_REGIMES), output_path=out_dir / "explore.html", version=VERSION)
        results["explore"] = _report_digest("explore", path, list(figures))

    with (
        patched(
            {
                "src.scenarios.reports.PATH_SHAPE_PARAMS": PARAMS,
                "src.scenarios.fitting.panel.PATH_PANEL_MONTHLY": PANEL,
                "src.core.constants.PATH_COMPARISON_SCENARIOS": comparison,
            }
        ),
        figure_capture() as figures,
    ):
        from src.scenarios.reports import build_comparison_report

        cmp_dir = ws.path("reports", "comparison")
        path = build_comparison_report("normal", version=VERSION, output_path=cmp_dir / "comparison.html")
        extra = sorted(p for p in cmp_dir.iterdir() if p.suffix in (".csv", ".json"))
        results["comparison"] = _report_digest("comparison", path, list(figures), extra)

    with figure_capture() as figures:
        from src.optimization.reports import build_flexibility_report as build_flex

        flex_dir = ws.path("reports", "flexibility")
        path = build_flex(FIXTURE_RESULTS_VERSION, output_path=flex_dir / "flex.html", root=FIXTURE_RESULTS / "flexibility")
        results["flexibility"] = _report_digest("flexibility", path, list(figures), [flex_dir / "summary.json"])

    with patched({"src.optimization.reports.RESULTS_DIR": FIXTURE_RESULTS}):
        from src.optimization.reports import build_evaluation_preview as build_partial_preview
        from src.optimization.reports import build_evaluation_report as build_vss

        root = FIXTURE_RESULTS / "flexibility_evaluation"
        with figure_capture() as figures:
            vss_dir = ws.path("reports", "vss")
            path = build_vss(FIXTURE_RESULTS_VERSION, output_path=vss_dir / "vss.html", root=root)
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
    from src.optimization.instance import Instance

    with patched({"src.core.constants.PATH_GENERATED_SCENARIOS": generated_root(ws)}):
        return Instance(
            id_instance="golden",
            is_continuous_var_x=continuous_x,
            type_of_flexibility=flexibility,
            regime=regime,
            scenario_version=VERSION,
            **case,
        )


def g7_ca(ws: Workspace) -> dict:
    return {label: _instance_digest(build_instance(ws, **case)) for label, case in INSTANCE_CASES.items()}


# ── G8: solves ────────────────────────────────────────────────────────────────


def _pinned_model():
    from src.optimization.models.flex import FlexSAAModel

    class PinnedFlexSAAModel(FlexSAAModel):
        def set_params(self, params):
            super().set_params({**params, **SOLVER})

    return PinnedFlexSAAModel


def _clean_payload(payload: dict, ws: Workspace) -> dict:
    payload = json.loads(normalize_ids(ws.relativize(json.dumps(payload))))
    if isinstance(payload.get("solve"), dict):
        payload["solve"].pop("actual_run_time", None)
    return payload


def g8_solve(ws: Workspace) -> dict:
    from src.optimization.experiments import flexibility as flexibility_module

    cases = {**flexibility_module.CASES, "optimization": {**flexibility_module.CASES["optimization"], "n_scenarios": 2}}
    results_root = ws.path("results")
    with patched(
        {
            "src.core.constants.PATH_GENERATED_SCENARIOS": generated_root(ws),
            "src.optimization.experiments.flexibility.FlexSAAModel": _pinned_model(),
            "src.optimization.experiments.flexibility.CASES": cases,
        }
    ):
        runs = flexibility_module.run_experiment(
            VERSION,
            ["normal"],
            list(FLEXIBILITIES),
            list(cases),
            time_limit=600.0,
            mip_gap=0.0,
            output_root=results_root / "flexibility",
        )
    out = {"runs": [_clean_payload(run, ws) for run in runs]}

    with patched(
        {
            "src.core.constants.PATH_GENERATED_SCENARIOS": generated_root(ws),
            "src.optimization.experiments.flexibility_evaluation.FlexSAAModel": _pinned_model(),
            "src.optimization.experiments.flexibility_evaluation.RESULTS_DIR": results_root,
            "src.optimization.experiments.flexibility_evaluation.VALIDATION_SCENARIOS": 3,
        }
    ):
        from src.optimization.experiments.flexibility_evaluation import SOLUTION_CASES, evaluate_experiment

        evaluations = evaluate_experiment(VERSION, ["normal"], list(FLEXIBILITIES), list(SOLUTION_CASES), time_limit=600.0)
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


PRODUCERS = {
    "g1_fit": g1_fit,
    "g2_recalibrate": g2_recalibrate,
    "g3_generate": g3_generate,
    "g4_compare": g4_compare,
    "g5_spatial": g5_spatial,
    "g6_reports": g6_reports,
    "g7_ca": g7_ca,
    "g8_solve": g8_solve,
}

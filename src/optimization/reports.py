"""Builders of the optimization-side reports: load persisted results, compute, hand them to a renderer.

The rendering lives in `src.visualization.results`. The derived tables are also persisted
next to each report (`summary.json`, `vss_summary.json`) for analysis outside the HTML.
"""

from pathlib import Path

from src.core.constants import RESULTS_DIR
from src.optimization.metrics import evaluation as ev
from src.optimization.metrics.flexibility import load_runs, run_tables
from src.tools.io import write_json
from src.visualization.results import evaluation_report, flexibility_report

EXPECTED_RUNS = 27
VALIDATION_SCENARIOS = 100


def build_flexibility_report(version: str, output_path: Path | None = None, root: Path | None = None) -> Path:
    """Comparison of the regime × policy × case runs of one version."""
    root = root or (RESULTS_DIR / "flexibility")
    output_path = output_path or (root / version / "flexibility_comparison.html")
    runs = load_runs(root, version)
    if not runs:
        raise FileNotFoundError(f"No result.json files under {root / version}")
    non_binary = [item["run"] for item in runs if item.get("assignment_variables") != "binary"]
    if non_binary:
        raise ValueError(
            "Flexibility results were generated with continuous or unknown X/W assignments. "
            "Rerun the experiment with --overwrite before building this report."
        )
    summary, decisions, operation = run_tables(runs)
    write_json(
        output_path.parent / "summary.json",
        {
            "version": version,
            "runs": summary.to_dict(orient="records"),
            "installation_decisions": decisions.to_dict(orient="records"),
            "operation_decisions": operation.to_dict(orient="records"),
        },
    )
    return flexibility_report.render(flexibility_report.FlexibilityReportData(version, summary, decisions), output_path)


def build_evaluation_preview(version: str, output_path: Path | None = None, root: Path | None = None) -> Path:
    """Progress preview while the fixed-Y evaluations are still running."""
    root = root or (RESULTS_DIR / "flexibility_evaluation")
    output_path = output_path or root / version / "vss_preview_partial.html"
    payloads = [item for item in ev.load_evaluations(root, version) if item.get("assignment_variables") == "binary"]
    if not payloads:
        raise FileNotFoundError("No binary evaluation files are available for a preview.")
    means, _, installations = ev.evaluation_frames(payloads)
    data = evaluation_report.EvaluationPreviewData(version, means, installations, len(payloads))
    return evaluation_report.render_preview(data, output_path)


def build_evaluation_report(
    version: str, output_path: Path | None = None, root: Path | None = None, benchmark_root: Path | None = None
) -> Path:
    """Out-of-sample VSS report over the 27 fixed-Y evaluations of one version."""
    root = root or (RESULTS_DIR / "flexibility_evaluation")
    benchmark_root = benchmark_root or (RESULTS_DIR / "flexibility_validation_benchmark")
    output_path = output_path or root / version / "vss_comparison.html"
    payloads = ev.load_evaluations(root, version)
    if len(payloads) != EXPECTED_RUNS:
        raise ValueError(f"Expected {EXPECTED_RUNS} evaluation files under {root / version}; found {len(payloads)}.")
    if any(payload.get("assignment_variables") != "binary" for payload in payloads):
        raise ValueError(
            "Evaluation files were generated with continuous or unknown X/W assignments. "
            "Rerun the flexibility experiment and evaluation with --overwrite first."
        )
    means, scenarios, installations = ev.evaluation_frames(payloads)
    if (
        len(scenarios) != EXPECTED_RUNS * VALIDATION_SCENARIOS
        or scenarios.groupby(["regime", "flexibility", "solution_case"]).size().nunique() != 1
    ):
        raise ValueError(f"Every evaluation must contain the same {VALIDATION_SCENARIOS} validation-scenario costs.")
    percentiles = ev.percentiles(scenarios)
    vss = ev.vss(percentiles)
    theoretical = ev.theoretical_vss(percentiles, ev.load_validation_benchmarks(benchmark_root, version))
    data = evaluation_report.EvaluationReportData(version, means, scenarios, installations, percentiles, vss, theoretical)
    path = evaluation_report.render(data, output_path)
    write_json(
        output_path.parent / "vss_summary.json",
        {
            "version": version,
            "means": means.to_dict("records"),
            "percentiles": percentiles.to_dict("records"),
            "vss_oos": vss.to_dict("records"),
            "vss_theoretical": theoretical.to_dict("records"),
            "installations": installations.to_dict("records"),
            "scenario_costs": scenarios.to_dict("records"),
        },
    )
    return path

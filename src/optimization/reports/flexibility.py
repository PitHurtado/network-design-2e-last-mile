"""Comparative HTML report for the flexibility experiment."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.core.constants import RESULTS_DIR

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"
PLOTLY_CFG = {"displayModeBar": True, "displaylogo": False}
CSS = """
body { font-family: sans-serif; margin: 0; padding: 0; background: #f4f6f8; }
#top-bar { background: #2c3e50; padding: 12px 24px; }
#top-bar h1 { color: white; margin: 0; font-size: 1.3em; }
#top-bar p { color: #aaa; margin: 4px 0 0; font-size: .85em; }
.section { background: white; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,.08); margin: 20px auto; max-width: 1440px; padding: 20px; }
.insight { background: #eef6fb; border-left: 4px solid #3498db; margin-top: 12px; padding: 10px 12px; }
.method { background: #f0fdf4; border-left: 4px solid #27ae60; margin: 12px 0; padding: 10px 12px; }
table { border-collapse: collapse; font-size: .8em; width: 100%; } th, td { border: 1px solid #ddd; padding: 6px; text-align: right; } th { background: #f4f6f8; }
"""


def to_html(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, config=PLOTLY_CFG)


def section(title: str, subtitle: str, body: str, insight: str = "", method: str = "") -> str:
    return f'<div class="section"><h2>{title}</h2><p>{subtitle}</p>{f"<div class=\"method\">📐 {method}</div>" if method else ""}{body}{f"<div class=\"insight\">💡 {insight}</div>" if insight else ""}</div>'


def _load(root: Path, version: str) -> list[dict]:
    runs = []
    for path in sorted((root / version).rglob("result.json")):
        payload = json.loads(path.read_text())
        if payload.get("status") != "ERROR":
            payload["_path"] = str(path)
            runs.append(payload)
    return runs


def _tables(runs: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary, decisions, operation = [], [], []
    for item in runs:
        run, costs, solve = item["run"], item["costs"], item["solve"]
        installed = item["decisions"]["installation"]
        summary.append(
            {
                **run,
                "objective": solve["objective_value"],
                "installation_cost": costs["installation"],
                "operation_cost": costs["operation_expected"],
                "routing_cost": costs["routing_facilities_expected"] + costs["routing_dc_expected"],
                "installed_satellites": sum(row["installed"] for row in installed),
                "installed_capacity": sum(row["capacity"] for row in installed),
                "status": solve["status"],
                "optimality_gap_pct": solve["optimality_gap"],
                "is_optimal": solve["is_optimal"],
            }
        )
        decisions.extend([{**run, **row} for row in installed])
        operation.extend([{**run, **row} for row in item["decisions"]["operation"]])
    return pd.DataFrame(summary), pd.DataFrame(decisions), pd.DataFrame(operation)


def _fig_costs(summary: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    x = [f"{r.regime}<br>{r.flexibility}<br>{r.case}" for r in summary.itertuples()]
    for column, label, color in (
        ("installation_cost", "Instalación", "#2c3e50"),
        ("operation_cost", "Operación", "#3498db"),
        ("routing_cost", "Ruteo esperado", "#e67e22"),
    ):
        fig.add_trace(go.Bar(x=x, y=summary[column], name=label, marker_color=color))
    fig.update_layout(
        barmode="stack", height=460, margin=dict(l=60, r=20, t=35, b=110), yaxis_title="Costo", plot_bgcolor="white"
    )
    return fig


def _fig_decisions(decisions: pd.DataFrame) -> go.Figure:
    columns = decisions.apply(lambda r: f"{r['regime']}|{r['flexibility']}|{r['case']}", axis=1)
    matrix = decisions.assign(column=columns).pivot(index="facility", columns="column", values="capacity").sort_index()
    fig = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(),
            x=matrix.columns,
            y=matrix.index,
            colorscale="Blues",
            colorbar_title="Capacidad instalada",
            hovertemplate="%{y}<br>%{x}<br>capacidad %{z}<extra></extra>",
        )
    )
    fig.update_layout(height=440, margin=dict(l=130, r=20, t=25, b=110), xaxis_tickangle=-45)
    return fig


def _fig_flex_value(summary: pd.DataFrame) -> go.Figure:
    fixed = summary[summary["flexibility"] == "fixed_operation"].set_index(["regime", "case"])["objective"]
    rows = summary[summary["flexibility"] != "fixed_operation"].copy()
    rows["delta"] = [row.objective - fixed[(row.regime, row.case)] for row in rows.itertuples()]
    fig = go.Figure()
    for flexibility, group in rows.groupby("flexibility"):
        fig.add_trace(go.Bar(x=[f"{r.regime}<br>{r.case}" for r in group.itertuples()], y=group["delta"], name=flexibility))
    fig.update_layout(
        barmode="group",
        height=400,
        margin=dict(l=60, r=20, t=35, b=80),
        yaxis_title="Δ objetivo/incumbente vs fixed_operation",
        plot_bgcolor="white",
    )
    return fig


def build_report(version: str, output_path: Path | None = None, root: Path | None = None) -> Path:
    root = root or (RESULTS_DIR / "flexibility")
    output_path = output_path or (root / version / "flexibility_comparison.html")
    runs = _load(root, version)
    if not runs:
        raise FileNotFoundError(f"No result.json files under {root / version}")
    non_binary = [item["run"] for item in runs if item.get("assignment_variables") != "binary"]
    if non_binary:
        raise ValueError(
            "Flexibility results were generated with continuous or unknown X/W assignments. "
            "Rerun the experiment with --overwrite before building this report."
        )
    summary, decisions, operation = _tables(runs)
    expected_runs = 27
    coverage = f"{len(summary)}/{expected_runs} corridas disponibles"
    best = summary.loc[summary["objective"].idxmin()]
    capacity_range = f"{decisions.groupby(['regime', 'flexibility', 'case'])['capacity'].sum().min():.0f}–{decisions.groupby(['regime', 'flexibility', 'case'])['capacity'].sum().max():.0f}"
    unresolved = summary[~summary["is_optimal"]]
    solve_insight = (
        "Todas las corridas disponibles están certificadas como óptimas."
        if unresolved.empty
        else f"{len(unresolved)} corridas alcanzaron el límite de tiempo: su brecha máxima es {unresolved['optimality_gap_pct'].max():.3f}% y sus objetivos se muestran como incumbentes factibles, no como óptimos certificados."
    )
    table = summary.sort_values(["regime", "case", "flexibility"]).to_html(index=False, float_format=lambda x: f"{x:,.2f}")
    html = f"""<!DOCTYPE html><html lang="es"><head><meta charset="utf-8"><title>Flexibilidad — {version}</title><script src="{PLOTLY_CDN}"></script><style>{CSS}</style></head><body><div id="top-bar"><h1>Experimento de flexibilidad — {version}</h1><p>{coverage} · 3 regímenes · 3 políticas · expected, optimization y annual_expected</p></div>{section('Costo total y componentes', 'Cada barra es una corrida; instalación no se promedia, operación y ruteo sí.', to_html(_fig_costs(summary)), f"La menor corrida disponible es {best.regime} / {best.flexibility} / {best.case}: {best.objective:,.0f}. {solve_insight}", method='Los componentes se leen directamente del modelo resuelto; operación y ruteo se dividen por el número de escenarios.')}{section('Decisiones de instalación', 'Capacidad elegida para cada satélite; cero significa no instalar.', to_html(_fig_decisions(decisions)), f"La capacidad total instalada varía entre {capacity_range} vehículos entre corridas.", method='Y[i,q] selecciona exactamente un nivel q, incluido q=0.')}{section('Valor de la flexibilidad', 'Cambio de objetivo/incumbente respecto a operar siempre a la capacidad instalada.', to_html(_fig_flex_value(summary)), 'Valores negativos indican ahorro frente a la política rígida; interpretar diferencias menores que la brecha reportada con cautela.', method='Comparación dentro del mismo régimen y caso de escenarios.')}{section('Tabla auditable', 'Una fila por corrida, con componentes de costo, capacidad, estado y brecha del solver.', table, method='Los JSON detallados conservan además la operación por período y escenario.')}</body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    (output_path.parent / "summary.json").write_text(
        json.dumps(
            {
                "version": version,
                "runs": summary.to_dict(orient="records"),
                "installation_decisions": decisions.to_dict(orient="records"),
                "operation_decisions": operation.to_dict(orient="records"),
            },
            indent=2,
        )
    )
    output_path.write_text(html)
    return output_path

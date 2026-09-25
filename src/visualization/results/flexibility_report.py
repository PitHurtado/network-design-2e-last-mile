"""Comparative HTML report of the flexibility experiment (27 runs: regimes × policies × cases)."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from src.visualization.html import BASE_CSS, fig_html, page, section, top_bar

CSS = BASE_CSS + """
.section { margin: 20px auto; max-width: 1440px; }
table { border-collapse: collapse; font-size: .8em; width: 100%; } th, td { border: 1px solid #ddd; padding: 6px; text-align: right; }
th { background: #f4f6f8; }
"""


@dataclass
class FlexibilityReportData:
    """Run and installation tables built by `src.optimization.reports`."""

    version: str
    summary: pd.DataFrame
    decisions: pd.DataFrame


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


def render(data: FlexibilityReportData, output_path: Path) -> Path:
    version, summary, decisions = data.version, data.summary, data.decisions
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
    sections = [
        section(
            "Costo total y componentes",
            "Cada barra es una corrida; instalación no se promedia, operación y ruteo sí.",
            fig_html(_fig_costs(summary)),
            f"La menor corrida disponible es {best.regime} / {best.flexibility} / {best.case}: {best.objective:,.0f}. {solve_insight}",
            method="Los componentes se leen directamente del modelo resuelto; operación y ruteo se dividen por el número de escenarios.",
            labelled=False,
        ),
        section(
            "Decisiones de instalación",
            "Capacidad elegida para cada satélite; cero significa no instalar.",
            fig_html(_fig_decisions(decisions)),
            f"La capacidad total instalada varía entre {capacity_range} vehículos entre corridas.",
            method="Y[i,q] selecciona exactamente un nivel q, incluido q=0.",
            labelled=False,
        ),
        section(
            "Valor de la flexibilidad",
            "Cambio de objetivo/incumbente respecto a operar siempre a la capacidad instalada.",
            fig_html(_fig_flex_value(summary)),
            "Valores negativos indican ahorro frente a la política rígida; interpretar diferencias menores que la brecha reportada con cautela.",
            method="Comparación dentro del mismo régimen y caso de escenarios.",
            labelled=False,
        ),
        section(
            "Tabla auditable",
            "Una fila por corrida, con componentes de costo, capacidad, estado y brecha del solver.",
            table,
            method="Los JSON detallados conservan además la operación por período y escenario.",
            labelled=False,
        ),
    ]
    header = top_bar(
        f"Experimento de flexibilidad — {version}",
        f"{coverage} · 3 regímenes · 3 políticas · expected, optimization y annual_expected",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page(f"Flexibilidad — {version}", header + "".join(sections), css=CSS))
    return output_path

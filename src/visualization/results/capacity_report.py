"""Satellite capacity report: peak-fleet distribution, the chosen levels and their costs."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from src.visualization.html import BASE_CSS, fig_html, page, section, top_bar

MONTHS = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]


@dataclass
class CapacityReportData:
    label: str
    method: str
    alpha_fixed: float
    satellites: dict  # capacity.json "satellites"
    peaks: pd.DataFrame  # regime, scenario, satellite, peak_fleet


def _fig_levels(data: CapacityReportData) -> go.Figure:
    fig = go.Figure()
    order = sorted(data.satellites, key=lambda s: data.satellites[s]["peak_fleet"]["p50"])
    for satellite in order:
        values = data.peaks.loc[data.peaks["satellite"] == satellite, "peak_fleet"]
        fig.add_trace(go.Box(y=values, name=satellite, marker_color="#3498db", boxpoints=False, showlegend=False))
        levels = [q for q in data.satellites[satellite]["levels"] if q > 0]
        fig.add_trace(
            go.Scatter(
                x=[satellite] * len(levels),
                y=levels,
                mode="markers",
                marker=dict(symbol="line-ew-open", size=26, color="#e67e22", line=dict(width=2)),
                name="niveles",
                showlegend=satellite == order[0],
                hovertemplate="nivel %{y}<extra></extra>",
            )
        )
    fig.update_layout(height=460, yaxis_title="Flota pico (vehículos)", plot_bgcolor="white", margin=dict(l=60, r=20, t=30, b=60))
    fig.update_yaxes(gridcolor="#eee")
    return fig


def _fig_seasonality(data: CapacityReportData) -> go.Figure:
    satellites = sorted(data.satellites)
    fig = go.Figure(
        go.Heatmap(
            z=[data.satellites[s]["seasonal_factor"] for s in satellites],
            x=MONTHS,
            y=satellites,
            colorscale="RdBu",
            reversescale=True,
            zmid=1.0,
            colorbar_title="factor",
            hovertemplate="%{y} · %{x}: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(height=380, margin=dict(l=140, r=20, t=20, b=40))
    return fig


def _table(data: CapacityReportData) -> str:
    rows = []
    for satellite, block in sorted(data.satellites.items()):
        for q in block["levels"][1:]:
            rows.append(
                {
                    "Satélite": satellite,
                    "Píxeles": block["n_pixels"],
                    "Nivel": q,
                    "Cobertura %": block["coverage_pct"][str(q)],
                    "Instalación": block["cost_installation"][str(q)],
                    "Operación media/período": sum(block["cost_operation"][str(q)]) / 12,
                }
            )
    return pd.DataFrame(rows).to_html(index=False, float_format=lambda x: f"{x:,.1f}")


def render(data: CapacityReportData, output_path: Path) -> Path:
    n_scenarios = data.peaks.groupby(["regime", "scenario"]).ngroups
    widest = max(data.satellites, key=lambda s: data.satellites[s]["peak_fleet"]["p95"])
    body = "".join(
        [
            section(
                "Flota pico y niveles de capacidad",
                "Distribución, sobre los escenarios de capacidad, de la flota máxima del año que necesita cada satélite; "
                "las marcas son los niveles instalables.",
                fig_html(_fig_levels(data)),
                f"El satélite de mayor exigencia es {widest}: P95 = {data.satellites[widest]['peak_fleet']['p95']:.1f} "
                f"vehículos. Cada nivel máximo se valida para cubrir al menos el 95% de los picos.",
                method=f"Cada píxel se asigna al satélite más cercano (haversine); la CA da la flota de vans por "
                f"píxel-período; la flota de un satélite es el techo de la suma de sus píxeles y el pico es el máximo "
                f"de los 12 períodos. {n_scenarios} escenarios (regímenes juntos). Método de niveles: {data.method}.",
            ),
            section(
                "Estacionalidad del costo de operación",
                "Demanda de los píxeles de cada satélite por mes, relativa a su media anual (escenario esperado).",
                fig_html(_fig_seasonality(data)),
                f"El costo de operación del nivel q en el mes t es OPEX(q) × [{data.alpha_fixed:.2f} + "
                f"{1 - data.alpha_fixed:.2f} × factor(t)]: la parte fija no depende de la demanda.",
                method="OPEX e instalación por nivel desde data/raw_facility/tariffs.json, con extrapolación por encima "
                "del mayor nivel conocido.",
            ),
            section("Tabla de capacidad", "Una fila por satélite y nivel instalable.", _table(data)),
        ]
    )
    header = top_bar(
        f"Análisis de capacidad de satélites — {data.label}", f"{len(data.satellites)} satélites · niveles {data.method}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page(f"Capacidad — {data.label}", header + body, css=BASE_CSS))
    return output_path

"""Interactive technical explorer for generated demand regimes and scenarios."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.core.constants import GRID_DLAT, GRID_DLON, N_PERIODS
from src.visualization.components.labels import REGIME_COLORS
from src.visualization.html import BASE_CSS, PLOTLY_CDN, fig_html, section


@dataclass
class ExploreReportData:
    """Validation sets of the three regimes, with the statistics `src.scenarios.reports` computed."""

    regimes: list[str]
    summaries: dict[str, pd.DataFrame]  # per pixel, with layer / lon / lat / n_cells
    period_metrics: pd.DataFrame
    pairwise_correlations: dict[str, np.ndarray]
    n_scenarios: int


SEQUENTIAL_BLUE = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]
METRICS = {
    "demand": ("mean_demand_period", "Demanda media por período"),
    "stop": ("mean_stop", "Stops medios por período"),
    "drop": ("mean_drop", "Drop medio (items por stop)"),
}


def _pixel_rect(lon: float, lat: float, n_cells: int) -> tuple[list, list]:
    side = math.sqrt(n_cells)
    half_dlon, half_dlat = side * GRID_DLON / 2, side * GRID_DLAT / 2
    return [lon - half_dlon, lon + half_dlon, lon + half_dlon, lon - half_dlon, lon - half_dlon, None], [
        lat - half_dlat,
        lat - half_dlat,
        lat + half_dlat,
        lat + half_dlat,
        lat - half_dlat,
        None,
    ]


def _hover_texts(view: pd.DataFrame, normal: pd.DataFrame) -> tuple[list[str], list[str]]:
    basic, detailed = [], []
    for id_pixel, row in view.iterrows():
        baseline = normal.loc[id_pixel]
        deltas = {
            label: (row[column] / baseline[column] - 1) * 100
            for column, label in (("mean_demand_period", "demanda"), ("mean_stop", "stops"), ("mean_drop", "drop"))
        }
        basic.append(
            f"<b>{id_pixel}</b> · layer {row['layer']}<br>Demanda: {row['mean_demand_period']:,.1f} · stops: {row['mean_stop']:.1f} · drop: {row['mean_drop']:.2f}"
        )
        detailed.append(
            basic[-1]
            + "<br>Vs normal: "
            + " · ".join(f"{label} {value:+.1f}%" for label, value in deltas.items())
            + f"<br>CV entre escenarios — demanda: {row['cv_demand']:.2f}; stops: {row['cv_stop']:.2f}; drop: {row['cv_drop']:.2f}"
        )
    return basic, detailed


def build_grid_figure(view: pd.DataFrame, normal: pd.DataFrame, column: str, label: str, cmin: float, cmax: float):
    values = view[column].to_numpy(dtype=float)
    colors = pc.sample_colorscale(SEQUENTIAL_BLUE, ((values - cmin) / (cmax - cmin or 1.0)).clip(0, 1).tolist())
    traces, hover_x, hover_y = [], [], []
    for (_, row), color in zip(view.iterrows(), colors):
        x, y = _pixel_rect(row["lon"], row["lat"], row["n_cells"])
        traces.append(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                fill="toself",
                fillcolor=color,
                line=dict(color=color, width=0.5),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        hover_x.append(row["lon"])
        hover_y.append(row["lat"])
    basic, detailed = _hover_texts(view, normal)
    traces.append(
        go.Scatter(
            x=hover_x,
            y=hover_y,
            mode="markers",
            marker=dict(
                size=14,
                opacity=0.001,
                color=values,
                colorscale=SEQUENTIAL_BLUE,
                cmin=cmin,
                cmax=cmax,
                showscale=True,
                colorbar=dict(title=dict(text=label, font=dict(size=10))),
            ),
            text=basic,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )
    fig = go.Figure(traces)
    fig.update_layout(
        xaxis=dict(title="Longitud", showgrid=True, zeroline=False, dtick=GRID_DLON, gridcolor="#e1e0d9"),
        yaxis=dict(
            title="Latitud", showgrid=True, zeroline=False, dtick=GRID_DLAT, gridcolor="#e1e0d9", scaleanchor="x", scaleratio=1
        ),
        margin=dict(l=50, r=20, t=35, b=40),
        height=460,
        plot_bgcolor="white",
    )
    return fig, basic, detailed


def fig_regime_distribution(metrics: pd.DataFrame, regimes: list[str]) -> go.Figure:
    fields = [("stops", "Stops totales"), ("drop", "Drop ponderado (items/stop)"), ("demand", "Demanda total")]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[label for _, label in fields])
    for col, (field, label) in enumerate(fields, 1):
        for regime in regimes:
            fig.add_trace(
                go.Box(
                    y=metrics.loc[metrics["regime"] == regime, field],
                    name=regime,
                    marker_color=REGIME_COLORS[regime],
                    boxpoints="all",
                    jitter=0.25,
                    pointpos=0,
                    legendgroup=regime,
                    showlegend=col == 1,
                ),
                row=1,
                col=col,
            )
        fig.update_yaxes(title_text=label, gridcolor="#eee", row=1, col=col)
    fig.update_layout(height=440, margin=dict(l=60, r=20, t=55, b=45), plot_bgcolor="white", legend=dict(orientation="h", y=1.16))
    return fig


def fig_component_cv(summaries: dict[str, pd.DataFrame], regimes: list[str]) -> go.Figure:
    fields = [("cv_demand", "Demanda anual"), ("cv_stop", "Stops anuales"), ("cv_drop", "Drop medio")]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[label for _, label in fields])
    for col, (field, label) in enumerate(fields, 1):
        for regime in regimes:
            fig.add_trace(
                go.Box(
                    y=summaries[regime][field],
                    name=regime,
                    marker_color=REGIME_COLORS[regime],
                    legendgroup=regime,
                    showlegend=False,
                ),
                row=1,
                col=col,
            )
        fig.update_yaxes(title_text=f"CV de {label.lower()}", gridcolor="#eee", row=1, col=col)
    fig.update_layout(height=390, margin=dict(l=60, r=20, t=55, b=45), plot_bgcolor="white")
    return fig


def fig_regime_bands(metrics: pd.DataFrame, regimes: list[str]) -> go.Figure:
    fig = make_subplots(rows=1, cols=3, subplot_titles=["Stops totales", "Drop ponderado", "Demanda total"])
    for col, field in enumerate(("stops", "drop", "demand"), 1):
        for regime in regimes:
            series = metrics[metrics["regime"] == regime].groupby("period")[field]
            q10, median, q90 = series.quantile(0.1), series.median(), series.quantile(0.9)
            rgb = ",".join(map(str, pc.hex_to_rgb(REGIME_COLORS[regime])))
            fig.add_trace(
                go.Scatter(
                    x=q10.index, y=q10, mode="lines", line=dict(width=0), hoverinfo="skip", legendgroup=regime, showlegend=False
                ),
                row=1,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=q90.index,
                    y=q90,
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=f"rgba({rgb},0.15)",
                    hoverinfo="skip",
                    legendgroup=regime,
                    showlegend=False,
                ),
                row=1,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=median.index,
                    y=median,
                    mode="lines+markers",
                    line=dict(color=REGIME_COLORS[regime], width=2),
                    name=regime,
                    legendgroup=regime,
                    showlegend=col == 1,
                ),
                row=1,
                col=col,
            )
        fig.update_xaxes(title_text="Período", dtick=1, gridcolor="#eee", row=1, col=col)
        fig.update_yaxes(gridcolor="#eee", row=1, col=col)
    fig.update_layout(height=400, margin=dict(l=60, r=20, t=55, b=45), plot_bgcolor="white", legend=dict(orientation="h", y=1.16))
    return fig


def fig_pairwise_corr(correlations: dict[str, np.ndarray], regimes: list[str]) -> go.Figure:
    fig = go.Figure()
    for regime in regimes:
        fig.add_trace(
            go.Histogram(
                x=correlations[regime],
                name=regime,
                marker_color=REGIME_COLORS[regime],
                opacity=0.6,
                histnorm="probability density",
                nbinsx=40,
            )
        )
    fig.update_layout(
        height=380,
        margin=dict(l=60, r=20, t=30, b=50),
        barmode="overlay",
        xaxis_title="Correlación entre escenarios (demanda anual por píxel)",
        yaxis_title="Densidad",
        legend=dict(orientation="h", y=1.12),
        plot_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="#eee")
    fig.update_yaxes(gridcolor="#eee")
    return fig


CSS_EXTRA = CSS_EXTRA = """
#top-bar { position: sticky; top: 0; z-index: 100; display: flex; align-items: center; gap: 20px; flex-wrap: wrap; }
#top-bar label { color: white; font-size: .9em; font-weight: bold; } #top-bar select { font-size: .95em; padding: 4px 8px; border-radius: 4px; border: none; }
#top-bar input[type=checkbox] { transform: scale(1.2); margin-right: 4px; } .combo-view { display: none; }
.grid-triptych { display: grid; grid-template-columns: repeat(3, minmax(280px, 1fr)); gap: 12px; } .grid-triptych h3 { text-align: center; margin: 8px 0 0; text-transform: capitalize; }
"""


def render(data: ExploreReportData, output_path: Path) -> Path:
    regimes = data.regimes
    summaries = data.summaries
    layers, n_pixels = sorted(summaries["normal"]["layer"].unique()), len(summaries["normal"])
    period_metrics = data.period_metrics
    n_scenarios = data.n_scenarios
    hover_store, combo_html = {}, {}
    for metric_key, (column, label) in METRICS.items():
        for layer in layers:
            key = f"{metric_key}_{layer}"
            subset = {regime: summaries[regime][summaries[regime]["layer"] == layer] for regime in regimes}
            values = pd.concat([view[column] for view in subset.values()])
            cards, hover_store[key] = [], {}
            for regime in regimes:
                fig, basic, detailed = build_grid_figure(
                    subset[regime], summaries["normal"], column, label, float(values.min()), float(values.max())
                )
                cards.append(f"<div><h3>{regime}</h3>{fig_html(fig)}</div>")
                hover_store[key][regime] = {"basic": basic, "detailed": detailed}
            combo_html[key] = '<div class="grid-triptych">' + "".join(cards) + "</div>"
    normal_median = period_metrics[period_metrics["regime"] == "normal"][["stops", "drop", "demand"]].median()
    comparison_insight, cv_insight = [], []
    for regime in regimes:
        medians = period_metrics[period_metrics["regime"] == regime][["stops", "drop", "demand"]].median()
        comparison_insight.append(
            f"{regime}: stops {medians['stops']:,.0f} ({(medians['stops'] / normal_median['stops'] - 1) * 100:+.1f}%), drop {medians['drop']:.2f} ({(medians['drop'] / normal_median['drop'] - 1) * 100:+.1f}%), demanda {medians['demand']:,.0f} ({(medians['demand'] / normal_median['demand'] - 1) * 100:+.1f}%)"
        )
        cv_insight.append(
            f"{regime}: CV mediano demanda {summaries[regime]['cv_demand'].median():.2f}, stops {summaries[regime]['cv_stop'].median():.2f}, drop {summaries[regime]['cv_drop'].median():.2f}"
        )
    corr_insight = "; ".join(f"{regime}: media {np.mean(data.pairwise_correlations[regime]):.2f}" for regime in regimes)
    metric_select = "".join(f'<option value="{key}">{label}</option>' for key, (_, label) in METRICS.items())
    layer_select = "".join(f'<option value="{layer}">{layer}</option>' for layer in layers)
    js = f"""const HOVER={json.dumps(hover_store)}, REGIMES={json.dumps(regimes)}; function currentKey(){{return document.getElementById('sel-metric').value+'_'+document.getElementById('sel-layer').value;}} function applyDetail(key){{const entry=HOVER[key],checked=document.getElementById('chk-detail').checked;if(!entry)return;document.querySelectorAll('.combo-view[data-key="'+key+'"] .js-plotly-plot').forEach(function(el,i){{const value=entry[REGIMES[i]];Plotly.restyle(el,{{text:[checked?value.detailed:value.basic]}},[el.data.length-1]);}});}} function updateView(){{const key=currentKey();document.querySelectorAll('.combo-view').forEach(el=>el.style.display='none');const active=document.querySelector('.combo-view[data-key="'+key+'"]');if(active){{active.style.display='block';active.querySelectorAll('.js-plotly-plot').forEach(el=>Plotly.Plots.resize(el));}}applyDetail(key);}}document.addEventListener('DOMContentLoaded',updateView);"""
    grid_views = "\n".join(f'<div class="combo-view" data-key="{key}">{html}</div>' for key, html in combo_html.items())
    sections = [
        section(
            "Comparación espacial por régimen",
            "Los tres mapas usan la misma escala para la métrica y layer seleccionados; normal es la referencia de los deltas del hover.",
            f'<div id="grid-views">{grid_views}</div>',
            "La escala se calcula sobre los tres regímenes del layer, por lo que los colores son directamente comparables.",
            method="Promedios de los escenarios simulados y 12 períodos; cada rectángulo representa un píxel.",
        ),
        section(
            "Nivel operativo por régimen",
            "Distribución de cada combinación escenario × período.",
            fig_html(fig_regime_distribution(period_metrics, regimes)),
            "; ".join(comparison_insight),
            method="Stops = Σ stop; drop ponderado = Σ demand / Σ stop; demanda = Σ demand.",
        ),
        section(
            "Variabilidad por píxel entre escenarios",
            "CV por píxel, separado para demanda, stops y drop.",
            fig_html(fig_component_cv(summaries, regimes)),
            "; ".join(cv_insight),
            method="Demanda y stops se agregan anualmente; drop es el promedio de los 12 períodos de cada escenario.",
        ),
        section(
            "Evolución por período",
            "Mediana y banda p10–p90 de los escenarios en cada período.",
            fig_html(fig_regime_bands(period_metrics, regimes)),
            "Las bandas muestran incertidumbre intra-régimen; la separación de niveles muestra el efecto del régimen.",
            method="Cuantiles sobre escenarios simulados, por período y régimen.",
        ),
        section(
            "Estructura espacial entre escenarios",
            "Correlación de la demanda anual por píxel entre pares de escenarios del mismo régimen.",
            fig_html(fig_pairwise_corr(data.pairwise_correlations, regimes)),
            corr_insight,
            method="Correlación de Pearson entre vectores anuales de los píxeles.",
        ),
    ]
    html = f"""<!DOCTYPE html><html lang="es"><head><meta charset="utf-8"><title>Explorador comparativo de escenarios</title><script src="{PLOTLY_CDN}"></script><style>{BASE_CSS}{CSS_EXTRA}</style><script>{js}</script></head><body><div id="top-bar"><h1 style="margin:0;">Explorador comparativo de escenarios</h1><label>Métrica: <select id="sel-metric" onchange="updateView()">{metric_select}</select></label><label>Layer: <select id="sel-layer" onchange="updateView()">{layer_select}</select></label><label><input type="checkbox" id="chk-detail" onchange="applyDetail(currentKey())">Mostrar detalle en hover</label><p style="width:100%; margin:4px 0 0;">{n_pixels} píxeles · {N_PERIODS} períodos · {len(regimes)} regímenes × {n_scenarios} escenarios.</p></div>{''.join(sections)}</body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    return output_path

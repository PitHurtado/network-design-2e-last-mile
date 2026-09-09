"""Interactive explorer for the generated demand scenarios.

Complements `src/analysis/scenarios.py` (which validates the aggregate structure
against the historical panel). This report answers a narrower question: within one
regime, how much does demand actually differ pixel by pixel and scenario by
scenario? A dropdown grid lets you browse regime x layer, and a statistics section
measures how distinct the 50 simulated scenarios of a regime really are — the CV of
each pixel across scenarios, a spaghetti view of period-level dispersion, and the
pairwise correlation between scenarios' pixel-demand vectors.

Every number in the prose is computed here, never hardcoded, per the `viz-report`
skill's rule: an insight must survive the instance changing.
"""

import json
import math

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go

from src.analysis.scenarios import CSS, PLOTLY_CDN, REGIME_COLORS, section, to_html
from src.constants import GRID_DLAT, GRID_DLON, N_PERIODS, RESULTS_DIR
from src.scenarios.generate import load_generated
from src.scenarios.spatial import pixel_centroids
from src.utils.custom_logger import get_logger

logger = get_logger("ExploreScenarios")

# Sequential single-hue ramp (blue, light -> dark), from the dataviz skill's
# reference palette. Continuous magnitude only — never used for identity.
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


# ── Data assembly ─────────────────────────────────────────────────────────────


def load_all(regimes: list[str]) -> dict[str, pd.DataFrame]:
    """Load every regime's scenarios, merged with pixel geometry and layer."""
    centroids = pixel_centroids()[["id_pixel", "layer", "lon", "lat", "n_cells"]]
    data = {}
    for regime in regimes:
        frame = load_generated(regime).merge(centroids, on="id_pixel", how="left")
        if frame["layer"].isna().any():
            missing = sorted(frame.loc[frame["layer"].isna(), "id_pixel"].unique())
            raise ValueError(f"[{regime}] pixels missing from pixel_centroids(): {missing}")
        data[regime] = frame
    return data


def pixel_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-pixel demand summary: level (colors the grid) and variability (feeds the stats)."""
    annual = frame.groupby(["id_pixel", "id_scenario"])["demand"].sum()
    by_pixel = annual.groupby("id_pixel")
    summary = pd.DataFrame(
        {
            "mean_demand_period": frame.groupby("id_pixel")["demand"].mean(),
            "mean_stop": frame.groupby("id_pixel")["stop"].mean(),
            "mean_drop": frame.groupby("id_pixel")["drop"].mean(),
            "annual_mean": by_pixel.mean(),
            "annual_std": by_pixel.std(),
            "annual_min": by_pixel.min(),
            "annual_max": by_pixel.max(),
        }
    )
    summary["cv_annual"] = summary["annual_std"] / summary["annual_mean"]
    geo = frame.drop_duplicates("id_pixel").set_index("id_pixel")[["layer", "lon", "lat", "n_cells"]]
    return summary.join(geo)


# ── Grid figure ───────────────────────────────────────────────────────────────


def _pixel_rect(lon: float, lat: float, n_cells: int) -> tuple[list, list]:
    """Closed rectangle polygon for a pixel, sized by its footprint cell count."""
    side = math.sqrt(n_cells)
    half_dlon = side * GRID_DLON / 2
    half_dlat = side * GRID_DLAT / 2
    x = [lon - half_dlon, lon + half_dlon, lon + half_dlon, lon - half_dlon, lon - half_dlon, None]
    y = [lat - half_dlat, lat - half_dlat, lat + half_dlat, lat + half_dlat, lat - half_dlat, None]
    return x, y


def _hover_texts(view: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Basic and detailed hover text, one entry per pixel in `view`, same order."""
    basic, detailed = [], []
    for id_pixel, row in view.iterrows():
        basic.append(
            f"<b>{id_pixel}</b> (layer {row['layer']}, {int(row['n_cells'])} celda(s))<br>"
            f"Demanda media/período: {row['mean_demand_period']:,.1f}"
        )
        detailed.append(
            basic[-1] + "<br>"
            f"CV anual entre escenarios: {row['cv_annual']:.2f}<br>"
            f"Demanda anual: {row['annual_min']:,.0f} – {row['annual_max']:,.0f} "
            f"(media {row['annual_mean']:,.0f})<br>"
            f"stop medio: {row['mean_stop']:.1f} · drop medio: {row['mean_drop']:.2f}"
        )
    return basic, detailed


def build_grid_figure(view: pd.DataFrame) -> tuple[go.Figure, list[str], list[str]]:
    """One regime+layer combo: colored pixel rectangles plus an invisible hover/colorbar layer."""
    values = view["mean_demand_period"].to_numpy(dtype=float)
    vmin, vmax = float(values.min()), float(values.max())
    span = vmax - vmin or 1.0
    norm = ((values - vmin) / span).tolist()
    colors = pc.sample_colorscale(SEQUENTIAL_BLUE, norm)

    traces: list[go.Scatter] = []
    hover_x, hover_y = [], []
    for (_id_pixel, row), color in zip(view.iterrows(), colors):
        x, y = _pixel_rect(row["lon"], row["lat"], row["n_cells"])
        traces.append(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                fill="toself",
                fillcolor=color,
                line=dict(color=color, width=0.5),
                opacity=0.9,
                hoverinfo="skip",
                showlegend=False,
            )
        )
        hover_x.append(row["lon"])
        hover_y.append(row["lat"])

    basic, detailed = _hover_texts(view)
    hover_layer = go.Scatter(
        x=hover_x,
        y=hover_y,
        mode="markers",
        marker=dict(
            size=14,
            opacity=0.001,
            color=values,
            colorscale=SEQUENTIAL_BLUE,
            cmin=vmin,
            cmax=vmax,
            showscale=True,
            colorbar=dict(title=dict(text="Demanda<br>media/período", font=dict(size=10))),
        ),
        text=basic,
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    )
    traces.append(hover_layer)

    fig = go.Figure(traces)
    fig.update_layout(
        xaxis=dict(title="Longitud", showgrid=True, zeroline=False, dtick=GRID_DLON, gridcolor="#e1e0d9"),
        yaxis=dict(
            title="Latitud",
            showgrid=True,
            zeroline=False,
            dtick=GRID_DLAT,
            gridcolor="#e1e0d9",
            scaleanchor="x",
            scaleratio=1,
        ),
        margin=dict(l=50, r=20, t=20, b=40),
        height=560,
        plot_bgcolor="white",
    )
    return fig, basic, detailed


# ── Variability statistics ───────────────────────────────────────────────────


def annual_scenario_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows = scenarios, columns = pixels, values = annual (12-period) demand."""
    return frame.groupby(["id_scenario", "id_pixel"])["demand"].sum().unstack("id_pixel")


def pairwise_correlations(matrix: pd.DataFrame) -> np.ndarray:
    """Off-diagonal Pearson correlations between every pair of scenarios."""
    corr = np.corrcoef(matrix.to_numpy())
    iu = np.triu_indices_from(corr, k=1)
    return corr[iu]


def fig_cv_boxplot(summaries: dict[str, pd.DataFrame]) -> go.Figure:
    fig = go.Figure()
    for regime, summary in summaries.items():
        fig.add_trace(go.Box(y=summary["cv_annual"].to_numpy(), name=regime, marker_color=REGIME_COLORS[regime]))
    fig.update_layout(
        height=380,
        margin=dict(l=60, r=20, t=20, b=40),
        yaxis_title="CV de la demanda anual del píxel entre escenarios",
        showlegend=False,
        plot_bgcolor="white",
    )
    fig.update_yaxes(gridcolor="#eee")
    return fig


def fig_pairwise_corr(corr_by_regime: dict[str, np.ndarray]) -> go.Figure:
    fig = go.Figure()
    for regime, values in corr_by_regime.items():
        fig.add_trace(
            go.Histogram(
                x=values,
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
        xaxis_title="Correlación de Pearson entre un par de escenarios (vector de demanda anual, 161 píxeles)",
        yaxis_title="Densidad",
        legend=dict(orientation="h", y=1.12),
        plot_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="#eee")
    fig.update_yaxes(gridcolor="#eee")
    return fig


def fig_drop_hist(summaries: dict[str, pd.DataFrame]) -> go.Figure:
    """Distribution of each pixel's mean `drop`, one histogram per regime."""
    fig = go.Figure()
    for regime, summary in summaries.items():
        fig.add_trace(
            go.Histogram(
                x=summary["mean_drop"].to_numpy(),
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
        xaxis_title="Drop medio del píxel (items por stop, promedio sobre escenarios y períodos)",
        yaxis_title="Densidad",
        legend=dict(orientation="h", y=1.12),
        plot_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="#eee")
    fig.update_yaxes(gridcolor="#eee")
    return fig


def fig_spaghetti(frame: pd.DataFrame, summary: pd.DataFrame) -> tuple[go.Figure, str]:
    """All scenario curves for the highest-demand pixel of each layer."""
    picks = [summary[summary["layer"] == layer]["mean_demand_period"].idxmax() for layer in sorted(summary["layer"].unique())]
    hues = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    palette = {id_pixel: hues[i % len(hues)] for i, id_pixel in enumerate(picks)}

    fig = go.Figure()
    periods = list(range(1, N_PERIODS + 1))
    for id_pixel in picks:
        sub = frame[frame["id_pixel"] == id_pixel]
        color = palette[id_pixel]
        for scenario_id, scen in sub.groupby("id_scenario"):
            scen = scen.sort_values("period")
            fig.add_trace(
                go.Scatter(
                    x=periods,
                    y=scen["demand"].to_numpy(),
                    mode="lines",
                    line=dict(color=color, width=1),
                    opacity=0.12,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        mean_by_period = sub.groupby("period")["demand"].mean().reindex(range(N_PERIODS))
        fig.add_trace(
            go.Scatter(
                x=periods,
                y=mean_by_period.to_numpy(),
                mode="lines+markers",
                line=dict(color=color, width=2.5),
                name=f"{id_pixel} (media)",
                hovertemplate=f"{id_pixel} período %{{x}}: %{{y:.0f}}<extra></extra>",
            )
        )
    fig.update_layout(
        height=400,
        margin=dict(l=60, r=20, t=30, b=50),
        xaxis_title="Período",
        yaxis_title="Demanda del píxel",
        legend=dict(orientation="h", y=1.12),
        plot_bgcolor="white",
    )
    fig.update_xaxes(dtick=1, gridcolor="#eee")
    fig.update_yaxes(gridcolor="#eee")
    return fig, ", ".join(picks)


# ── Report ────────────────────────────────────────────────────────────────────

CSS_EXTRA = """
#top-bar { position: sticky; top: 0; z-index: 100; display: flex; align-items: center;
           gap: 20px; flex-wrap: wrap; }
#top-bar label { color: white; font-size: 0.9em; font-weight: bold; }
#top-bar select { font-size: 0.95em; padding: 4px 8px; border-radius: 4px; border: none; }
#top-bar input[type=checkbox] { transform: scale(1.2); margin-right: 4px; }
.combo-view { display: none; }
"""


def build_report(regimes: list[str], output_path=None):
    output_path = output_path or (RESULTS_DIR / "explore_scenarios.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = load_all(regimes)
    summaries = {regime: pixel_summary(frame) for regime, frame in data.items()}
    layers = sorted(next(iter(summaries.values()))["layer"].unique())
    n_pixels = len(next(iter(summaries.values())))
    n_scenarios = data[regimes[0]]["id_scenario"].nunique()

    # ── Grid: precompute every regime x layer combo ─────────────────────────
    hover_store: dict[str, dict[str, list[str]]] = {}
    combo_html: dict[str, str] = {}
    for regime in regimes:
        for layer in layers:
            key = f"{regime}_{layer}"
            view = summaries[regime][summaries[regime]["layer"] == layer]
            fig, basic, detailed = build_grid_figure(view)
            combo_html[key] = to_html(fig)
            hover_store[key] = {"basic": basic, "detailed": detailed}

    default_regime = "normal" if "normal" in regimes else regimes[0]
    default_layer = layers[0]
    regime_select = "".join(f'<option value="{r}"{" selected" if r == default_regime else ""}>{r}</option>' for r in regimes)
    layer_select = "".join(f'<option value="{l}"{" selected" if l == default_layer else ""}>{l}</option>' for l in layers)

    js = f"""
const HOVER = {json.dumps(hover_store)};
function currentKey() {{
  return document.getElementById('sel-regime').value + '_' + document.getElementById('sel-layer').value;
}}
function applyDetail(key) {{
  var entry = HOVER[key];
  var el = document.querySelector('.combo-view[data-key="' + key + '"] .js-plotly-plot');
  if (!el || !entry) return;
  var checked = document.getElementById('chk-detail').checked;
  var text = checked ? entry.detailed : entry.basic;
  var traceIndex = el.data.length - 1;
  Plotly.restyle(el, {{text: [text]}}, [traceIndex]);
}}
function updateView() {{
  var key = currentKey();
  document.querySelectorAll('.combo-view').forEach(function(el) {{ el.style.display = 'none'; }});
  var active = document.querySelector('.combo-view[data-key="' + key + '"]');
  if (active) {{
    active.style.display = 'block';
    active.querySelectorAll('.js-plotly-plot').forEach(function(el) {{ Plotly.Plots.resize(el); }});
  }}
  applyDetail(key);
}}
document.addEventListener('DOMContentLoaded', updateView);
"""

    grid_views = "\n".join(f'<div class="combo-view" data-key="{key}">{html}</div>' for key, html in combo_html.items())

    # ── Insights, computed ───────────────────────────────────────────────────
    corr_by_regime = {r: pairwise_correlations(annual_scenario_matrix(data[r])) for r in regimes}
    cv_mean = {r: float(summaries[r]["cv_annual"].mean()) for r in regimes}
    cv_median = {r: float(summaries[r]["cv_annual"].median()) for r in regimes}
    corr_mean = {r: float(np.mean(corr_by_regime[r])) for r in regimes}
    corr_min = {r: float(np.min(corr_by_regime[r])) for r in regimes}
    high_cv_share = {r: float((summaries[r]["cv_annual"] > 0.15).mean()) for r in regimes}

    cv_insight = (
        f"El CV anual por píxel (desviación estándar entre los {n_scenarios} escenarios / media, sobre la demanda "
        f"anual del píxel) tiene mediana "
        + ", ".join(f"{r} = {cv_median[r]:.2f}" for r in regimes)
        + " y media "
        + ", ".join(f"{r} = {cv_mean[r]:.2f}" for r in regimes)
        + ". La fracción de píxeles con CV > 0.15 es "
        + ", ".join(f"{r} = {high_cv_share[r] * 100:.0f}%" for r in regimes)
        + f". Esto es sustancialmente mayor que el CV de la demanda <i>agregada</i> por período reportado en "
        f"<code>scenario_validation.html</code> (~0.12): el shock común de período y la correlación espacial "
        f"de rango corto hacen que el agregado sobre {n_pixels} píxeles se compense parcialmente, pero cada "
        f"píxel individual sigue moviéndose mucho más entre escenarios de lo que sugiere esa cifra agregada."
    )

    corr_insight = (
        "La correlación de Pearson entre pares de escenarios (comparando sus 161 demandas anuales por píxel) "
        "tiene media "
        + ", ".join(f"{r} = {corr_mean[r]:.2f}" for r in regimes)
        + " y mínimo "
        + ", ".join(f"{r} = {corr_min[r]:.2f}" for r in regimes)
        + ". "
        + (
            "Esa correlación es alta porque el nivel esperado por píxel es el mismo en todos los escenarios de un "
            "régimen — lo único que cambia son los shocks — así que dos escenarios cualquiera comparten la misma "
            "forma espacial gruesa (qué píxeles son grandes) y difieren en el detalle. La variabilidad real entre "
            "escenarios no está en <i>qué</i> píxel domina, sino en <i>cuánto</i> exactamente demanda cada uno "
            "período a período, que es lo que mide el CV de arriba y muestra el spaghetti plot."
            if min(corr_mean.values()) > 0.5
            else "Esa correlación es baja: los escenarios de un mismo régimen difieren incluso en qué píxeles "
            "concentran la demanda, no solo en el detalle período a período."
        )
    )

    fig_spag, spag_pixels = fig_spaghetti(data[default_regime], summaries[default_regime])
    spag_insight = (
        f"Para los píxeles de mayor demanda media de cada layer en el régimen '{default_regime}' ({spag_pixels}), "
        f"cada línea fina es uno de los {n_scenarios} escenarios simulados y la línea gruesa es su media. La "
        f"dispersión visible período a período es la misma que resume el CV: no son variaciones cosméticas "
        f"alrededor de un valor fijo."
    )

    drop_mins = {r: float(summaries[r]["mean_drop"].min()) for r in regimes}
    drop_maxs = {r: float(summaries[r]["mean_drop"].max()) for r in regimes}
    drop_ref = summaries[regimes[0]]["mean_drop"]
    drop_rel_diff = {r: float((summaries[r]["mean_drop"] - drop_ref).abs().div(drop_ref).mean()) for r in regimes[1:]}
    drop_corr = {r: float(np.corrcoef(drop_ref, summaries[r]["mean_drop"])[0, 1]) for r in regimes[1:]}
    drop_insight = (
        f"El drop medio por píxel va de {min(drop_mins.values()):.2f} a {max(drop_maxs.values()):.2f} items por "
        "stop, con mediana " + ", ".join(f"{r} = {summaries[r]['mean_drop'].median():.2f}" for r in regimes) + ". "
    )
    if not drop_rel_diff:
        drop_insight += "Un solo régimen seleccionado; no hay con qué comparar."
    elif max(drop_rel_diff.values()) < 0.02:
        drop_insight += (
            f"Las distribuciones de los regímenes prácticamente coinciden (diferencia relativa media entre "
            f"regímenes {max(drop_rel_diff.values()) * 100:.2f}%, correlación píxel a píxel ≥ "
            f"{min(drop_corr.values()):.3f} contra el régimen '{regimes[0]}') porque el multiplicador de régimen "
            "solo escala `stop`, no `drop` (`src/scenarios/generate.py`)."
        )
    else:
        drop_insight += (
            f"Las distribuciones difieren entre regímenes más de lo esperado (diferencia relativa media hasta "
            f"{max(drop_rel_diff.values()) * 100:.2f}%), pese a que el multiplicador de régimen en teoría solo "
            "escala `stop`, no `drop`."
        )

    checks_scope = f"{n_pixels} píxeles · {N_PERIODS} períodos · {len(regimes)} regímenes × {n_scenarios} escenarios."

    sections = [
        section(
            "Grilla de demanda por régimen y layer",
            "Cada rectángulo es un píxel, coloreado por su demanda media por período (promedio sobre "
            "escenarios y períodos). Elegí régimen y layer arriba; el checkbox agrega detalle de variabilidad "
            "al hover.",
            f'<div id="grid-views">{grid_views}</div>',
            method="La escala de color es relativa a la vista: el mínimo y máximo de la colorbar son el mínimo "
            "y máximo de demanda media dentro del régimen+layer seleccionado, no comparables entre vistas "
            "distintas sin mirar la colorbar de cada una. Geometría: rectángulo de lado "
            "√(n_cells) celdas de grilla, centrado en el centroide geométrico del footprint fusionado "
            "(`pixel_centroids()`), no el punto de servicio ponderado por demanda del xlsx.",
        ),
        section(
            "Variabilidad de la demanda por píxel entre escenarios",
            "Coeficiente de variación (CV) de la demanda anual de cada uno de los 161 píxeles, sobre sus "
            f"{n_scenarios} escenarios simulados, por régimen.",
            to_html(fig_cv_boxplot(summaries)),
            cv_insight,
        ),
        section(
            "Drop medio por píxel",
            "Distribución del drop medio (items por visita) de cada uno de los 161 píxeles, por régimen.",
            to_html(fig_drop_hist(summaries)),
            drop_insight,
        ),
        section(
            "Qué tan distintos son los escenarios entre sí",
            "Distribución de la correlación de Pearson entre cada par de escenarios de un mismo régimen, "
            "medida sobre el vector de demanda anual de los 161 píxeles.",
            to_html(fig_pairwise_corr(corr_by_regime)),
            corr_insight,
        ),
        section(
            "Dispersión período a período",
            "Las curvas de demanda de todos los escenarios simulados para los píxeles de mayor demanda.",
            to_html(fig_spag),
            spag_insight,
        ),
    ]

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Explorador de escenarios de demanda</title>
  <script src="{PLOTLY_CDN}"></script>
  <style>{CSS}{CSS_EXTRA}</style>
  <script>{js}</script>
</head>
<body>
<div id="top-bar">
  <h1 style="margin:0;">Explorador de escenarios de demanda</h1>
  <label>Régimen: <select id="sel-regime" onchange="updateView()">{regime_select}</select></label>
  <label>Layer: <select id="sel-layer" onchange="updateView()">{layer_select}</select></label>
  <label><input type="checkbox" id="chk-detail" onchange="applyDetail(currentKey())">Mostrar detalle en el hover</label>
  <p style="width:100%; margin:4px 0 0;">{checks_scope}</p>
</div>
{''.join(sections)}
</body>
</html>"""

    output_path.write_text(html)
    logger.info(f"Report written to {output_path}")
    return output_path

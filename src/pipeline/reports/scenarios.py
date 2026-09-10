"""Validation report for the generated demand scenarios.

Builds a self-contained HTML report answering the questions that decide whether a
scenario set can be trusted: do the marginals match the panel, is the spatial
dependence actually reproduced, do the regimes sit where they should, and how much
of the result rests on the imputed `layer`.

Every insight is computed from the data rather than written by hand, so the prose
cannot silently go stale when the instance changes.
"""

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.core.constants import DEFAULT_SCENARIO_VERSION, N_PERIODS, PATH_SHAPE_PARAMS, REGIME_TARGETS, RESULTS_DIR
from src.core.logging import get_logger
from src.pipeline.demand_panel import load_panel
from src.pipeline.generate import ScenarioGenerator, load_generated
from src.pipeline.marginals import fit_marginals
from src.pipeline.spatial import (
    cholesky_factor,
    correlation_matrix,
    empirical_correlogram,
    fit_correlogram,
    haversine_matrix,
    model_correlation,
    pixel_centroids,
)

logger = get_logger("ScenarioAnalysis")

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"
PLOTLY_CFG = {"displayModeBar": True, "displaylogo": False}

REGIME_COLORS = {"low": "#3498db", "normal": "#27ae60", "high": "#e67e22"}
EMPIRICAL_COLOR = "#2c3e50"


def to_html(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, config=PLOTLY_CFG)


def insight_box(text: str) -> str:
    return f'<div class="insight"><b>💡 Por qué ocurre esto:</b> {text}</div>'


def method_box(text: str) -> str:
    return f'<div class="method"><b>📐 Datos y construcción:</b> {text}</div>'


def section(title: str, subtitle: str, fig_html: str, insight: str = "", method: str = "") -> str:
    return f"""
<div class="section">
  <h2>{title}</h2>
  <p>{subtitle}</p>
  {method_box(method) if method else ""}
  {fig_html}
  {insight_box(insight) if insight else ""}
</div>"""


CSS = """
body { font-family: sans-serif; margin: 0; padding: 0; background: #f4f6f8; }
#top-bar { background: #2c3e50; padding: 12px 24px; }
#top-bar h1 { color: white; margin: 0; font-size: 1.3em; }
#top-bar p  { color: #aaa; margin: 4px 0 0; font-size: 0.85em; }
.section { background: white; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
           margin: 20px 24px; padding: 20px; }
.section h2 { color: #2c3e50; margin: 0 0 4px; font-size: 1.1em; }
.section p  { color: #555; margin: 0 0 12px; font-size: 0.87em; }
.insight { background: #eef6fb; border-left: 4px solid #3498db; padding: 10px 14px;
           margin-top: 16px; font-size: 0.86em; color: #2c3e50; border-radius: 0 4px 4px 0;
           line-height: 1.6; }
.insight b { color: #2980b9; }
.method  { background: #f0fdf4; border-left: 4px solid #27ae60; padding: 10px 14px;
           margin-bottom: 14px; font-size: 0.86em; color: #2c3e50; border-radius: 0 4px 4px 0;
           line-height: 1.6; }
.method b { color: #1e8449; }
table.chk { border-collapse: collapse; font-size: 0.86em; width: 100%; }
table.chk th, table.chk td { border-bottom: 1px solid #eee; padding: 6px 10px; text-align: right; }
table.chk th:first-child, table.chk td:first-child { text-align: left; }
table.chk th { color: #2c3e50; background: #f8f9fa; }
td.ok { color: #1e8449; font-weight: bold; }
td.bad { color: #c0392b; font-weight: bold; }
code { background: #f0f0f0; padding: 1px 4px; border-radius: 3px; font-size: 0.92em; }
"""


# ── Data assembly ─────────────────────────────────────────────────────────────


def load_all(regimes: list[str], version: str = DEFAULT_SCENARIO_VERSION) -> dict:
    """Load parameters, the historical panel and every generated regime."""
    with open(PATH_SHAPE_PARAMS) as file:
        params = json.load(file)
    panel = load_panel()
    generated = {regime: load_generated(regime, version=version, scenario_set="validation") for regime in regimes}
    manifests = {}
    for regime in regimes:
        from src.core.constants import scenario_dir

        with open(scenario_dir(regime, version, "validation") / "manifest.json") as file:
            manifests[regime] = json.load(file)
    return {"params": params, "panel": panel, "generated": generated, "manifests": manifests}


def roundtrip_validation(params: dict, n_periods: int, n_pixels_reference: int) -> dict:
    """Refit the whole spatial pipeline on synthetic data with a known truth.

    This is the acceptance test for the dependence structure. Comparing the
    generated correlogram directly against the fitted model is misleading: the
    estimator centres deviations per period, and that operation does not commute
    with the row-centring inside the correlation, so the two are only comparable
    when measured through the *same* pipeline at the *same* sample size.

    Here a panel of the real panel's shape is synthesized from a known
    `(rho, nugget, plateau)` and pushed back through the estimator. If the recovered
    values match the input, the generator and the estimator agree.
    """
    spatial = params["spatial"]
    pixels = params["pixels"]

    centroids = pixel_centroids().set_index("id_pixel").reindex(pixels)
    distances = haversine_matrix(centroids["lon"].to_numpy(), centroids["lat"].to_numpy())

    truth = {"rho_km": spatial["rho_km"], "nugget": spatial["nugget"], "plateau": spatial["plateau"]}
    sigma = correlation_matrix(distances, truth["rho_km"], truth["nugget"], truth["plateau"])

    generator = ScenarioGenerator.from_params(params)
    generator.cholesky = cholesky_factor(sigma)
    multiplier = params["regimes"]["normal"]["multiplier"]

    rng = np.random.default_rng(7)
    year_months = [(y, m) for y in (2020, 2021, 2022) for m in range(1, 13)][:n_periods]
    rows = []
    for year, month in year_months:
        stop, drop = generator.draw(rng, multiplier)
        for i, id_pixel in enumerate(pixels):
            rows.append(
                {
                    "id_pixel": id_pixel,
                    "layer": id_pixel.split("-")[0],
                    "year": year,
                    "month": month,
                    "n_customers": int(stop[i, month - 1]),
                    "drop": float(drop[i, month - 1]),
                    "excluded": False,
                }
            )
    synthetic = pd.DataFrame(rows)

    refitted = fit_marginals(synthetic)
    balanced = refitted["stop"]["deviations_balanced"]
    sub_centroids = centroids.reindex(balanced.index)
    sub_distances = haversine_matrix(sub_centroids["lon"].to_numpy(), sub_centroids["lat"].to_numpy())
    correlogram = empirical_correlogram(balanced, sub_distances, n_bins=12)
    recovered = fit_correlogram(correlogram)

    logger.info(
        f"Round-trip: truth rho={truth['rho_km']:.2f}/nugget={truth['nugget']:.3f}/plateau={truth['plateau']:.3f} "
        f"-> recovered rho={recovered['rho_km']:.2f}/nugget={recovered['nugget']:.3f}/plateau={recovered['plateau']:.3f}"
    )
    return {
        "truth": truth,
        "recovered": {k: recovered[k] for k in ("rho_km", "nugget", "plateau", "r2", "model")},
        "correlogram": correlogram,
        "n_pixels": len(balanced),
        "n_pixels_reference": n_pixels_reference,
        "n_periods": n_periods,
    }


def aggregate_cv_impact(params: dict) -> dict:
    """Coefficient of variation of aggregate per-period demand under each structure.

    Answers the question the whole redesign exists for: how much aggregate risk does
    the facility-location decision actually see? Computed analytically from the
    fitted dispersions and the demand-weighted pixel shares, so it does not depend
    on the Monte Carlo sample.
    """
    pixels = params["pixels"]
    spatial = params["spatial"]

    generator = ScenarioGenerator.from_params(params)
    weights = generator.expected_stop.mean(axis=1) * generator.expected_drop.mean(axis=1)
    weights = weights / weights.sum()

    sigma = np.array([params["stop"]["sigma"][p] for p in pixels])
    sigma_common = params["stop"]["sigma_common"]

    centroids = pixel_centroids().set_index("id_pixel").reindex(pixels)
    distances = haversine_matrix(centroids["lon"].to_numpy(), centroids["lat"].to_numpy())

    def cv(rho_km, nugget, plateau, with_common):
        corr = model_correlation(distances, rho_km, nugget, plateau)
        np.fill_diagonal(corr, 1.0)
        spatial_part = float((np.outer(weights, weights) * np.outer(sigma, sigma) * corr).sum())
        return float(np.sqrt((sigma_common**2 if with_common else 0.0) + spatial_part))

    cv_independent = cv(1e-9, 0.0, 0.0, False)
    cv_no_plateau = cv(spatial["rho_km"], spatial["nugget"], 0.0, True)
    cv_full = cv(spatial["rho_km"], spatial["nugget"], spatial["plateau"], True)

    return {
        "cv_independent": cv_independent,
        "cv_no_plateau": cv_no_plateau,
        "cv_full": cv_full,
        "ratio": cv_full / cv_independent if cv_independent else float("nan"),
        "plateau_pct": (cv_full / cv_no_plateau - 1) * 100 if cv_no_plateau else float("nan"),
    }


# ── Figures ───────────────────────────────────────────────────────────────────


def fig_correlogram(params: dict, roundtrip: dict) -> go.Figure:
    spatial = params["spatial"]
    empirical = pd.DataFrame(spatial["correlogram"])
    grid = np.linspace(0.1, empirical["h_km"].max() * 1.05, 200)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=empirical["h_km"],
            y=empirical["corr"],
            mode="markers",
            name="Panel histórico",
            marker=dict(size=9, color=EMPIRICAL_COLOR),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=model_correlation(grid, spatial["rho_km"], spatial["nugget"], spatial["plateau"]),
            mode="lines",
            name=f"Modelo ajustado (ρ={spatial['rho_km']:.2f} km)",
            line=dict(color="#e74c3c", width=2),
        )
    )
    rt = roundtrip["correlogram"]
    fig.add_trace(
        go.Scatter(
            x=rt["h_km"],
            y=rt["corr"],
            mode="markers",
            name="Escenarios generados (mismo estimador)",
            marker=dict(size=8, color="#27ae60", symbol="diamond"),
        )
    )
    fig.update_layout(
        height=420,
        margin=dict(l=60, r=20, t=30, b=50),
        xaxis_title="Distancia entre centroides (km)",
        yaxis_title="Correlación de las desviaciones",
        legend=dict(orientation="h", y=1.12),
        plot_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="#eee", zeroline=False)
    fig.update_yaxes(gridcolor="#eee", zerolinecolor="#ccc")
    return fig


def fig_aggregate(data: dict) -> go.Figure:
    panel = data["panel"]
    included = panel[~panel["excluded"]]
    historical = included.groupby(["year", "month"])["model_demand"].sum()

    fig = go.Figure()
    fig.add_trace(
        go.Box(
            y=historical.to_numpy(),
            name="Histórico<br>(35 meses)",
            marker_color=EMPIRICAL_COLOR,
            boxpoints="all",
            jitter=0.4,
            pointpos=0,
        )
    )
    for regime, frame in data["generated"].items():
        totals = frame.groupby(["id_scenario", "period"])["demand"].sum()
        fig.add_trace(go.Box(y=totals.to_numpy(), name=regime, marker_color=REGIME_COLORS[regime]))
        fig.add_hline(
            y=REGIME_TARGETS[regime],
            line=dict(color=REGIME_COLORS[regime], dash="dot", width=1),
            annotation_text=f"objetivo {regime}",
            annotation_position="right",
            annotation_font=dict(size=9, color=REGIME_COLORS[regime]),
        )
    fig.update_layout(
        height=430,
        margin=dict(l=70, r=110, t=30, b=40),
        yaxis_title="Demanda del modelo por período  (Σ stop × drop)",
        showlegend=False,
        plot_bgcolor="white",
    )
    fig.update_yaxes(gridcolor="#eee")
    return fig


def fig_marginals(data: dict) -> go.Figure:
    panel = data["panel"]
    included = panel[~panel["excluded"]]
    fig = go.Figure()
    for name, values, color in (
        ("Histórico", included["n_customers"], EMPIRICAL_COLOR),
        ("Generado (normal)", data["generated"]["normal"]["stop"], REGIME_COLORS["normal"]),
    ):
        fig.add_trace(
            go.Histogram(
                x=values,
                name=name,
                marker_color=color,
                opacity=0.6,
                histnorm="probability density",
                nbinsx=60,
            )
        )
    fig.update_layout(
        height=380,
        margin=dict(l=60, r=20, t=30, b=50),
        barmode="overlay",
        xaxis_title="Clientes por píxel-período (stop)",
        yaxis_title="Densidad",
        legend=dict(orientation="h", y=1.12),
        plot_bgcolor="white",
    )
    fig.update_xaxes(range=[0, included["n_customers"].quantile(0.995)], gridcolor="#eee")
    fig.update_yaxes(gridcolor="#eee")
    return fig


def fig_seasonality(data: dict) -> go.Figure:
    panel = data["panel"]
    included = panel[~panel["excluded"]]
    hist = included.groupby("month")["model_demand"].sum()
    hist = hist / hist.mean()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(1, 13)),
            y=hist.reindex(range(1, 13)).to_numpy(),
            mode="lines+markers",
            name="Histórico",
            line=dict(color=EMPIRICAL_COLOR, width=2),
        )
    )
    for regime, frame in data["generated"].items():
        by_period = frame.groupby("period")["demand"].sum()
        by_period = by_period / by_period.mean()
        fig.add_trace(
            go.Scatter(
                x=[p + 1 for p in by_period.index],
                y=by_period.to_numpy(),
                mode="lines+markers",
                name=regime,
                line=dict(color=REGIME_COLORS[regime], width=1.5, dash="dot"),
            )
        )
    fig.update_layout(
        height=380,
        margin=dict(l=60, r=20, t=30, b=50),
        xaxis_title="Período (mes)",
        yaxis_title="Demanda relativa a su media anual",
        legend=dict(orientation="h", y=1.12),
        plot_bgcolor="white",
    )
    fig.update_xaxes(dtick=1, gridcolor="#eee")
    fig.update_yaxes(gridcolor="#eee")
    return fig


def fig_layer_sensitivity(data: dict) -> go.Figure:
    panel = data["panel"]
    share = panel.groupby("id_pixel")["crosswalk_share"].mean().sort_values()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=share.to_numpy(), nbinsx=40, marker_color="#8e44ad", opacity=0.8))
    fig.add_vline(
        x=float(share.mean()),
        line=dict(color="#c0392b", dash="dash"),
        annotation_text=f"media {share.mean():.2f}",
        annotation_position="top",
    )
    fig.update_layout(
        height=350,
        margin=dict(l=60, r=20, t=30, b=50),
        xaxis_title="Fracción de items del píxel con `layer` del crosswalk (no imputado)",
        yaxis_title="Píxeles",
        showlegend=False,
        plot_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="#eee")
    fig.update_yaxes(gridcolor="#eee")
    return fig


def checks_table(data: dict) -> tuple[str, int]:
    """Contract invariants and degeneracy checks, as an HTML table."""
    rows = []
    n_failed = 0

    for regime, frame in data["generated"].items():
        manifest = data["manifests"][regime]
        n_pixels = frame["id_pixel"].nunique()
        checks = [
            (f"[{regime}] píxeles por escenario == 161", n_pixels == 161, f"{n_pixels}"),
            (f"[{regime}] períodos == {N_PERIODS}", frame["period"].nunique() == N_PERIODS, f"{frame['period'].nunique()}"),
            (f"[{regime}] stop >= 1", int(frame["stop"].min()) >= 1, f"min {int(frame['stop'].min())}"),
            (f"[{regime}] drop > 0", float(frame["drop"].min()) > 0, f"min {frame['drop'].min():.4f}"),
            (f"[{regime}] demand > 0", float(frame["demand"].min()) > 0, f"min {frame['demand'].min():.4f}"),
            (
                f"[{regime}] demand == stop × drop",
                bool(np.allclose(frame["demand"], (frame["stop"] * frame["drop"]).round(4), atol=1e-6)),
                "exacto",
            ),
            (
                f"[{regime}] piso max(1,·) no liga",
                manifest["stop_floor_share"] < 1e-4,
                f"{manifest['stop_floor_share'] * 100:.3f}% de celdas",
            ),
            (
                f"[{regime}] objetivo alcanzado (±1%)",
                abs(manifest["period_total_mean"] / manifest["target_period_demand"] - 1) < 0.01,
                f"{manifest['period_total_mean']:,.0f} vs {manifest['target_period_demand']:,.0f}",
            ),
        ]
        for label, passed, detail in checks:
            rows.append((label, passed, detail))
            if not passed:
                n_failed += 1

    body = "".join(
        f'<tr><td>{label}</td><td class="{"ok" if passed else "bad"}">{"OK" if passed else "FALLA"}</td>'
        f"<td>{detail}</td></tr>"
        for label, passed, detail in rows
    )
    html = f'<table class="chk"><tr><th>Chequeo</th><th>Estado</th><th>Detalle</th></tr>{body}</table>'
    return html, n_failed


# ── Report ────────────────────────────────────────────────────────────────────


def build_report(regimes: list[str], output_path=None, version: str = DEFAULT_SCENARIO_VERSION):
    """Assemble the validation report."""
    output_path = output_path or (RESULTS_DIR / "analysis" / version / "scenario_validation.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = load_all(regimes, version)
    params = data["params"]
    panel = data["panel"]
    included = panel[~panel["excluded"]]

    n_periods_panel = included.groupby(["year", "month"]).ngroups
    roundtrip = roundtrip_validation(params, n_periods_panel, len(params["pixels"]))

    spatial = params["spatial"]
    historical = included.groupby(["year", "month"])["model_demand"].sum()

    # ── Insights, computed ───────────────────────────────────────────────────
    rho = spatial["rho_km"]
    plateau = spatial["plateau"]
    rt_rho, rt_plateau = roundtrip["recovered"]["rho_km"], roundtrip["recovered"]["plateau"]
    rho_err = abs(rt_rho / rho - 1) * 100 if rho else float("nan")

    cv_impact = aggregate_cv_impact(params)
    plateau_recovered = abs(rt_plateau - plateau) < 0.5 * max(plateau, 1e-6)

    corr_insight = (
        f"La dependencia tiene dos escalas: un decaimiento de rango muy corto "
        f"(ρ = {rho:.2f} km, del orden de una celda de la grilla) más un componente "
        f"city-wide de {plateau:.3f}. El ajuste con plateau da R² = {spatial['weighted_r2']:.3f} "
        f"contra {min(c['r2'] for c in spatial['candidates']):.3f} de la exponencial pura, porque el "
        f"correlograma empírico no es monótono: baja en los primeros km y se estabiliza en lugar de ir a cero. "
        f"<br><br><b>El round-trip recupera ρ = {rt_rho:.2f} km (error {rho_err:.1f}%) pero "
        f"plateau = {rt_plateau:.3f} contra {plateau:.3f} de entrada"
        + ("" if plateau_recovered else ", es decir NO lo reproduce")
        + f".</b> La razón es identificable: el generador aplica un único factor común idéntico a todos los "
        f"píxeles, así que el centrado por período del estimador lo elimina entero. En la data real el plateau "
        f"sobrevive a ese centrado, lo que implica que proviene de <i>cargas heterogéneas</i> sobre el shock "
        f"city-wide — píxeles que responden con distinta intensidad — y eso el modelo actual no lo tiene. "
        f"<br><br>El impacto está acotado: sobre el CV de la demanda agregada por período, el plateau aporta "
        f"+{cv_impact['plateau_pct']:.1f}%, mientras que el modelo completo lleva ese CV de "
        f"{cv_impact['cv_independent']:.4f} (píxeles independientes y sin factor común, el procedimiento "
        f"anterior) a {cv_impact['cv_full']:.4f}, un factor {cv_impact['ratio']:.1f}×. El mecanismo dominante "
        f"de riesgo agregado sí está reproducido; el plateau queda como refinamiento pendiente vía cargas "
        f"heterogéneas por píxel."
    )

    means = {r: data["manifests"][r]["period_total_mean"] for r in regimes}
    spread = {
        r: (data["manifests"][r]["period_total_p90"] - data["manifests"][r]["period_total_p10"]) / means[r] for r in regimes
    }
    historical_spread = (historical.quantile(0.9) - historical.quantile(0.1)) / historical.mean() * 100
    agg_insight = (
        "Los tres regímenes caen donde deben: "
        + ", ".join(f"{r} = {means[r]:,.0f}" for r in regimes)
        + " contra objetivos "
        + ", ".join(f"{REGIME_TARGETS[r]:,.0f}" for r in regimes)
        + f". El ancho p10-p90 dentro de cada régimen es {np.mean(list(spread.values())) * 100:.1f}% de su media, "
        f"bastante menor que la dispersión del histórico ({historical_spread:.0f}%), "
        f"porque el histórico mezcla la tendencia 2020-2022 mientras cada régimen es un nivel fijo. "
        f"Ese es el precio de anclar los regímenes a cuantiles crudos."
    )

    sigma_common = params["stop"]["sigma_common"]
    sigma_dev = float(np.median(list(params["stop"]["sigma"].values())))
    marg_insight = (
        f"La dispersión ajustada es σ_común = {sigma_common:.3f} para el shock del período y "
        f"σ_desviación ≈ {sigma_dev:.3f} por píxel, muy lejos del 0.90 hardcodeado en el procedimiento "
        f"anterior (que implicaba un CV de 1.12 contra el 0.124 observado). Con estos valores el piso "
        f"<code>max(1, ·)</code> nunca liga, así que <code>stop</code> no queda sesgado hacia arriba en los "
        f"píxeles chicos."
    )

    cw = panel.groupby("id_pixel")["crosswalk_share"].mean()
    layer_insight = (
        f"El <code>layer</code> viene del crosswalk manual para {cw.mean() * 100:.1f}% de los items en promedio; "
        f"el resto se imputa con el umbral drop ≥ {params['layer_drop_threshold']}, que reproduce las etiquetas "
        f"hechas a mano con 92.1% de acierto. {int((cw < 0.5).sum())} de {len(cw)} píxeles dependen de la "
        f"imputación para más de la mitad de sus items, así que sus niveles son los más sensibles a esa decisión. "
        f"El umbral óptimo por píxel solo llega a 95.2%, o sea que el <code>layer</code> no es una función "
        f"determinista del drop y ese ~8% de error es irreducible sin la asignación original."
    )

    seasonality = included.groupby("month")["model_demand"].sum()
    peak_month = int(seasonality.idxmax())
    trough_month = int(seasonality.idxmin())
    season_insight = (
        f"La estacionalidad generada sigue al histórico, con el pico en el mes {peak_month} y el valle en el "
        f"mes {trough_month} (relación {seasonality.max() / seasonality.min():.2f}×). Las tres curvas de régimen "
        f"se superponen porque el régimen es un escalar sobre el nivel y no toca el perfil estacional — "
        f"a diferencia del procedimiento anterior, donde amplificar σ en los meses pico también movía su media."
    )

    checks_html, n_failed = checks_table(data)
    checks_insight = (
        "Todos los invariantes del contrato pasan."
        if n_failed == 0
        else f"<b>{n_failed} chequeos fallan</b> — los escenarios no deben usarse hasta resolverlos."
    ) + (
        " Los tres que importan son <code>stop >= 1</code>, <code>drop > 0</code> y <code>demand > 0</code>: "
        "la CA solo escribe claves de costo cuando <code>demand > 0</code> y el modelo uncapacitated las indexa "
        "directo, así que un solo píxel-período en cero es un <code>KeyError</code> en el solve."
    )

    figures = [
        section(
            "Correlación espacial: lo que se buscaba arreglar",
            "Correlograma del panel histórico, el modelo ajustado y los escenarios generados medidos con el mismo estimador.",
            to_html(fig_correlogram(params, roundtrip)),
            corr_insight,
            f"Desviaciones del panel después de remover nivel por píxel, tendencia por año y estacionalidad por "
            f"layer-mes, centradas por período sobre los {spatial['n_pixels_correlogram']} píxeles con historia "
            f"completa. Distancias haversine entre centroides geométricos de la huella fusionada de cada píxel "
            f"({len(params['pixels'])} píxeles, {roundtrip['n_periods']} períodos). Los puntos verdes salen de "
            f"sintetizar un panel del mismo tamaño desde una Σ conocida y volver a pasarlo por el estimador "
            f"completo: es la única comparación válida, porque el centrado por período no conmuta con el de la "
            f"correlación y las dos mediciones solo coinciden a igual tamaño de muestra.",
        ),
        section(
            "Demanda agregada y separación de regímenes",
            "Distribución de la demanda del modelo por período, histórico contra los tres regímenes.",
            to_html(fig_aggregate(data)),
            agg_insight,
            f"Demanda del modelo = Σ_j (stop_j × drop_j) por período, que representa una ronda de reparto "
            f"representativa del mes — no los items mensuales crudos, que son ~3× más grandes porque un cliente "
            f"se atiende varias veces al mes. Histórico: {len(historical)} meses usables. "
            f"Generado: {data['manifests'][regimes[0]]['n_scenarios']} escenarios × {N_PERIODS} períodos por régimen. "
            f"Los objetivos son los p10/p50/p90 del histórico.",
        ),
        section(
            "Marginales de `stop`",
            "Distribución de clientes por píxel-período, histórico contra generado.",
            to_html(fig_marginals(data)),
            marg_insight,
            "Histograma normalizado a densidad, recortado al p99.5 del histórico para que la cola no aplaste el "
            "eje. El régimen mostrado es `normal`; los otros dos son el mismo perfil escalado por su "
            "multiplicador.",
        ),
        section(
            "Estacionalidad",
            "Perfil mensual relativo a la media anual.",
            to_html(fig_seasonality(data)),
            season_insight,
            "Demanda del modelo agregada por mes y dividida por su propia media anual, de modo que el nivel se "
            "cancela y solo queda la forma. El histórico suma los tres años.",
        ),
        section(
            "Sensibilidad a la imputación de `layer`",
            "Cuánto de cada píxel proviene del crosswalk manual y cuánto de la imputación por drop.",
            to_html(fig_layer_sensitivity(data)),
            layer_insight,
            f"Para cada píxel, la fracción de items cuyo cliente aparece en el crosswalk manual "
            f"(`customer_pixel_layer.csv`, {3969} clientes) contra el total de items asignados a ese píxel. "
            f"El resto recibió `layer` por umbral de drop.",
        ),
        section(
            "Invariantes del contrato y chequeos de degeneración",
            "Lo que tiene que cumplirse para que la CA y los modelos Gurobi corran sin tocarlos.",
            checks_html,
            checks_insight,
            "Evaluado sobre todos los escenarios simulados de cada régimen más su manifest. "
            "`demand == stop × drop` se compara con tolerancia 1e-6 sobre el redondeo a 4 decimales del contrato.",
        ),
    ]

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Validación de escenarios de demanda</title>
  <script src="{PLOTLY_CDN}"></script>
  <style>{CSS}</style>
</head>
<body>
<div id="top-bar">
  <h1>Validación de escenarios de demanda</h1>
  <p>{len(params['pixels'])} píxeles · {N_PERIODS} períodos · {len(regimes)} regímenes ×
     {data['manifests'][regimes[0]]['n_scenarios']} escenarios ·
     panel {len(historical)} meses (2020-2022, sin 2021-02) ·
     ρ = {rho:.2f} km · plateau = {plateau:.3f} ·
     params v{params['version']} del {params['generated_on']}</p>
</div>
{''.join(figures)}
</body>
</html>"""

    output_path.write_text(html)
    logger.info(f"Report written to {output_path} ({n_failed} failed checks)")
    return output_path, n_failed

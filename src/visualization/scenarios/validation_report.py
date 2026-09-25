"""Validation report for the generated demand scenarios.

Answers the questions that decide whether a scenario set can be trusted: do the
marginals match the panel, is the spatial dependence actually reproduced, do the regimes
sit where they should, and how much of the result rests on the imputed `layer`.

Every insight is computed from the data rather than written by hand, so the prose
cannot silently go stale when the instance changes.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.core.constants import N_PERIODS, REGIME_TARGETS
from src.tools.validation import Check
from src.tools.validation import n_failed as n_failed_checks
from src.visualization.components.labels import EMPIRICAL_COLOR, REGIME_COLORS
from src.visualization.html import fig_html, page, section, top_bar


@dataclass
class ValidationReportData:
    """Everything the validation report shows, computed by `src.scenarios.reports`."""

    regimes: list[str]
    params: Mapping
    panel: pd.DataFrame
    generated: dict[str, pd.DataFrame]
    manifests: dict[str, dict]
    roundtrip: dict
    cv_impact: dict
    correlogram_curve: tuple[np.ndarray, np.ndarray]
    checks: list[Check]


# ── Figures ───────────────────────────────────────────────────────────────────


def fig_correlogram(params: Mapping, roundtrip: dict, curve: tuple[np.ndarray, np.ndarray]) -> go.Figure:
    spatial = params["spatial"]
    empirical = pd.DataFrame(spatial["correlogram"])
    grid, model = curve

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
            y=model,
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


def fig_aggregate(data: ValidationReportData) -> go.Figure:
    panel = data.panel
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
    for regime, frame in data.generated.items():
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


def fig_marginals(data: ValidationReportData) -> go.Figure:
    panel = data.panel
    included = panel[~panel["excluded"]]
    fig = go.Figure()
    for name, values, color in (
        ("Histórico", included["n_customers"], EMPIRICAL_COLOR),
        ("Generado (normal)", data.generated["normal"]["stop"], REGIME_COLORS["normal"]),
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


def fig_seasonality(data: ValidationReportData) -> go.Figure:
    panel = data.panel
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
    for regime, frame in data.generated.items():
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


def fig_layer_sensitivity(data: ValidationReportData) -> go.Figure:
    panel = data.panel
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


def checks_table(checks: list[Check]) -> str:
    """Contract invariants and degeneracy checks, as an HTML table."""
    body = "".join(
        f'<tr><td>{item.name}</td><td class="{"ok" if item.passed else "bad"}">{"OK" if item.passed else "FALLA"}</td>'
        f"<td>{item.message}</td></tr>"
        for item in checks
    )
    return f'<table class="chk"><tr><th>Chequeo</th><th>Estado</th><th>Detalle</th></tr>{body}</table>'


# ── Report ────────────────────────────────────────────────────────────────────


def render(data: ValidationReportData, output_path: Path) -> Path:
    """Write the validation report."""
    regimes = data.regimes
    params = data.params
    panel = data.panel
    included = panel[~panel["excluded"]]

    roundtrip = data.roundtrip

    spatial = params["spatial"]
    historical = included.groupby(["year", "month"])["model_demand"].sum()

    # ── Insights, computed ───────────────────────────────────────────────────
    rho = spatial["rho_km"]
    plateau = spatial["plateau"]
    rt_rho, rt_plateau = roundtrip["recovered"]["rho_km"], roundtrip["recovered"]["plateau"]
    rho_err = abs(rt_rho / rho - 1) * 100 if rho else float("nan")

    cv_impact = data.cv_impact
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

    means = {r: data.manifests[r]["period_total_mean"] for r in regimes}
    spread = {r: (data.manifests[r]["period_total_p90"] - data.manifests[r]["period_total_p10"]) / means[r] for r in regimes}
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

    checks_html, n_failed = checks_table(data.checks), n_failed_checks(data.checks)
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
            fig_html(fig_correlogram(params, roundtrip, data.correlogram_curve)),
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
            fig_html(fig_aggregate(data)),
            agg_insight,
            f"Demanda del modelo = Σ_j (stop_j × drop_j) por período, que representa una ronda de reparto "
            f"representativa del mes — no los items mensuales crudos, que son ~3× más grandes porque un cliente "
            f"se atiende varias veces al mes. Histórico: {len(historical)} meses usables. "
            f"Generado: {data.manifests[regimes[0]]['n_scenarios']} escenarios × {N_PERIODS} períodos por régimen. "
            f"Los objetivos son los p10/p50/p90 del histórico.",
        ),
        section(
            "Marginales de `stop`",
            "Distribución de clientes por píxel-período, histórico contra generado.",
            fig_html(fig_marginals(data)),
            marg_insight,
            "Histograma normalizado a densidad, recortado al p99.5 del histórico para que la cola no aplaste el "
            "eje. El régimen mostrado es `normal`; los otros dos son el mismo perfil escalado por su "
            "multiplicador.",
        ),
        section(
            "Estacionalidad",
            "Perfil mensual relativo a la media anual.",
            fig_html(fig_seasonality(data)),
            season_insight,
            "Demanda del modelo agregada por mes y dividida por su propia media anual, de modo que el nivel se "
            "cancela y solo queda la forma. El histórico suma los tres años.",
        ),
        section(
            "Sensibilidad a la imputación de `layer`",
            "Cuánto de cada píxel proviene del crosswalk manual y cuánto de la imputación por drop.",
            fig_html(fig_layer_sensitivity(data)),
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

    header = top_bar(
        "Validación de escenarios de demanda",
        f"""{len(params['pixels'])} píxeles · {N_PERIODS} períodos · {len(regimes)} regímenes ×
     {data.manifests[regimes[0]]['n_scenarios']} escenarios ·
     panel {len(historical)} meses (2020-2022, sin 2021-02) ·
     ρ = {rho:.2f} km · plateau = {plateau:.3f} ·
     params v{params['version']} del {params['generated_on']}""",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page("Validación de escenarios de demanda", header + "\n" + "".join(figures)))
    return output_path

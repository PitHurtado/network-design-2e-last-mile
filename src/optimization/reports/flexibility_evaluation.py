"""Interactive out-of-sample and VSS report for flexibility policies."""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from src.core.constants import RESULTS_DIR

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"
PLOTLY_CFG = {"displayModeBar": True, "displaylogo": False, "responsive": True}
CSS = """
* { box-sizing:border-box; } body { font-family:Arial, Helvetica, sans-serif; margin:0; background:#edf2f5; color:#173042; }
#top { background:#123b57; color:#fff; padding:34px max(24px, calc((100vw - 1320px) / 2)); border-bottom:5px solid #e0a82e; }
#top h1 { font-size:2.05rem; letter-spacing:-.04em; margin:0; max-width:760px; } #top p { color:#c9d9e3; margin:9px 0 0; }
.wrap { max-width:1320px; margin:0 auto; padding:0 24px; }.hero { display:grid; grid-template-columns:1.25fr .75fr; gap:28px; padding:30px 0 18px; align-items:end; }
.hero h2 { font-size:1.45rem; letter-spacing:-.025em; margin:0 0 9px; }.hero p { line-height:1.55; margin:0; max-width:720px; }
.cards { display:grid; grid-template-columns:repeat(4, 1fr); gap:12px; margin:18px 0 26px; }.metric { background:#fff; border-top:4px solid #1e7092; padding:16px; min-height:106px; }
.metric.good { border-color:#22855a; }.metric.warn { border-color:#d97a4a; }.metric .value { color:#123b57; font-size:1.7rem; font-weight:700; letter-spacing:-.04em; margin:4px 0; }.metric .label { color:#557081; font-size:.78rem; }
.section { background:#fff; margin:18px 0; padding:24px; border:1px solid #d7e0e5; }.section h2 { font-size:1.24rem; letter-spacing:-.025em; margin:0 0 5px; }.section > p { color:#557081; margin:0 0 14px; line-height:1.45; }
.method { background:#eef7f5; border-left:4px solid #22855a; padding:11px 13px; margin:12px 0; line-height:1.45; }.insight { background:#fff7df; border-left:4px solid #e0a82e; padding:11px 13px; margin-top:12px; line-height:1.45; }
.two { display:grid; grid-template-columns:1fr 1fr; gap:22px; align-items:start; }.small { font-size:.86rem; color:#557081; } .detail { margin-top:10px; }
table { border-collapse:collapse; font-size:.8em; width:100%; } th, td { border-bottom:1px solid #d7e0e5; padding:8px 7px; text-align:right; white-space:nowrap; } th { background:#eef3f6; color:#345264; font-weight:600; } td:first-child, th:first-child { text-align:left; } tr:nth-child(even) { background:#fafcfd; }
@media (max-width:1000px) { .cards { grid-template-columns:repeat(2, 1fr); } } @media (max-width:800px) { .hero,.two { grid-template-columns:1fr; }.cards { grid-template-columns:1fr; } #top h1 { font-size:1.7rem; }.wrap { padding:0 14px; } .section { padding:16px; overflow:auto; } }
"""

CASE_LABELS = {"annual_expected": "Promedio anual", "expected": "Escenario esperado", "optimization": "Optimización (30)"}
FLEX_LABELS = {"fixed_operation": "Operación fija", "on_off_installed": "Encendido / apagado", "up_to_installed": "Hasta capacidad"}
REGIME_LABELS = {"low": "Bajo", "normal": "Normal", "high": "Alto"}
SOLVE_STATUS_LABELS = {"OPTIMAL": "Óptima", "TIME_LIMIT": "Límite de tiempo"}


def _html(fig):
    return fig.to_html(full_html=False, include_plotlyjs=False, config=PLOTLY_CFG)


def _load(root: Path, version: str):
    return [json.loads(path.read_text()) for path in sorted((root / version).rglob("evaluation.json"))]


def _frames(payloads):
    rows, scenarios, installations = [], [], []
    for data in payloads:
        key = {name: data[name] for name in ("version", "regime", "flexibility", "solution_case")}
        source = json.loads(Path(data["source_result"]).read_text())
        rows.append({
            **key,
            **data["means"],
            **{f"evaluation_{k}": v for k, v in data["solve"].items()},
            **{f"source_{k}": v for k, v in source["solve"].items()},
        })
        scenarios.extend([{**key, **row} for row in data["scenario_costs"]])
        installations.extend([{**key, **row} for row in data["fixed_installation"]])
    return pd.DataFrame(rows), pd.DataFrame(scenarios), pd.DataFrame(installations)


def _percentiles(scenarios):
    grouped = scenarios.groupby(["regime", "flexibility", "solution_case"])
    rows = []
    for key, group in grouped:
        total, recourse = group["total_cost"], group["second_stage_cost"]
        rows.append({
            "regime": key[0], "flexibility": key[1], "solution_case": key[2],
            "mean_total": total.mean(), "p05_total": total.quantile(.05), "p50_total": total.quantile(.50), "p95_total": total.quantile(.95),
            "mean_second_stage": recourse.mean(), "p95_second_stage": recourse.quantile(.95),
            "mean_operation": group["operation_cost"].mean(),
            "mean_routing_facilities": group["routing_facilities_cost"].mean(),
            "mean_routing_dc": group["routing_dc_cost"].mean(),
        })
    return pd.DataFrame(rows)


def _vss(percentiles):
    index = percentiles.set_index(["regime", "flexibility", "solution_case"])
    rows = []
    for regime, flexibility in index.index.droplevel("solution_case").unique():
        optimized = index.loc[(regime, flexibility, "optimization")]
        for baseline in ("expected", "annual_expected"):
            compared = index.loc[(regime, flexibility, baseline)]
            rows.append({
                "regime": regime, "flexibility": flexibility, "baseline_solution": baseline,
                "optimized_validation_mean": optimized["mean_total"],
                "baseline_validation_mean": compared["mean_total"],
                "vss_cost_reduction": compared["mean_total"] - optimized["mean_total"],
                "vss_pct": 100 * (compared["mean_total"] - optimized["mean_total"]) / compared["mean_total"],
            })
    return pd.DataFrame(rows)


def _load_validation_benchmarks(version: str) -> pd.DataFrame:
    root = RESULTS_DIR / "flexibility_validation_benchmark" / version
    rows = []
    for path in sorted(root.rglob("rp_validation.json")):
        item = json.loads(path.read_text())
        if item.get("assignment_variables") == "binary":
            rows.append({"regime": item["regime"], "flexibility": item["flexibility"], **item["solve"]})
    return pd.DataFrame(rows)


def _theoretical_vss(percentiles: pd.DataFrame, benchmarks: pd.DataFrame) -> pd.DataFrame:
    """VSS interval from an RP incumbent and lower bound; exact only if RP is optimal."""
    if benchmarks.empty:
        return pd.DataFrame()
    base = percentiles[percentiles.solution_case.isin(("annual_expected", "expected"))]
    rows = []
    for item in base.itertuples():
        rp = benchmarks[(benchmarks.regime == item.regime) & (benchmarks.flexibility == item.flexibility)]
        if rp.empty:
            continue
        rp = rp.iloc[0]
        if pd.isna(rp["objective_value"]) or pd.isna(rp["best_bound_value"]):
            continue
        lower = max(0.0, item.mean_total - rp["objective_value"])
        upper = item.mean_total - rp["best_bound_value"]
        rows.append({
            "regime": item.regime,
            "flexibility": item.flexibility,
            "baseline_solution": item.solution_case,
            "baseline_validation_mean": item.mean_total,
            "rp_incumbent": rp["objective_value"],
            "rp_bound": rp["best_bound_value"],
            "vss_lower": lower,
            "vss_upper": upper,
            "is_exact": bool(rp["is_optimal"]),
            "rp_status": rp["status"],
            "rp_gap_pct": rp["optimality_gap"],
        })
    return pd.DataFrame(rows)


def _vss_figure(vss):
    rows = [(regime, flexibility) for regime in ("low", "normal", "high") for flexibility in FLEX_LABELS]
    columns = ["annual_expected", "expected"]
    values = [[vss.loc[(vss.regime == regime) & (vss.flexibility == flexibility) & (vss.baseline_solution == baseline), "vss_cost_reduction"].iloc[0] for baseline in columns] for regime, flexibility in rows]
    fig = go.Figure(go.Heatmap(
        z=values, x=[CASE_LABELS[x] for x in columns], y=[f"{REGIME_LABELS[r]} · {FLEX_LABELS[f]}" for r, f in rows],
        colorscale=[[0, "#b94f4f"], [.48, "#f4e9d8"], [.5, "#ffffff"], [1, "#22855a"]], zmid=0,
        text=[[f"{value:,.0f}" for value in row] for row in values], texttemplate="%{text}", textfont={"size": 12},
        colorbar_title="Ahorro VSS", hovertemplate="%{y}<br>vs %{x}<br>VSS: %{z:,.2f}<extra></extra>",
    ))
    fig.update_layout(height=460, margin=dict(l=190, r=45, t=18, b=40), xaxis_side="top", paper_bgcolor="white", plot_bgcolor="white")
    return fig


def _distribution_figure(scenarios):
    fig = go.Figure()
    groups = list(scenarios.groupby(["regime", "flexibility"], sort=True))
    colors = {"annual_expected": "#d98b32", "expected": "#678aa1", "optimization": "#22855a"}
    for group_index, ((regime, flexibility), group) in enumerate(groups):
        for case in ("annual_expected", "expected", "optimization"):
            values = group[group.solution_case == case]
            fig.add_trace(go.Histogram(x=values["total_cost"], name=CASE_LABELS[case], marker_color=colors[case], opacity=.55, nbinsx=22, visible=group_index == 0))
    buttons = []
    for group_index, ((regime, flexibility), _) in enumerate(groups):
        visible = [False] * (len(groups) * 3)
        visible[group_index * 3 : group_index * 3 + 3] = [True] * 3
        buttons.append({"label": f"{REGIME_LABELS[regime]} · {FLEX_LABELS[flexibility]}", "method": "update", "args": [{"visible": visible}, {"title": f"Distribución de costo total · {REGIME_LABELS[regime]} · {FLEX_LABELS[flexibility]}"}]})
    fig.update_layout(barmode="overlay", height=430, title=f"Distribución de costo total · {REGIME_LABELS[groups[0][0][0]]} · {FLEX_LABELS[groups[0][0][1]]}", xaxis_title="Costo anual por escenario", yaxis_title="Número de escenarios", plot_bgcolor="white", updatemenus=[{"buttons": buttons, "x": 0, "y": 1.18, "xanchor": "left", "yanchor": "top"}], legend={"orientation": "h", "y": -0.22})
    return fig


def _decision_comparison(installations):
    pivot = installations.pivot(index=["regime", "flexibility", "facility"], columns="solution_case", values="capacity").reset_index()
    rows = []
    for (regime, flexibility), group in pivot.groupby(["regime", "flexibility"]):
        optimized_total = group["optimization"].sum()
        for baseline in ("annual_expected", "expected"):
            rows.append({"Régimen": REGIME_LABELS[regime], "Política": FLEX_LABELS[flexibility], "Solución comparada": CASE_LABELS[baseline], "Capacidad Y base": group[baseline].sum(), "Capacidad Y optimización": optimized_total, "Satélites con Y distinta": int((group[baseline] != group["optimization"]).sum())})
    return pd.DataFrame(rows)


def _pretty(frame):
    output = frame.copy()
    for source, target, labels in (("regime", "Régimen", REGIME_LABELS), ("flexibility", "Política", FLEX_LABELS), ("solution_case", "Solución Y", CASE_LABELS), ("baseline_solution", "Solución base", CASE_LABELS)):
        if source in output:
            output[target] = output.pop(source).map(labels)
    return output


def _solution_status_table(means: pd.DataFrame) -> pd.DataFrame:
    """Certification status of the source solve that produced each fixed Y."""
    status = means[[
        "regime", "flexibility", "solution_case", "source_status",
        "source_is_optimal", "source_optimality_gap", "source_actual_run_time",
    ]].copy()
    status["source_status"] = status["source_status"].map(SOLVE_STATUS_LABELS).fillna(status["source_status"])
    status["Certificada óptima"] = status.pop("source_is_optimal").map({True: "Sí", False: "No"})
    status.rename(columns={
        "source_status": "Estado de resolución",
        "source_optimality_gap": "Gap final",
        "source_actual_run_time": "Tiempo (s)",
    }, inplace=True)
    return _pretty(status).sort_values(["Régimen", "Política", "Solución Y"])


def build_partial_preview(version: str, output_path: Path | None = None, root: Path | None = None) -> Path:
    """Write a clearly-labelled progress preview while the full VSS batch is running."""
    root = root or (RESULTS_DIR / "flexibility_evaluation")
    output_path = output_path or root / version / "vss_preview_partial.html"
    payloads = [item for item in _load(root, version) if item.get("assignment_variables") == "binary"]
    if not payloads:
        raise FileNotFoundError("No binary evaluation files are available for a preview.")
    means, _, installations = _frames(payloads)
    labels = _pretty(means)
    fig = go.Figure()
    for case, group in means.groupby("solution_case"):
        fig.add_trace(go.Bar(
            name=CASE_LABELS[case],
            x=[f"{REGIME_LABELS[row.regime]}<br>{FLEX_LABELS[row.flexibility]}" for row in group.itertuples()],
            y=group["total_cost"],
            customdata=group[["second_stage_cost"]],
            hovertemplate="%{x}<br>Costo total: %{y:,.0f}<br>Segunda etapa: %{customdata[0]:,.0f}<extra></extra>",
        ))
    fig.update_layout(barmode="group", height=420, yaxis_title="Costo anual", plot_bgcolor="white", paper_bgcolor="white")
    decision_table = _pretty(installations).sort_values(["Régimen", "Política", "Solución Y", "facility"]).to_html(index=False, float_format=lambda x: f"{x:,.0f}")
    means_table = labels.sort_values(["Régimen", "Política", "Solución Y"]).to_html(index=False, float_format=lambda x: f"{x:,.0f}")
    html = f'''<!doctype html><html lang="es"><head><meta charset="utf-8"><title>Vista parcial VSS — {version}</title><script src="{PLOTLY_CDN}"></script><style>{CSS}</style></head><body>
<div id="top"><h1>Vista parcial de evaluaciones binarias</h1><p>Versión {version} · {len(payloads)} de 27 evaluaciones disponibles</p></div>
<main class="wrap"><div class="hero"><div><h2>Progreso, no VSS final</h2><p>Esta vista muestra exclusivamente las evaluaciones binarias ya terminadas. El VSS requiere comparar una misma política y régimen entre `optimization` y una solución base; se calculará sólo cuando el grupo esté completo.</p></div><div class="method"><strong>Interpretación</strong><br>Las barras comparan costos disponibles, pero no constituyen una comparación VSS si faltan soluciones del mismo grupo.</div></div>
<section class="section"><h2>Costos medios disponibles</h2>{_html(fig)}</section>
<section class="section"><h2>Resultados guardados</h2>{means_table}</section>
<section class="section"><h2>Decisiones Y disponibles</h2>{decision_table}</section>
</main></body></html>'''
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    return output_path


def build_report(version: str, output_path: Path | None = None, root: Path | None = None) -> Path:
    root = root or (RESULTS_DIR / "flexibility_evaluation")
    output_path = output_path or root / version / "vss_comparison.html"
    payloads = _load(root, version)
    if len(payloads) != 27:
        raise ValueError(f"Expected 27 evaluation files under {root / version}; found {len(payloads)}.")
    if any(payload.get("assignment_variables") != "binary" for payload in payloads):
        raise ValueError(
            "Evaluation files were generated with continuous or unknown X/W assignments. "
            "Rerun the flexibility experiment and evaluation with --overwrite first."
        )
    means, scenarios, installations = _frames(payloads)
    if len(scenarios) != 2700 or scenarios.groupby(["regime", "flexibility", "solution_case"]).size().nunique() != 1:
        raise ValueError("Every evaluation must contain the same 100 validation-scenario costs.")
    percentiles = _percentiles(scenarios)
    vss = _vss(percentiles)
    theoretical = _theoretical_vss(percentiles, _load_validation_benchmarks(version))
    ptable = _pretty(percentiles).sort_values(["Régimen", "Política", "Solución Y"]).to_html(index=False, float_format=lambda x: f"{x:,.0f}")
    vtable = _pretty(vss).sort_values(["Régimen", "Política", "Solución base"]).to_html(index=False, float_format=lambda x: f"{x:,.2f}")
    decisions = _decision_comparison(installations)
    dtable = decisions.to_html(index=False, float_format=lambda x: f"{x:,.0f}")
    status_table = _solution_status_table(means)
    stable_status_table = status_table.to_html(
        index=False,
        float_format=lambda x: f"{x:,.3f}",
    )
    source_optimal = int(means["source_is_optimal"].sum())
    source_time_limit = int((means["source_status"] == "TIME_LIMIT").sum())
    evaluation_optimal = int(means["evaluation_is_optimal"].sum())
    best = vss.loc[vss["vss_cost_reduction"].idxmax()]
    positive = int((vss["vss_cost_reduction"] > 0.01).sum())
    changed = int((decisions["Satélites con Y distinta"] > 0).sum())
    winners = percentiles.loc[percentiles.groupby(["regime", "flexibility"])["mean_total"].idxmin(), ["regime", "flexibility", "solution_case", "mean_total", "p95_total"]]
    winners.columns = ["regime", "flexibility", "solution_case", "Costo medio", "P95"]
    winners_table = _pretty(winners).sort_values(["Régimen", "Política"]).to_html(index=False, float_format=lambda x: f"{x:,.0f}")
    if theoretical.empty:
        theoretical_section = '<section class="section"><h2>VSS teórico en validation</h2><p>No hay benchmarks RP<sub>100</sub> disponibles todavía. Ejecuta <code>run_validation_benchmark</code> para comparar cada solución base con el óptimo estocástico de los mismos 100 escenarios validation.</p></section>'
    else:
        ttable = _pretty(theoretical).sort_values(["Régimen", "Política", "Solución base"]).to_html(index=False, float_format=lambda x: f"{x:,.2f}")
        exact = int(theoretical["is_exact"].sum())
        theoretical_section = f'<section class="section"><h2>VSS teórico en validation</h2><p>Compara cada solución base contra RP<sub>100</sub>, resuelto sobre los mismos 100 escenarios validation. {exact} de {len(theoretical) // 2} benchmarks están certificados como óptimos.</p><div class="method">Si RP<sub>100</sub> terminó óptimo, <strong>vss_lower = vss_upper</strong> y el VSS es exacto y no negativo. Con <code>TIME_LIMIT</code>, el reporte entrega el intervalo certificado <code>[vss_lower, vss_upper]</code>, usando el incumbente y la cota del solver; no lo presenta como un VSS puntual exacto.</div>{ttable}</section>'
    html = f'''<!doctype html><html lang="es"><head><meta charset="utf-8"><title>VSS flexibilidad — {version}</title><script src="{PLOTLY_CDN}"></script><style>{CSS}</style></head><body>
<div id="top"><h1>¿Cuánto vale optimizar la instalación?</h1><p>Evaluación fuera de muestra · versión {version} · 100 escenarios por régimen</p></div>
<main class="wrap"><div class="hero"><div><h2>La lectura en una frase</h2><p>La matriz muestra el ahorro de elegir la instalación obtenida con 30 escenarios, frente a una solución de promedio anual o de escenario esperado. Verde: optimizar reduce el costo. Rojo: la solución base fue mejor en la muestra de validación.</p></div><div class="method"><strong>Cómo leer VSS</strong><br>VSS = costo medio validación de Y base − costo medio validación de Y optimización. Todos los costos incluyen instalación una sola vez y segunda etapa reoptimizada por escenario.</div></div>
<div class="cards"><div class="metric good"><div class="label">Mayor ahorro VSS</div><div class="value">{best.vss_cost_reduction:,.0f}</div><div class="label">{REGIME_LABELS[best.regime]} · {FLEX_LABELS[best.flexibility]} · vs {CASE_LABELS[best.baseline_solution]}</div></div><div class="metric"><div class="label">Comparaciones con ahorro</div><div class="value">{positive} de {len(vss)}</div><div class="label">Un valor positivo favorece Y de optimización</div></div><div class="metric warn"><div class="label">Soluciones fuente con límite</div><div class="value">{source_time_limit} de {len(means)}</div><div class="label">La Y usada tiene incumbente factible, no certificado óptimo</div></div><div class="metric good"><div class="label">Evaluaciones de recourse óptimas</div><div class="value">{evaluation_optimal} de {len(means)}</div><div class="label">Cada Y se reoptimizó sobre sus 100 escenarios validation</div></div></div>
<section class="section"><h2>VSS: señal principal</h2><p>Cada celda compara la misma política y régimen. El número es ahorro anual medio fuera de muestra.</p>{_html(_vss_figure(vss))}<div class="insight">Resultado más fuerte: <strong>{best.vss_cost_reduction:,.0f}</strong> de ahorro ({best.vss_pct:.2f}%) en <strong>{REGIME_LABELS[best.regime]} / {FLEX_LABELS[best.flexibility]}</strong>, al comparar contra <strong>{CASE_LABELS[best.baseline_solution]}</strong>.</div></section>
<section class="section"><h2>Calidad de resolución de cada instalación Y</h2><p>Esta dimensión se refiere al modelo fuente que generó cada decisión Y, antes de validarla. <strong>{source_optimal} de {len(means)}</strong> terminaron con óptimo certificado; <strong>{source_time_limit}</strong> alcanzaron el límite de tiempo y usan la mejor solución factible encontrada. Las evaluaciones de segunda etapa sí terminaron óptimas en {evaluation_optimal} de {len(means)} casos.</p><div class="method"><strong>Cómo interpretar un límite de tiempo</strong><br>El VSS observado sigue siendo una comparación válida de las Y guardadas, pero no certifica que esa Y sea la mejor posible para su modelo fuente. Revise el gap final: cuanto más cercano a cero, menor la brecha entre el incumbente y la cota del solver.</div>{stable_status_table}</section>
{theoretical_section}
<section class="section"><h2>Qué solución gana en validación</h2><p>La menor media entre las tres instalaciones candidatas. P95 permite ver el costo de un escenario exigente.</p>{winners_table}</section>
<section class="section"><h2>¿La instalación cambia?</h2><p>Resumen de las decisiones Y. Si no cambian satélites o capacidad, un VSS igual a cero es esperado: el subproblema es idéntico.</p>{dtable}</section>
<section class="section"><h2>Distribución de costos</h2><p>El selector cambia régimen y política; se comparan las tres Y sobre exactamente los mismos 100 escenarios.</p>{_html(_distribution_figure(scenarios))}</section>
<section class="section detail"><h2>Detalle auditable</h2><p>Medias y percentiles. Segunda etapa = operación + ruteo de satélites + ruteo directo desde el DC.</p>{ptable}<h2 style="margin-top:28px">Tabla VSS</h2>{vtable}</section>
</main></body></html>'''
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    (output_path.parent / "vss_summary.json").write_text(json.dumps({"version": version, "means": means.to_dict("records"), "percentiles": percentiles.to_dict("records"), "vss_oos": vss.to_dict("records"), "vss_theoretical": theoretical.to_dict("records"), "installations": installations.to_dict("records"), "scenario_costs": scenarios.to_dict("records")}, indent=2))
    return output_path

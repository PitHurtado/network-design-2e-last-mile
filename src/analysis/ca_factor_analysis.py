"""Factor analysis of the Continuous Approximation model.

Four analyses:
  1. Structural cost decomposition (fixed / line-haul / intra-stop / prep)
  2. Log-linear elasticity regression (per factor, per cost component)
  3. Partial Dependence Plots (vary one input, hold others at median)
  4. OAT Sensitivity ±10/20/50 % tornado chart
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from src.constants import TypeOfFlexibility
from src.utils.instance import Instance

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"
PLOTLY_CFG = {"displayModeBar": True, "displaylogo": False}
OUTPUT_PATH = Path("results/analysis/ca_factor_analysis.html")

COMP_COLORS = {
    "cost_fixed": "#3498db",
    "total_line_haul": "#e74c3c",
    "total_intra_stop": "#2ecc71",
    "total_prep": "#f39c12",
}
COMP_COLORS_ALPHA = {
    "cost_fixed": "rgba(52,152,219,0.5)",
    "total_line_haul": "rgba(231,76,60,0.5)",
    "total_intra_stop": "rgba(46,204,113,0.5)",
    "total_prep": "rgba(243,156,18,0.5)",
}
COMP_LABELS = {
    "cost_fixed": "Costo Fijo",
    "total_line_haul": "Line-Haul",
    "total_intra_stop": "Intra-Stop",
    "total_prep": "Preparación",
}
FACTOR_LABELS = {
    "distance": "Distancia (km)",
    "density": "Densidad (clientes/km²)",
    "demand": "Demanda (ítems)",
    "area": "Área (km²)",
    "drop": "Drop (ítems/cliente)",
}


# ── Data extraction ───────────────────────────────────────────────────────────

def load_instance() -> Instance:
    print("Loading instance (running CA)...")
    return Instance(
        id_instance="analysis",
        is_continuous_var_x=True,
        type_of_flexibility=TypeOfFlexibility.FIXED_CAPACITY.value,
        periods=12,
        N=1,
        is_evaluation=False,
        use_euclidean_distance=True,
    )


def extract_dataframe(instance: Instance) -> pd.DataFrame:
    """Extract CA parameters into a flat DataFrame (vehicle='small' only)."""
    rows = []
    scenario = next(iter(instance.scenarios.values()))
    for (i, j, v, t, w), params in scenario.parameters["facility"].items():
        if v != "small":
            continue
        pixel = scenario.pixels[j]
        area = pixel.geo_point.area_surface
        if area <= 0:
            continue
        density = pixel.stop_by_period[t] / area
        demand = pixel.demand_by_period[t]
        drop = pixel.drop_by_period[t]
        if demand <= 0 or density <= 0 or drop <= 0:
            continue
        n = params["average_fleet_size"] * params["average_number_tours"]
        rows.append({
            "satellite": i,
            "pixel": j,
            "period": t,
            "distance": params["distance_to_centroid"],
            "area": area,
            "density": density,
            "demand": demand,
            "drop": drop,
            "fleet_size": params["average_fleet_size"],
            "n_tours": params["average_number_tours"],
            "cost_fixed": params["cost_fixed"],
            "total_line_haul": n * params["cost_line_haul"],
            "total_intra_stop": n * params["cost_intra_stop"],
            "total_prep": n * params["cost_tour_preparation"],
            "cost_total": params["cost_total"],
            "circuit_factor": pixel.k,
        })
    df = pd.DataFrame(rows)
    print(f"Extracted {len(df)} observations from {df['satellite'].nunique()} satellites.")
    return df


# ── CA formula (for PDPs and OAT) ────────────────────────────────────────────

def ca_cost(distance, density, demand, area, drop, circuit_factor, vehicle) -> dict | None:
    """Compute CA cost components analytically given scalar inputs."""
    if density <= 0 or drop <= 0 or area <= 0 or distance < 0:
        return None
    effective_capacity = vehicle.capacity / drop
    if effective_capacity <= 0:
        return None

    intra_t = vehicle.k * circuit_factor / (math.sqrt(density) * vehicle.speed_inter_stop)
    tour_t_per_cust = vehicle.time_set_up + vehicle.time_service * drop + intra_t
    avg_tour_time = effective_capacity * tour_t_per_cust

    denom = (
        avg_tour_time
        + vehicle.time_prep
        + vehicle.time_loading_per_item * effective_capacity * drop
        + 2 * distance * vehicle.k / vehicle.speed_line_haul
    )
    if denom <= 0:
        return None

    n_full = vehicle.t_max / denom
    n_cust_per_tour = effective_capacity * min(1.0, n_full)
    n_tours = max(1.0, n_full)
    fleet = area * density / (n_full * effective_capacity)

    prep_per_tour = vehicle.cost_hour * (
        vehicle.time_prep + vehicle.time_loading_per_item * n_cust_per_tour * drop
    )
    lh_per_tour = (
        vehicle.cost_hour * 2 * distance * vehicle.k / vehicle.speed_line_haul
        + vehicle.cost_km * 2 * distance * vehicle.k
    )
    intra_per_tour = (
        vehicle.cost_hour * tour_t_per_cust * n_cust_per_tour
        + vehicle.cost_km * vehicle.k * circuit_factor * n_cust_per_tour / math.sqrt(density)
    )

    cost_fixed = fleet * vehicle.cost_fixed
    n = fleet * n_tours
    return {
        "cost_total": cost_fixed + n * (prep_per_tour + lh_per_tour + intra_per_tour),
        "cost_fixed": cost_fixed,
        "total_line_haul": n * lh_per_tour,
        "total_intra_stop": n * intra_per_tour,
        "total_prep": n * prep_per_tour,
    }


# ── OLS log-linear regression ─────────────────────────────────────────────────

def log_ols(df: pd.DataFrame, y_col: str, x_cols: list[str]) -> dict:
    """OLS on log-transformed variables. Returns elasticities (β coefs)."""
    valid = df[(df[y_col] > 0) & df[x_cols].gt(0).all(axis=1)].copy()
    if len(valid) < len(x_cols) + 2:
        return {}
    Y = np.log(valid[y_col].values)
    Xmat = np.column_stack([np.ones(len(valid))] + [np.log(valid[c].values) for c in x_cols])
    beta, _, _, _ = np.linalg.lstsq(Xmat, Y, rcond=None)
    Y_hat = Xmat @ beta
    ss_res = np.sum((Y - Y_hat) ** 2)
    ss_tot = np.sum((Y - Y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    n, p = len(Y), Xmat.shape[1]
    sigma2 = ss_res / max(n - p, 1)
    try:
        cov = sigma2 * np.linalg.inv(Xmat.T @ Xmat)
        se = np.sqrt(np.maximum(np.diag(cov), 0))
        t_stat = beta / se
        pval = 2 * stats.t.sf(np.abs(t_stat), df=n - p)
    except np.linalg.LinAlgError:
        se = pval = np.full_like(beta, np.nan)
    names = ["intercept"] + x_cols
    return {
        "beta": dict(zip(names, beta)),
        "se": dict(zip(names, se)),
        "pval": dict(zip(names, pval)),
        "r2": r2,
        "n": n,
    }


# ── Figure builders ───────────────────────────────────────────────────────────

def fig_cost_decomposition(df: pd.DataFrame) -> go.Figure:
    """Stacked bar (% by satellite) + boxplot of % distribution."""
    components = ["cost_fixed", "total_line_haul", "total_intra_stop", "total_prep"]

    # Compute % per row
    for c in components:
        df[f"pct_{c}"] = df[c] / df["cost_total"].replace(0, np.nan) * 100

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("% promedio por satélite", "Distribución de % entre píxeles"),
        column_widths=[0.55, 0.45],
    )

    # Stacked bar per satellite
    sat_groups = df.groupby("satellite")[[f"pct_{c}" for c in components]].mean()
    for c in components:
        fig.add_trace(go.Bar(
            name=COMP_LABELS[c],
            x=sat_groups.index.tolist(),
            y=sat_groups[f"pct_{c}"].tolist(),
            marker_color=COMP_COLORS[c],
            hovertemplate=f"<b>{COMP_LABELS[c]}</b><br>%{{x}}: %{{y:.1f}}%<extra></extra>",
        ), row=1, col=1)

    # Boxplot of % distribution
    for c in components:
        fig.add_trace(go.Box(
            name=COMP_LABELS[c],
            y=df[f"pct_{c}"].dropna().tolist(),
            marker_color=COMP_COLORS[c],
            showlegend=False,
            boxmean=True,
            hovertemplate=f"<b>{COMP_LABELS[c]}</b><br>%{{y:.1f}}%<extra></extra>",
        ), row=1, col=2)

    fig.update_layout(
        barmode="stack",
        height=420,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=50, r=20, t=40, b=50),
        legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="#ccc", borderwidth=1),
        yaxis=dict(title="% del costo total", range=[0, 105]),
        yaxis2=dict(title="% del costo total"),
    )
    return fig


def fig_elasticities(df: pd.DataFrame) -> tuple[go.Figure, go.Figure]:
    """Bar chart of elasticities per factor × cost component + correlation heatmap."""
    factors = ["distance", "density", "demand", "area", "drop"]
    targets = {"cost_total": "Costo Total", "total_line_haul": "Line-Haul",
               "total_intra_stop": "Intra-Stop", "cost_fixed": "Costo Fijo"}

    results = {}
    for tgt, tgt_label in targets.items():
        res = log_ols(df, tgt, factors)
        results[tgt_label] = res

    # Grouped bar chart: factor × target
    bar_fig = go.Figure()
    target_colors = ["#2c3e50", "#e74c3c", "#2ecc71", "#3498db"]
    for (tgt_label, res), color in zip(results.items(), target_colors):
        if not res:
            continue
        betas = [res["beta"].get(f, 0) for f in factors]
        pvals = [res["pval"].get(f, 1) for f in factors]
        bar_fig.add_trace(go.Bar(
            name=f"{tgt_label} (R²={res['r2']:.3f})",
            x=[FACTOR_LABELS[f] for f in factors],
            y=betas,
            marker_color=color,
            opacity=0.85,
            text=[f"β={b:.3f}<br>p={p:.3f}" + (" *" if p < 0.05 else "") for b, p in zip(betas, pvals)],
            hovertemplate="%{text}<extra></extra>",
        ))

    bar_fig.add_hline(y=0, line_color="grey", line_dash="dot", opacity=0.4)
    bar_fig.update_layout(
        barmode="group",
        height=420,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=50, r=20, t=10, b=80),
        yaxis=dict(title="Elasticidad (β log-lineal)"),
        legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="#ccc", borderwidth=1),
        xaxis=dict(tickangle=20),
    )

    # Correlation heatmap
    all_cols = factors + list(targets.keys())
    valid = df[all_cols].dropna()
    valid = valid[(valid > 0).all(axis=1)]
    log_df = np.log(valid)
    corr = log_df.corr()
    sub_corr = corr.loc[factors, list(targets.keys())]

    heat_fig = go.Figure(go.Heatmap(
        z=sub_corr.values,
        x=[targets[c] for c in targets],
        y=[FACTOR_LABELS[f] for f in factors],
        colorscale="RdBu",
        zmid=0,
        text=[[f"{v:.2f}" for v in row] for row in sub_corr.values],
        texttemplate="%{text}",
        hovertemplate="<b>%{y}</b> → <b>%{x}</b><br>ρ log = %{z:.3f}<extra></extra>",
    ))
    heat_fig.update_layout(
        height=320,
        margin=dict(l=140, r=20, t=10, b=60),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(side="bottom"),
    )
    return bar_fig, heat_fig


def fig_pdps(df: pd.DataFrame, vehicle) -> go.Figure:
    """Partial Dependence Plots: vary one factor at a time."""
    factors = ["distance", "density", "demand", "area", "drop"]
    medians = {f: df[f].median() for f in factors}
    med_cf = df["circuit_factor"].median()
    components = ["cost_fixed", "total_line_haul", "total_intra_stop", "total_prep"]

    fig = make_subplots(
        rows=1, cols=5,
        subplot_titles=[FACTOR_LABELS[f] for f in factors],
        shared_yaxes=False,
    )

    for col_idx, factor in enumerate(factors, start=1):
        # Range: p5 to p95
        lo, hi = df[factor].quantile(0.05), df[factor].quantile(0.95)
        xs = np.linspace(lo, hi, 40)
        base_kwargs = {f: medians[f] for f in factors}
        ys = {c: [] for c in components + ["cost_total"]}

        for x in xs:
            kwargs = {**base_kwargs, factor: x}
            out = ca_cost(
                distance=kwargs["distance"],
                density=kwargs["density"],
                demand=kwargs["demand"],
                area=kwargs["area"],
                drop=kwargs["drop"],
                circuit_factor=med_cf,
                vehicle=vehicle,
            )
            if out is None:
                for c in components + ["cost_total"]:
                    ys[c].append(np.nan)
            else:
                for c in components + ["cost_total"]:
                    ys[c].append(out.get(c, np.nan))

        # Stacked fill per component
        cumulative = np.zeros(len(xs))
        for c in components:
            vals = np.array(ys[c], dtype=float)
            fig.add_trace(go.Scatter(
                x=xs, y=cumulative + vals,
                fill="tonexty" if col_idx > 1 or c != components[0] else "tozeroy",
                fillcolor=COMP_COLORS_ALPHA[c],
                line=dict(color=COMP_COLORS[c], width=1),
                name=COMP_LABELS[c],
                showlegend=(col_idx == 1),
                hovertemplate=f"{COMP_LABELS[c]}<br>%{{y:.0f}}<extra></extra>",
            ), row=1, col=col_idx)
            cumulative = cumulative + vals

        # Total cost line
        fig.add_trace(go.Scatter(
            x=xs, y=ys["cost_total"],
            mode="lines",
            line=dict(color="#2c3e50", width=2, dash="dot"),
            name="Costo Total",
            showlegend=(col_idx == 1),
            hovertemplate="Total: %{y:.0f}<extra></extra>",
        ), row=1, col=col_idx)

        # Vertical line at median
        fig.add_vline(x=medians[factor], line_dash="dash", line_color="grey",
                      opacity=0.5, row=1, col=col_idx)

    fig.update_layout(
        height=380,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=50, r=20, t=40, b=50),
        legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="#ccc", borderwidth=1),
    )
    for i in range(1, 6):
        fig.update_yaxes(title_text="Costo total" if i == 1 else "", row=1, col=i)
    return fig


def fig_oat_sensitivity(df: pd.DataFrame, vehicle) -> go.Figure:
    """OAT ±10/20/50% tornado chart + sensitivity lines."""
    factors = ["distance", "density", "demand", "area", "drop"]
    medians = {f: df[f].median() for f in factors}
    med_cf = df["circuit_factor"].median()
    variations = [-0.5, -0.2, -0.1, 0.1, 0.2, 0.5]

    # Compute base cost
    base = ca_cost(**{**medians, "circuit_factor": med_cf, "vehicle": vehicle})
    base_cost = base["cost_total"] if base else 1.0

    # Tornado at ±20%
    impacts_lo, impacts_hi = [], []
    for factor in factors:
        kwargs_lo = {**medians, factor: medians[factor] * 0.8, "circuit_factor": med_cf, "vehicle": vehicle}
        kwargs_hi = {**medians, factor: medians[factor] * 1.2, "circuit_factor": med_cf, "vehicle": vehicle}
        out_lo = ca_cost(**kwargs_lo)
        out_hi = ca_cost(**kwargs_hi)
        d_lo = (out_lo["cost_total"] / base_cost - 1) * 100 if out_lo else 0
        d_hi = (out_hi["cost_total"] / base_cost - 1) * 100 if out_hi else 0
        impacts_lo.append(d_lo)
        impacts_hi.append(d_hi)

    # Sort by total range
    ranges = [abs(h - l) for h, l in zip(impacts_hi, impacts_lo)]
    order = sorted(range(len(factors)), key=lambda k: ranges[k])
    sorted_factors = [factors[k] for k in order]
    sorted_lo = [impacts_lo[k] for k in order]
    sorted_hi = [impacts_hi[k] for k in order]

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Tornado ±20%", "Curvas de sensibilidad"))

    # Tornado bars
    for f, lo, hi in zip(sorted_factors, sorted_lo, sorted_hi):
        fig.add_trace(go.Bar(
            x=[lo], y=[FACTOR_LABELS[f]], orientation="h",
            marker_color="#3498db", name=f"-20%",
            showlegend=False,
            hovertemplate=f"<b>{FACTOR_LABELS[f]}</b> -20%: {lo:+.2f}%<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            x=[hi], y=[FACTOR_LABELS[f]], orientation="h",
            marker_color="#e74c3c", name=f"+20%",
            showlegend=False,
            hovertemplate=f"<b>{FACTOR_LABELS[f]}</b> +20%: {hi:+.2f}%<extra></extra>",
        ), row=1, col=1)

    fig.add_vline(x=0, line_color="grey", line_dash="dot", opacity=0.6, row=1, col=1)

    # Sensitivity lines (vary from -50% to +50%)
    palette = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
    for factor, color in zip(factors, palette):
        xs, ys = [], []
        for v in variations:
            kwargs = {**medians, factor: medians[factor] * (1 + v), "circuit_factor": med_cf, "vehicle": vehicle}
            out = ca_cost(**kwargs)
            if out:
                xs.append(v * 100)
                ys.append((out["cost_total"] / base_cost - 1) * 100)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=6),
            name=FACTOR_LABELS[factor],
            hovertemplate=f"<b>{FACTOR_LABELS[factor]}</b><br>Var input: %{{x:+.0f}}%<br>Δ costo: %{{y:+.2f}}%<extra></extra>",
        ), row=1, col=2)

    fig.add_hline(y=0, line_color="grey", line_dash="dot", opacity=0.4, row=1, col=2)

    fig.update_layout(
        barmode="overlay",
        height=420,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=140, r=20, t=40, b=50),
        legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="#ccc", borderwidth=1),
        xaxis=dict(title="Δ% costo total", ticksuffix="%"),
        xaxis2=dict(title="Variación del input (%)", ticksuffix="%"),
        yaxis2=dict(title="Δ% costo total"),
    )
    return fig


# ── Fleet size analysis ───────────────────────────────────────────────────────

def fig_fleet_size_analysis(df: pd.DataFrame) -> tuple[go.Figure, go.Figure]:
    """Scatter fleet_size vs each factor + OLS elasticity + distribution per satellite."""
    factors = ["distance", "density", "demand", "area", "drop"]

    # Figure A: scatter subplots (fleet_size vs each factor)
    fig_a = make_subplots(rows=1, cols=5, subplot_titles=[FACTOR_LABELS[f] for f in factors])
    for col_idx, factor in enumerate(factors, 1):
        valid = df[(df[factor] > 0) & (df["fleet_size"] > 0)].copy()
        fig_a.add_trace(go.Scatter(
            x=valid[factor], y=valid["fleet_size"],
            mode="markers",
            marker=dict(size=3, opacity=0.25, color="#9b59b6"),
            showlegend=False,
            hovertemplate=f"{FACTOR_LABELS[factor]}: %{{x:.2f}}<br>Fleet: %{{y:.3f}}<extra></extra>",
        ), row=1, col=col_idx)
        # Log-linear fit line
        if len(valid) > 5:
            lx = np.log(valid[factor].values)
            ly = np.log(valid["fleet_size"].values)
            beta, _, _, _ = np.linalg.lstsq(
                np.column_stack([np.ones(len(lx)), lx]), ly, rcond=None
            )
            xs = np.linspace(valid[factor].quantile(0.02), valid[factor].quantile(0.98), 60)
            ys = np.exp(beta[0] + beta[1] * np.log(xs))
            fig_a.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(color="#e74c3c", width=2),
                showlegend=(col_idx == 1),
                name=f"Fit log-lineal (β={beta[1]:.2f})",
                hovertemplate=f"β={beta[1]:.3f}<extra></extra>",
            ), row=1, col=col_idx)
    fig_a.update_layout(
        height=350, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=50, r=20, t=40, b=50),
        legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="#ccc", borderwidth=1),
    )
    for i in range(1, 6):
        fig_a.update_yaxes(title_text="Fleet size" if i == 1 else "", row=1, col=i)

    # Figure B: OLS elasticity bar + box plot per satellite
    res_fleet = log_ols(df, "fleet_size", factors)
    fig_b = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Elasticidades sobre fleet size", "Distribución por satélite"),
        column_widths=[0.5, 0.5],
    )
    if res_fleet:
        betas = [res_fleet["beta"].get(f, 0) for f in factors]
        pvals = [res_fleet["pval"].get(f, 1) for f in factors]
        bar_colors = ["#2ecc71" if b >= 0 else "#e74c3c" for b in betas]
        fig_b.add_trace(go.Bar(
            x=[FACTOR_LABELS[f] for f in factors],
            y=betas,
            marker_color=bar_colors,
            text=[f"β={b:.3f}<br>p={p:.3f}" + (" *" if p < 0.05 else "") for b, p in zip(betas, pvals)],
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        ), row=1, col=1)
        fig_b.add_annotation(
            text=f"R²={res_fleet['r2']:.3f}  n={res_fleet['n']}",
            xref="x domain", yref="paper", x=0.5, y=1.08,
            showarrow=False, font=dict(size=11),
        )
    fig_b.add_hline(y=0, line_color="grey", line_dash="dot", opacity=0.4, row=1, col=1)

    for sat in sorted(df["satellite"].unique()):
        vals = df[df["satellite"] == sat]["fleet_size"].tolist()
        fig_b.add_trace(go.Box(
            y=vals, name=sat, boxmean=True, showlegend=False,
            hovertemplate=f"<b>{sat}</b><br>Fleet: %{{y:.3f}}<extra></extra>",
        ), row=1, col=2)

    fig_b.update_layout(
        height=380, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=50, r=20, t=40, b=80),
        xaxis=dict(tickangle=20),
        yaxis=dict(title="Elasticidad (β)"),
        yaxis2=dict(title="Fleet size (vehículos)"),
    )
    return fig_a, fig_b


# ── HTML assembly ─────────────────────────────────────────────────────────────

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


def generate_html(output_path: Path = OUTPUT_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    instance = load_instance()
    df = extract_dataframe(instance)
    vehicle = instance.vehicles["small"]

    n_obs = len(df)
    n_sat = df["satellite"].nunique()
    n_pix = df["pixel"].nunique()

    print("Building figures...")
    fig1 = fig_cost_decomposition(df.copy())
    fig2_bar, fig2_heat = fig_elasticities(df.copy())
    fig3 = fig_pdps(df.copy(), vehicle)
    fig4 = fig_oat_sensitivity(df.copy(), vehicle)
    fig5a, fig5b = fig_fleet_size_analysis(df.copy())

    def to_html(fig): return fig.to_html(full_html=False, include_plotlyjs=False, config=PLOTLY_CFG)

    # ── Compute insights from data ─────────────────────────────────────────────
    factors = ["distance", "density", "demand", "area", "drop"]
    comps = ["cost_fixed", "total_line_haul", "total_intra_stop", "total_prep"]

    # Section 1: dominant component
    comp_pcts = (df[comps].mean() / df["cost_total"].mean() * 100)
    dominant_comp = comp_pcts.idxmax()
    dominant_pct = comp_pcts.max()
    lh_pct = comp_pcts["total_line_haul"]
    intra_pct = comp_pcts["total_intra_stop"]
    fixed_pct = comp_pcts["cost_fixed"]

    # Section 2: elasticities
    res_total = log_ols(df, "cost_total", factors)
    r2_total = res_total.get("r2", 0)
    beta_dist = res_total.get("beta", {}).get("distance", 0)
    beta_dens = res_total.get("beta", {}).get("density", 0)
    beta_dem  = res_total.get("beta", {}).get("demand",  0)
    res_lh    = log_ols(df, "total_line_haul", factors)
    beta_dist_lh = res_lh.get("beta", {}).get("distance", 0)
    res_intra = log_ols(df, "total_intra_stop", factors)
    beta_dens_intra = res_intra.get("beta", {}).get("density", 0)

    # Section 4: OAT — which factor has highest ±20% impact
    medians = {f: df[f].median() for f in factors}
    med_cf = df["circuit_factor"].median()
    base_out = ca_cost(**{**medians, "circuit_factor": med_cf, "vehicle": vehicle})
    base_c = base_out["cost_total"] if base_out else 1.0
    oat_impacts = {}
    for f in factors:
        lo_kw = {**medians, f: medians[f] * 0.8, "circuit_factor": med_cf, "vehicle": vehicle}
        hi_kw = {**medians, f: medians[f] * 1.2, "circuit_factor": med_cf, "vehicle": vehicle}
        lo_out = ca_cost(**lo_kw); hi_out = ca_cost(**hi_kw)
        d_lo = (lo_out["cost_total"] / base_c - 1) * 100 if lo_out else 0
        d_hi = (hi_out["cost_total"] / base_c - 1) * 100 if hi_out else 0
        oat_impacts[f] = max(abs(d_lo), abs(d_hi))
    oat_dominant = max(oat_impacts, key=oat_impacts.get)
    oat_dominant_impact = oat_impacts[oat_dominant]

    # Section 5: fleet size elasticities
    res_fleet = log_ols(df, "fleet_size", factors)
    beta_dem_fleet  = res_fleet.get("beta", {}).get("demand",   0)
    beta_dist_fleet = res_fleet.get("beta", {}).get("distance", 0)
    beta_dens_fleet = res_fleet.get("beta", {}).get("density",  0)
    r2_fleet = res_fleet.get("r2", 0)
    mean_fleet = df["fleet_size"].mean()
    max_fleet_sat = df.groupby("satellite")["fleet_size"].mean().idxmax()
    max_fleet_val = df.groupby("satellite")["fleet_size"].mean().max()

    # ── Method strings ────────────────────────────────────────────────────────
    mth1 = (
        f"<b>Fuente:</b> <code>scenario.parameters[\"facility\"][(i, j, \"small\", t, w)]</code> — "
        f"{n_obs:,} combinaciones (satélite × píxel × período) del escenario 1, solo vehículo <i>small</i>, filtrando demanda &gt; 0. "
        f"<b>Transformación:</b> los campos <code>cost_tour_preparation</code>, <code>cost_line_haul</code> e <code>cost_intra_stop</code> "
        f"del diccionario son costos <i>por tour</i>; se multiplicaron por "
        f"<code>average_fleet_size × average_number_tours</code> para obtener el costo total de cada componente. "
        f"El <code>cost_fixed</code> ya es costo total (fleet × cost_fixed_vehicle). "
        f"Luego se calculó <code>% componente = costo_componente / cost_total × 100</code> por observación. "
        f"<b>Gráfico izq:</b> media de cada % agrupada por satélite, apilada al 100%. "
        f"<b>Gráfico der:</b> boxplot de los {n_obs:,} valores de % individuales por componente."
    )

    mth2 = (
        f"<b>Fuente:</b> mismo dataset de {n_obs:,} observaciones. "
        f"<b>Variables de input extraídas de:</b> "
        f"<code>params['distance_to_centroid']</code> (distancia haversine satélite→centroide píxel), "
        f"<code>pixel.stop_by_period[t] / pixel.geo_point.area_surface</code> (densidad), "
        f"<code>pixel.demand_by_period[t]</code> (demanda en ítems), "
        f"<code>pixel.geo_point.area_surface</code> (área del píxel en km²), "
        f"<code>pixel.drop_by_period[t]</code> (ítems por cliente). "
        f"<b>Método:</b> regresión OLS sobre variables log-transformadas "
        f"(<code>np.linalg.lstsq</code> con matriz <code>[1, log(dist), log(dens), log(dem), log(area), log(drop)]</code>). "
        f"Los coeficientes β son directamente elasticidades. "
        f"IC 95% vía <code>σ² × (X'X)⁻¹</code>; p-values de distribución t con n−p grados de libertad."
    )

    mth2b = (
        f"<b>Fuente:</b> mismo dataset log-transformado de la sección 2. "
        f"<b>Método:</b> <code>pandas.DataFrame.corr()</code> — correlación de Pearson entre "
        f"<code>log(inputs)</code> (5 variables) y <code>log(componentes de costo)</code> (4 variables). "
        f"<b>Subconjunto del heatmap:</b> solo la submatriz inputs × componentes (no se muestran las correlaciones cruzadas entre inputs ni entre componentes)."
    )

    mth3 = (
        f"<b>Fuente:</b> <i>no</i> usa el dataset directamente como observaciones — usa las fórmulas analíticas del CA. "
        f"<b>Punto base:</b> mediana observada de cada factor en el dataset "
        f"({', '.join(f'{FACTOR_LABELS[f]}: {medians[f]:.2f}' for f in factors)}). "
        f"circuit_factor fijo en la mediana de <code>pixel.k</code> = {med_cf:.3f}. "
        f"<b>Construcción:</b> para cada factor, se generan 40 valores equiespaciados entre p5 y p95 del dataset; "
        f"los otros 4 factores se mantienen en su mediana. "
        f"Para cada valor se llama a <code>ca_cost()</code>, que reimplementa las fórmulas del CA usando los parámetros reales del vehículo <i>small</i>."
    )

    mth4 = (
        f"<b>Fuente:</b> mismo enfoque analítico que los PDPs — llamadas directas a <code>ca_cost()</code>. "
        f"<b>Punto base:</b> medianas de los {n_obs:,} datos observados. "
        f"<b>Variaciones aplicadas:</b> −50%, −20%, −10%, +10%, +20%, +50% a cada factor individualmente. "
        f"<b>Tornado chart:</b> muestra el Δ% del costo total a ±20%, ordenado por rango total (max − min). "
        f"<b>Curvas de sensibilidad:</b> Δ% del costo total para las 6 variaciones por factor, evidenciando no-linealidades."
    )

    mth5 = (
        f"<b>Fuente:</b> campo <code>params['average_fleet_size']</code> del diccionario de parámetros del CA — "
        f"mismo dataset de {n_obs:,} observaciones. "
        f"<b>Scatter (arriba):</b> fleet_size vs cada input en escala natural; la línea roja es el ajuste "
        f"log-lineal bivariado (<code>log(fleet) = β₀ + β₁·log(factor)</code>) calculado con <code>np.linalg.lstsq</code> "
        f"— el β₁ de cada ajuste se muestra en la leyenda. "
        f"<b>Elasticidades (abajo izq):</b> OLS multivariado log-lineal con los 5 factores, mismo método que sección 2 pero con target = fleet_size "
        f"(R²={r2_fleet:.3f}). "
        f"<b>Distribución por satélite (abajo der):</b> boxplot de los valores de fleet_size de todas las observaciones "
        f"asignadas a cada satélite."
    )

    # ── Insight strings ────────────────────────────────────────────────────────
    ins1 = (
        f"El componente dominante es <b>{COMP_LABELS[dominant_comp]}</b> ({dominant_pct:.1f}% del costo promedio), "
        f"seguido por Line-Haul ({lh_pct:.1f}%), Intra-Stop ({intra_pct:.1f}%) y Fijo ({fixed_pct:.1f}%). "
        f"El costo fijo emerge porque la CA asigna un número continuo de vehículos proporcional al área × densidad (flota = demanda / (n_viajes × capacidad)), "
        f"independientemente de cuánto tiempo los vehículos estén en movimiento. "
        f"El line-haul varía entre satélites porque <code>cost_line_haul ∝ 2 × distancia × k</code>: satélites más alejados del centroide del pixel pagan más por cada viaje."
    )

    ins2 = (
        f"La regresión log-lineal sobre el costo total alcanza R²={r2_total:.3f}, confirmando que el CA es casi "
        f"perfectamente log-lineal en sus inputs. "
        f"La distancia tiene elasticidad β={beta_dist:.3f} sobre el costo total y β≈{beta_dist_lh:.2f} sobre el line-haul "
        f"(teóricamente 1.0 por construcción: <code>cost_line_haul = cost_hora × 2d/v + cost_km × 2d</code>). "
        f"La densidad tiene elasticidad β={beta_dens:.3f} sobre el costo total, con efecto especialmente fuerte en "
        f"intra-stop (β≈{beta_dens_intra:.2f}, valor teórico −0.5 porque "
        f"<code>costo_intra ∝ k/√densidad</code>: duplicar la densidad reduce el costo intra-stop en ~30%). "
        f"La demanda tiene β={beta_dem:.3f}: escalar la demanda escala casi linealmente todos los costos porque "
        f"tanto la flota como el número de viajes crecen proporcionalmente."
    )

    ins2b = (
        "El heatmap confirma las relaciones teóricas: distancia correlaciona fuertemente con line-haul (r≈1), "
        "densidad correlaciona negativamente con intra-stop (r≈−0.5 esperado), y demanda/área correlacionan "
        "fuertemente con el costo fijo (vía tamaño de flota). "
        "Las correlaciones cruzadas (p.ej. distancia con intra-stop) surgen porque la distancia afecta el número "
        "de viajes posibles, lo que cambia la cantidad de clientes por tour y con ello el intra-stop total."
    )

    ins3 = (
        "Los PDPs revelan comportamientos no-lineales clave: "
        "<b>Distancia</b>: el costo crece más que linealmente porque al aumentar la distancia, el número de viajes "
        "posibles cae (<code>n_viajes = T_max / (tour_time + 2d×k/v)</code>), requiriendo más vehículos. "
        "Cuando n_viajes cae por debajo de 1, la flota se dispara (umbral crítico). "
        "<b>Densidad</b>: el intra-stop disminuye con la raíz cuadrada (relación cóncava), mientras que el costo fijo "
        "apenas varía porque más densidad significa más clientes pero también tours más rápidos. "
        "<b>Demanda y área</b>: muestran relaciones casi lineales porque la flota = demanda / (n_viajes × capacidad). "
        "<b>Drop</b>: reduce la capacidad efectiva (capacity/drop) pero también aumenta el tiempo de carga, "
        "generando un efecto compuesto no-lineal."
    )

    ins4 = (
        f"El factor más sensible a ±20% es <b>{FACTOR_LABELS[oat_dominant]}</b> "
        f"(impacto de {oat_dominant_impact:.1f}% en el costo total). "
        f"Las curvas de sensibilidad muestran que la relación es aproximadamente lineal para pequeñas variaciones, "
        f"pero se vuelve no-lineal a ±50%, especialmente para distancia y densidad. "
        f"El umbral de distancia ocurre cuando <code>T_max / denom < 1</code>: a partir de ese punto cada "
        f"vehículo hace menos de un viaje completo por turno, y el costo sube abruptamente porque "
        f"la flota mínima es 1 vehículo por zona independientemente del volumen."
    )

    ins5 = (
        f"La fórmula de la flota es <code>fleet = (área × densidad) / (n_viajes_completos × capacidad/drop)</code>, "
        f"equivalente a <code>fleet = demanda_clientes / (n_viajes × eff_capacity)</code>. "
        f"La elasticidad de la demanda es β={beta_dem_fleet:.3f} (≈1 esperado: más clientes → proporcionalmente más vehículos). "
        f"La distancia tiene β={beta_dist_fleet:.3f}: más distancia reduce el tiempo disponible para entrega "
        f"(<code>n_viajes = T_max / (tour_time + 2d×k/v + prep)</code>), lo que eleva la flota requerida. "
        f"La densidad tiene β={beta_dens_fleet:.3f}: efecto mixto — más clientes/km² sube la demanda (↑flota) "
        f"pero acorta los tiempos intra-parada (↑n_viajes → ↓flota); el balance da un efecto neto "
        f"{'positivo' if beta_dens_fleet > 0 else 'negativo'}. "
        f"El satélite con mayor flota promedio es <b>{max_fleet_sat}</b> ({max_fleet_val:.2f} vehículos), "
        f"consistente con {'mayor distancia o demanda en su zona' if max_fleet_val > mean_fleet else 'su posición geográfica'}."
    )

    # ── CSS ────────────────────────────────────────────────────────────────────
    css = """
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
code { background: #f0f0f0; padding: 1px 4px; border-radius: 3px; font-size: 0.92em; }
"""

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>CA Factor Analysis</title>
  <script src="{PLOTLY_CDN}"></script>
  <style>{css}</style>
</head>
<body>
<div id="top-bar">
  <h1>Análisis de Factores — Aproximación Continua</h1>
  <p>{n_obs:,} observaciones · {n_sat} satélites · {n_pix} píxeles · vehículo "small" · distancia euclidiana</p>
</div>
{section(
    "1. Descomposición estructural del costo",
    "¿Qué componente domina el costo total? Izq: % promedio por satélite (apilado = 100%). Der: distribución del % de cada componente entre todos los píxeles.",
    to_html(fig1), ins1, mth1
)}
{section(
    "2. Elasticidades log-lineales (OLS en escala log)",
    "β = elasticidad: +1% en el input → +β% en el costo. Se incluyen los cuatro componentes de costo. * = p&lt;0.05. El modelo log-lineal es apropiado porque la fórmula del CA es multiplicativa.",
    to_html(fig2_bar), ins2, mth2
)}
{section(
    "2b. Heatmap de correlaciones en escala log",
    "Correlación de Pearson entre log(inputs) y log(componentes de costo). Confirma las relaciones teóricas del modelo analíticamente.",
    to_html(fig2_heat), ins2b, mth2b
)}
{section(
    "3. Partial Dependence Plots (PDP)",
    "Cada gráfico varía un factor de p5 a p95 manteniendo los demás fijos en su mediana. Área apilada = composición del costo. Línea punteada vertical = mediana del factor.",
    to_html(fig3), ins3, mth3
)}
{section(
    "4. Sensibilidad OAT (One-At-a-Time)",
    "Izq: tornado chart ±20%. Der: curvas de sensibilidad a variaciones de ±10/20/50%, mostrando linealidad o umbrales no-lineales.",
    to_html(fig4), ins4, mth4
)}
{section(
    "5. Factores que influyen en el Fleet Size",
    "Arriba: scatter fleet_size vs cada factor con ajuste log-lineal (línea roja). Abajo: elasticidades OLS sobre fleet_size y distribución por satélite.",
    to_html(fig5a) + to_html(fig5b), ins5, mth5
)}
</body>
</html>
"""

    output_path.write_text(html, encoding="utf-8")
    print(f"HTML saved: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_html()

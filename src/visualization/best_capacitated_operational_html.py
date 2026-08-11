"""Summary HTML for best_capacitated_operational results.

Same structure as best_capacitated_html.py but for the flex operational model:
  - Includes cost_operation component
  - Includes Z-variable operational level analysis
  - Capacity utilization from X + fleet_size
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly

RESULTS_DIR = Path("results/best_capacitated_operational")
OUTPUT      = RESULTS_DIR / "summary.html"
PLOTLY_CFG  = {"displayModeBar": True, "displaylogo": False}

PALETTE = ["#e6194b","#3cb44b","#4363d8","#f58231","#911eb4",
           "#42d4f4","#f032e6","#bfef45","#fabed4","#469990"]
ALL_SATS = ["ABAROA","ACHACHICALA","COTA_COTA","LLOJETA","LOS_PINOS",
            "MALLASA","PERIFERICA","SOPOCACHI","ZONA_CEMENTERIO"]
SAT_COLOR = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(ALL_SATS)}


def _hex_rgba(hex_color: str, alpha: float = 0.12) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _load(folder: Path) -> list[dict]:
    records = []
    for p in sorted(folder.glob("best_*_N30.json"),
                    key=lambda x: int(x.name.split("_")[1].replace("active", ""))):
        d   = json.loads(p.read_text())
        N   = d.get("N", 30)
        cap = d.get("chosen_capacity", {})
        si  = d.get("solver_info", {})
        n_active = sum(1 for v in cap.values() if v.get("vehicles", 0) > 0)

        c_fix = d["cost_installation"]
        c_op  = d["cost_operation"] / N
        c_sat = d["cost_served_from_facilities"] / N
        c_dc  = d["cost_served_from_dc"] / N

        utilization = _compute_utilization(d, cap, N)
        op_profile  = _compute_op_profile(d, cap, N)

        records.append({
            "file":       p.name,
            "label":      d["configuration"],
            "n_active":   n_active,
            "N":          N,
            "objective":  d["objective"],
            "c_fix":      c_fix,
            "c_op":       c_op,
            "c_sat":      c_sat,
            "c_dc":       c_dc,
            "time_s":     si.get("actual_run_time", 0),
            "gap_pct":    si.get("optimality_gap", 0),
            "best_bound": si.get("best_bound_value", 0),
            "cap":        cap,
            "utilization": utilization,
            "op_profile":  op_profile,
        })
    return records


def _compute_utilization(d: dict, cap: dict, N: int) -> dict:
    X_raw  = d.get("X", {})
    fs_raw = d.get("fleet_size_serving", {}).get("facility", {})
    fs: dict[tuple, float] = {}
    for k, v in fs_raw.items():
        try: fs[ast.literal_eval(k)] = v
        except: pass
    X: dict[tuple, float] = {}
    for k, v in X_raw.items():
        try: X[ast.literal_eval(k)] = v
        except: pass

    result = {}
    for sat, cap_info in cap.items():
        installed = cap_info.get("vehicles", 0)
        if installed == 0:
            result[sat] = {"installed": 0, "avg_util": 0.0, "max_util": 0.0, "p90_util": 0.0}
            continue
        util_per_tn: list[float] = []
        for t in range(12):
            for n in [str(i) for i in range(1, N + 1)]:
                total_fleet = sum(
                    X.get((sat, j, t, n), 0) * fs.get((sat, j, "small", t, n), 0)
                    for j in {k[1] for k in X if k[0] == sat and k[2] == t and k[3] == n}
                )
                util_per_tn.append(total_fleet / installed * 100)
        result[sat] = {
            "installed":  installed,
            "avg_util":   float(np.mean(util_per_tn)) if util_per_tn else 0.0,
            "max_util":   float(np.max(util_per_tn))  if util_per_tn else 0.0,
            "p90_util":   float(np.percentile(util_per_tn, 90)) if util_per_tn else 0.0,
        }
    return result


def _compute_op_profile(d: dict, cap: dict, N: int) -> dict:
    """For each active satellite: fraction of periods it OPERATED (Z>0) vs stayed idle."""
    op_counts = d.get("operational_level_counts", {})
    result = {}
    for sat, cap_info in cap.items():
        installed = cap_info.get("vehicles", 0)
        if installed == 0:
            result[sat] = {"pct_operating": 0.0, "pct_idle": 100.0}
            continue
        counts = op_counts.get(sat, {})
        total_active = sum(v for k, v in counts.items() if int(k) > 0)
        total_idle   = counts.get("0", 0)
        total = total_active + total_idle
        result[sat] = {
            "pct_operating": round(total_active / total * 100, 1) if total > 0 else 0.0,
            "pct_idle":      round(total_idle   / total * 100, 1) if total > 0 else 100.0,
        }
    return result


def _fig_objective(records: list[dict]) -> go.Figure:
    labels = [f"{r['n_active']} activos" for r in records]
    fig = go.Figure()
    for col, name, key in [
        ("#2c3e50", "Costo fijo instalación",    "c_fix"),
        ("#e67e22", "Costo operacional E[·]",    "c_op"),
        ("#3498db", "Transp. satélites E[·]",    "c_sat"),
        ("#e74c3c", "Transp. DC E[·]",           "c_dc"),
    ]:
        vals = [r[key] for r in records]
        fig.add_trace(go.Bar(name=name, x=labels, y=vals, marker_color=col,
            text=[f"${v:,.0f}" for v in vals], textposition="inside",
            textfont=dict(color="white", size=9)))
    fig.update_layout(
        barmode="stack", height=420,
        title=dict(text="Descomposición del objetivo por configuración", x=0.5, font=dict(size=13)),
        yaxis=dict(title="$", gridcolor="#eee", tickformat="$,.0f"),
        xaxis=dict(title=""),
        legend=dict(x=0.98, y=0.99, xanchor="right"),
        paper_bgcolor="#f4f6f8", plot_bgcolor="white",
        margin=dict(l=60, r=10, t=44, b=40),
        annotations=[dict(x=f"{r['n_active']} activos",
                          y=r["objective"] + 15000,
                          text=f"<b>${r['objective']:,.0f}</b>",
                          showarrow=False, font=dict(size=10))
                     for r in records],
    )
    return fig


def _fig_frontier(records: list[dict]) -> go.Figure:
    x = [r["n_active"] for r in records]
    y = [r["objective"] for r in records]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines+markers+text",
        line=dict(color="#2c3e50", width=2.5),
        marker=dict(size=10, color=[PALETTE[i % len(PALETTE)] for i in x]),
        text=[f"${v:,.0f}" for v in y],
        textposition="top center", textfont=dict(size=9),
        hovertemplate="<b>%{x} activos</b><br>$%{y:,.0f}<extra></extra>",
    ))
    min_idx = y.index(min(y))
    fig.add_trace(go.Scatter(
        x=[x[min_idx]], y=[y[min_idx]], mode="markers",
        marker=dict(size=16, color="#27ae60", symbol="star"),
        name=f"Óptimo: {x[min_idx]} satélites",
    ))
    fig.update_layout(
        height=380,
        title=dict(text="Frontera: objetivo vs satélites activos", x=0.5, font=dict(size=13)),
        xaxis=dict(title="# Satélites activos", tickvals=x, gridcolor="#eee"),
        yaxis=dict(title="Objetivo ($)", gridcolor="#eee", tickformat="$,.0f"),
        paper_bgcolor="#f4f6f8", plot_bgcolor="white",
        margin=dict(l=70, r=10, t=44, b=50),
        showlegend=True, legend=dict(x=0.98, y=0.99, xanchor="right"),
    )
    return fig


def _fig_utilization(records: list[dict]) -> go.Figure:
    configs = [f"{r['n_active']} act." for r in records]
    z, text_vals = [], []
    for sat in ALL_SATS:
        row_z, row_t = [], []
        for r in records:
            info = r["utilization"].get(sat, {})
            inst = info.get("installed", 0)
            p90  = info.get("p90_util", 0.0)
            if inst == 0:
                row_z.append(None); row_t.append("—")
            else:
                row_z.append(round(p90, 1))
                row_t.append(f"{p90:.0f}%\n({inst}v)")
        z.append(row_z); text_vals.append(row_t)

    fig = go.Figure(go.Heatmap(
        z=z, x=configs, y=[s.replace("_", " ").title() for s in ALL_SATS],
        text=text_vals, texttemplate="%{text}", textfont=dict(size=9),
        colorscale=[[0, "#f0f0f0"], [0.5, "#f39c12"], [0.8, "#e74c3c"], [1, "#7f0000"]],
        zmin=0, zmax=100,
        colorbar=dict(title="Util. P90 %", thickness=12, len=0.8),
        hovertemplate="<b>%{y}</b><br>%{x}<br>Util P90: %{z:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        height=400,
        title=dict(text="Utilización P90 capacidad instalada (% flota usada / instalada)", x=0.5, font=dict(size=13)),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="#f4f6f8",
        margin=dict(l=130, r=60, t=44, b=50),
    )
    return fig


def _fig_op_profile(records: list[dict]) -> go.Figure:
    """Stacked bar: % periods operating vs idle per satellite per config."""
    configs = [f"{r['n_active']} act." for r in records]
    fig = go.Figure()
    for i_sat, sat in enumerate(ALL_SATS):
        pct_op = []
        for r in records:
            prof = r["op_profile"].get(sat, {})
            pct_op.append(prof.get("pct_operating", 0.0))
        fig.add_trace(go.Bar(
            name=sat.replace("_", " ").title(),
            x=configs, y=pct_op,
            marker_color=SAT_COLOR[sat], opacity=0.85,
            hovertemplate=f"<b>{sat.replace('_',' ').title()}</b><br>%{{x}}: %{{y:.1f}}% operando<extra></extra>",
        ))
    fig.update_layout(
        barmode="group", height=380,
        title=dict(text="% períodos × escenarios en que cada satélite decide OPERAR (Z>0)", x=0.5, font=dict(size=13)),
        yaxis=dict(title="% períodos operando", gridcolor="#eee", range=[0, 110]),
        xaxis=dict(title=""),
        legend=dict(x=1.01, y=1, xanchor="left", font=dict(size=9)),
        paper_bgcolor="#f4f6f8", plot_bgcolor="white",
        margin=dict(l=60, r=130, t=44, b=40),
    )
    return fig


def _fig_cap_heatmap(records: list[dict]) -> go.Figure:
    configs = [f"{r['n_active']} act." for r in records]
    z, text_vals = [], []
    for sat in ALL_SATS:
        row_z, row_t = [], []
        for r in records:
            info = r["cap"].get(sat, {})
            veh  = info.get("vehicles", 0)
            row_z.append(veh)
            row_t.append(f"{veh}v" if veh > 0 else "—")
        z.append(row_z); text_vals.append(row_t)

    fig = go.Figure(go.Heatmap(
        z=z, x=configs, y=[s.replace("_", " ").title() for s in ALL_SATS],
        text=text_vals, texttemplate="%{text}", textfont=dict(size=11),
        colorscale=[[0, "#f0f0f0"], [0.01, "#d5e8f9"], [0.5, "#3498db"], [1, "#1a5276"]],
        zmin=0, zmax=20,
        colorbar=dict(title="Vehículos", thickness=12, len=0.8),
        hovertemplate="<b>%{y}</b><br>%{x}<br>Instalado: %{z}v<extra></extra>",
    ))
    fig.update_layout(
        height=380,
        title=dict(text="Capacidad instalada por satélite y configuración (vehículos)", x=0.5, font=dict(size=13)),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="#f4f6f8",
        margin=dict(l=130, r=60, t=44, b=50),
    )
    return fig


def _fig_solver_table(records: list[dict]) -> go.Figure:
    row_fill = []
    for r in records:
        sats = [s for s, v in r["cap"].items() if v.get("vehicles", 0) > 0]
        c = SAT_COLOR.get(sats[0], "#999") if sats else "#999999"
        row_fill.append(_hex_rgba(c))

    cap_str = [
        " · ".join(f"{s.split('_')[0].title()}:{v['vehicles']}v"
                   for s, v in sorted(r["cap"].items()) if v.get("vehicles", 0) > 0)
        or "DC_only"
        for r in records
    ]

    fig = go.Figure(go.Table(
        columnwidth=[55, 100, 45, 90, 90, 90, 90, 65, 60, 80, 120],
        header=dict(
            values=["<b># Act.</b>", "<b>Config</b>", "<b>N</b>",
                    "<b>Objetivo</b>", "<b>C.Fijo</b>",
                    "<b>C.Oper E[·]</b>", "<b>Transp.Sat E[·]</b>",
                    "<b>Transp.DC E[·]</b>", "<b>Tiempo(s)</b>",
                    "<b>Gap%</b>", "<b>Capacidades</b>"],
            fill_color="#2c3e50", font=dict(color="white", size=10),
            align="center", height=36,
        ),
        cells=dict(
            values=[
                [r["n_active"] for r in records],
                [r["label"] for r in records],
                [r["N"] for r in records],
                [f"${r['objective']:,.0f}" for r in records],
                [f"${r['c_fix']:,.0f}" for r in records],
                [f"${r['c_op']:,.0f}" for r in records],
                [f"${r['c_sat']:,.0f}" for r in records],
                [f"${r['c_dc']:,.0f}" for r in records],
                [f"{r['time_s']:.1f}s" for r in records],
                [f"{r['gap_pct']:.3f}%" for r in records],
                cap_str,
            ],
            fill_color=[row_fill] * 11,
            align=["center", "left", "center"] + ["right"] * 5 + ["center", "center", "left"],
            font=dict(size=10, color="#2c3e50"),
            height=26,
        ),
    ))
    fig.update_layout(
        paper_bgcolor="#f4f6f8",
        margin=dict(l=0, r=0, t=0, b=0),
        height=len(records) * 26 + 80,
    )
    return fig


def generate(folder: Path = RESULTS_DIR) -> Path:
    print("Loading results...")
    records = _load(folder)
    if not records:
        raise FileNotFoundError(f"No best_*_N30.json files in {folder}")
    N       = records[0]["N"]
    optimal = min(records, key=lambda r: r["objective"])

    print("Building figures...")
    figs = {
        "objective":    _fig_objective(records),
        "frontier":     _fig_frontier(records),
        "cap_heatmap":  _fig_cap_heatmap(records),
        "utilization":  _fig_utilization(records),
        "op_profile":   _fig_op_profile(records),
        "solver_table": _fig_solver_table(records),
    }

    _js_path = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    _js_tag  = f"<script>{_js_path.read_text()}</script>"

    def _div(fig, div_id):
        return pio.to_html(fig, include_plotlyjs=False, full_html=False,
                           div_id=div_id, config=PLOTLY_CFG)

    css = """
body{font-family:sans-serif;margin:0;padding:0;background:#f4f6f8;}
h1{color:#fff;margin:0;font-size:1.2em;font-weight:600;}
#top-bar{position:sticky;top:0;z-index:100;background:#2c3e50;padding:10px 24px;
 display:flex;align-items:center;gap:16px;box-shadow:0 2px 6px rgba(0,0,0,.25);}
.page{padding:16px;}
.section{background:white;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.10);
 margin-bottom:14px;padding:14px;}
.section h2{color:#2c3e50;margin:0 0 4px;font-size:1.05em;}
.section p{color:#7f8c8d;margin:0 0 10px;font-size:.85em;}
.row{display:flex;gap:14px;margin-bottom:14px;}
.row .section{flex:1;}
.kpi-grid{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:14px;}
.kpi{background:white;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.10);
 padding:14px 18px;min-width:130px;flex:1;}
.kpi .val{font-size:1.5em;font-weight:700;color:#2c3e50;}
.kpi .lbl{font-size:.78em;color:#7f8c8d;margin-top:2px;}
"""

    kpi_html = f"""
<div class="kpi-grid">
  <div class="kpi"><div class="val">{optimal['n_active']}</div><div class="lbl">Satélites activos óptimo</div></div>
  <div class="kpi"><div class="val">${optimal['objective']:,.0f}</div><div class="lbl">Objetivo (N={N})</div></div>
  <div class="kpi"><div class="val">${optimal['c_fix']:,.0f}</div><div class="lbl">Costo fijo instalación</div></div>
  <div class="kpi"><div class="val">${optimal['c_op']:,.0f}</div><div class="lbl">Costo operacional E[·]</div></div>
  <div class="kpi"><div class="val">${optimal['c_sat']:,.0f}</div><div class="lbl">Transp. satélites E[·]</div></div>
  <div class="kpi"><div class="val">${optimal['c_dc']:,.0f}</div><div class="lbl">Transp. DC E[·]</div></div>
  <div class="kpi"><div class="val">{optimal['gap_pct']:.3f}%</div><div class="lbl">MIP gap</div></div>
  <div class="kpi"><div class="val">{optimal['time_s']:.0f}s</div><div class="lbl">Tiempo solver</div></div>
</div>"""

    html = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<title>Best Capacitado Operacional — Summary</title>
{_js_tag}<style>{css}</style>
</head><body>
<div id="top-bar">
  <h1>Mejores Configuraciones — Capacitado + Costos Operacionales (Flex Z) — N={N} · X binario · Propuesta A</h1>
</div>
<div class="page">
  {kpi_html}

  <div class="section">
    <h2>Métricas del solver y descomposición del objetivo</h2>
    <p>C.Oper E[·] = costo operacional esperado — paga solo cuando Z=1 (decide operar). E[·] = promedio sobre {N} escenarios.</p>
    {_div(figs['solver_table'], 'fig_solver_table')}
  </div>

  <div class="row">
    <div class="section">
      <h2>Frontera costo vs satélites activos</h2>
      {_div(figs['frontier'], 'fig_frontier')}
    </div>
    <div class="section">
      <h2>Descomposición del objetivo por componente</h2>
      <p>Costo operacional (naranja) es la novedad vs el modelo sin costos operacionales.</p>
      {_div(figs['objective'], 'fig_objective')}
    </div>
  </div>

  <div class="row">
    <div class="section">
      <h2>Niveles de capacidad instalados</h2>
      {_div(figs['cap_heatmap'], 'fig_cap_heatmap')}
    </div>
    <div class="section">
      <h2>Utilización P90 de capacidad instalada</h2>
      <p>% de la capacidad usada en el P90 de escenarios × períodos. Rojo = binding.</p>
      {_div(figs['utilization'], 'fig_utilization')}
    </div>
  </div>

  <div class="section">
    <h2>Decisión operacional — % períodos en que cada satélite decide OPERAR (Z &gt; 0)</h2>
    <p>100% = opera en todos los períodos/escenarios. Bajo = satélite instalado pero frecuentemente "apagado" para ahorrar costo operacional.</p>
    {_div(figs['op_profile'], 'fig_op_profile')}
  </div>

</div>
</body></html>"""

    OUTPUT_PATH = folder / "summary.html"
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Summary -> {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    generate()

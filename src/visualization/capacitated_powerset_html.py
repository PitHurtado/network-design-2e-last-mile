"""Generate summary HTML for the capacitated powerset experiment.

Shows objective breakdown: installation cost + satellite routing + DC routing.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import plotly.graph_objects as go

RESULTS_DIR = Path("results/powerset_experiment_capacitated")
PLOTLY_CFG  = {"displayModeBar": True, "displaylogo": False}


def generate_summary_html(results_dir: Path = RESULTS_DIR) -> Path:
    jsons = sorted(results_dir.glob("capacitated_*_None.json"))
    if not jsons:
        raise FileNotFoundError(f"No JSONs in {results_dir}")

    # ── Load records ─────────────────────────────────────────────────────────
    raw = []
    for p in jsons:
        d = json.loads(p.read_text())
        label = d.get("configuration", p.stem)
        cap   = d.get("chosen_capacity", {})
        n_active = sum(1 for v in cap.values() if v.get("vehicles", 0) > 0)
        # Cost components (N=1 so no scaling needed)
        raw.append({
            "label":       label,
            "objective":   d.get("objective", 0.0),
            "c_install":   d.get("cost_installation", 0.0),
            "c_sat":       d.get("cost_served_from_facilities", 0.0),
            "c_dc":        d.get("cost_served_from_dc", 0.0),
            "n_active":    n_active,
            "cap":         cap,
        })

    baseline = next(
        (r["objective"] for r in raw if r["label"] == "DC_only"),
        max(r["objective"] for r in raw),
    )

    records = []
    for r in raw:
        d_abs = r["objective"] - baseline
        d_pct = d_abs / baseline * 100 if baseline else 0.0
        records.append({**r, "delta_abs": d_abs, "delta_pct": d_pct})
    records.sort(key=lambda r: r["objective"])

    best_per_size: dict[int, dict] = {}
    for r in records:
        k = r["n_active"]
        if k not in best_per_size or r["objective"] < best_per_size[k]["objective"]:
            best_per_size[k] = r

    # ── Scatter ───────────────────────────────────────────────────────────────
    palette = ["#95a5a6","#3498db","#2ecc71","#f39c12","#e74c3c",
               "#9b59b6","#1abc9c","#e67e22","#34495e","#16a085"]
    random.seed(42)
    scatter_traces = []
    for k in range(0, 10):
        grp = [r for r in records if r["n_active"] == k]
        if not grp:
            continue
        jitter = [k + random.uniform(-0.35, 0.35) for _ in grp]
        scatter_traces.append(go.Scatter(
            x=jitter, y=[r["delta_pct"] for r in grp],
            mode="markers", name=f"{k} sat.",
            marker=dict(color=palette[k % len(palette)], size=7, opacity=0.75),
            text=[
                f"<b>{r['label']}</b><br>"
                f"Obj: ${r['objective']:,.0f}<br>"
                f"Fijo: ${r['c_install']:,.0f} | "
                f"Sat: ${r['c_sat']:,.0f} | "
                f"DC: ${r['c_dc']:,.0f}<br>"
                f"Δ%: {r['delta_pct']:+.2f}%"
                for r in grp
            ],
            hovertemplate="%{text}<extra></extra>",
        ))
    frontier_x = sorted(best_per_size)
    scatter_traces.append(go.Scatter(
        x=frontier_x,
        y=[best_per_size[k]["delta_pct"] for k in frontier_x],
        mode="lines+markers", name="Frontera",
        line=dict(color="#2c3e50", width=2, dash="dash"),
        marker=dict(size=9, color="#2c3e50"),
        hovertemplate="<b>Mejor %{x} sat.</b><br>Δ%: %{y:.2f}%<extra></extra>",
    ))
    scatter_fig = go.Figure(scatter_traces)
    scatter_fig.add_hline(y=0, line_color="grey", line_dash="dot", opacity=0.5)
    scatter_fig.update_layout(
        xaxis=dict(title="Satélites activos", tickvals=list(range(10))),
        yaxis=dict(title="Δ% vs DC_only"),
        margin=dict(l=60, r=20, t=10, b=50), height=420,
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="#ccc", borderwidth=1),
    )

    # ── Box plot ──────────────────────────────────────────────────────────────
    box_traces = [
        go.Box(
            y=[r["delta_pct"] for r in records if r["n_active"] == k],
            name=f"{k} sat.", marker_color=palette[k % len(palette)], boxmean=True,
        )
        for k in range(0, 10)
        if any(r["n_active"] == k for r in records)
    ]
    box_fig = go.Figure(box_traces)
    box_fig.add_hline(y=0, line_color="grey", line_dash="dot", opacity=0.5)
    box_fig.update_layout(
        xaxis=dict(title="Satélites activos"),
        yaxis=dict(title="Distribución Δ% vs DC_only"),
        margin=dict(l=60, r=20, t=10, b=50), height=380,
        plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
    )

    scatter_html = scatter_fig.to_html(full_html=False, include_plotlyjs=False, config=PLOTLY_CFG)
    box_html     = box_fig.to_html(    full_html=False, include_plotlyjs=False, config=PLOTLY_CFG)

    # ── KPI table ─────────────────────────────────────────────────────────────
    best_global = records[0]
    kpi_rows = "".join(
        f"<tr>"
        f"<td>{k}</td><td class='lc'>{best_per_size[k]['label']}</td>"
        f"<td>${best_per_size[k]['objective']:,.0f}</td>"
        f"<td>${best_per_size[k]['c_install']:,.0f}</td>"
        f"<td>${best_per_size[k]['c_sat']:,.0f}</td>"
        f"<td>${best_per_size[k]['c_dc']:,.0f}</td>"
        f"<td>{best_per_size[k]['delta_pct']:+.2f}%</td>"
        f"</tr>"
        for k in sorted(best_per_size)
    )
    kpi = f"""
<div class="section">
  <h2>Resumen — mejor configuración por número de satélites activos</h2>
  <p>
    <b>Configs totales:</b> {len(records)} &nbsp;|&nbsp;
    <b>DC_only baseline:</b> ${baseline:,.0f} &nbsp;|&nbsp;
    <b>Mejor global:</b> {best_global['label']} (${best_global['objective']:,.0f}, {best_global['delta_pct']:+.2f}%)
  </p>
  <table class="kpi-table">
    <thead>
      <tr><th># Act.</th><th>Config</th><th>Objetivo</th>
      <th>Costo fijo</th><th>Transp. sat.</th><th>Transp. DC</th><th>Δ%</th></tr>
    </thead>
    <tbody>{kpi_rows}</tbody>
  </table>
</div>"""

    # ── Full table ────────────────────────────────────────────────────────────
    size_opts = '<option value="">Todos</option>' + "".join(
        f'<option value="{k}">{k} sat. activos</option>' for k in range(0, 10)
    )
    table_rows = ""
    for i, r in enumerate(records):
        cap = r.get("cap", {})
        active_sats = "; ".join(
            f"{s.replace('_',' ').title()}:{v['vehicles']}v"
            for s, v in sorted(cap.items())
            if v.get("vehicles", 0) > 0
        ) if cap else "—"
        cls = "green-row" if r["delta_pct"] < 0 else ("red-row" if r["delta_pct"] > 0 else "")
        table_rows += (
            f'<tr data-nsat="{r["n_active"]}" class="{cls}">'
            f'<td>{i+1}</td>'
            f'<td class="label-cell">{r["label"]}</td>'
            f'<td>{r["n_active"]}</td>'
            f'<td>${r["objective"]:,.0f}</td>'
            f'<td>${r["c_install"]:,.0f}</td>'
            f'<td>${r["c_sat"]:,.0f}</td>'
            f'<td>${r["c_dc"]:,.0f}</td>'
            f'<td>{r["delta_abs"]:+,.0f}</td>'
            f'<td>{r["delta_pct"]:+.2f}%</td>'
            f'<td style="font-size:0.78em">{active_sats}</td>'
            f'</tr>'
        )

    table_section = f"""
<div class="section">
  <h2>Tabla completa — {len(records)} configuraciones</h2>
  <p>Ordenar por columna · Filtrar:
    <select id="size-filter" onchange="filterTable()">{size_opts}</select>
  </p>
  <div style="max-height:500px;overflow-y:auto;">
  <table id="results-table" class="data-table">
    <thead><tr>
      <th onclick="sortTable(0)">Rank↕</th>
      <th onclick="sortTable(1)">Config↕</th>
      <th onclick="sortTable(2)"># Act.↕</th>
      <th onclick="sortTable(3)">Objetivo↕</th>
      <th onclick="sortTable(4)">Costo fijo↕</th>
      <th onclick="sortTable(5)">Transp. sat.↕</th>
      <th onclick="sortTable(6)">Transp. DC↕</th>
      <th onclick="sortTable(7)">Δ abs↕</th>
      <th onclick="sortTable(8)">Δ%↕</th>
      <th>Capacidades elegidas</th>
    </tr></thead>
    <tbody id="table-body">{table_rows}</tbody>
  </table>
  </div>
</div>"""

    # ── CSS / JS ──────────────────────────────────────────────────────────────
    css = """
body{font-family:sans-serif;margin:0;padding:0;background:#f4f6f8;}
h1{color:#fff;margin:0;font-size:1.3em;}
#top-bar{position:sticky;top:0;z-index:100;background:#2c3e50;padding:10px 24px;
         display:flex;align-items:center;gap:16px;flex-wrap:wrap;
         box-shadow:0 2px 6px rgba(0,0,0,.25);}
.section{background:white;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.1);
         margin:18px 24px;padding:16px;}
.section h2{color:#2c3e50;margin:0 0 6px;font-size:1.1em;}
.section p{color:#555;margin:0 0 10px;font-size:.88em;}
.kpi-table{border-collapse:collapse;font-size:.84em;}
.kpi-table th,.kpi-table td{border:1px solid #ddd;padding:5px 10px;}
.kpi-table th{background:#2c3e50;color:white;text-align:center;}
.lc{text-align:left!important;}
.data-table{border-collapse:collapse;font-size:.82em;width:100%;}
.data-table thead{position:sticky;top:0;background:#2c3e50;color:white;}
.data-table th{padding:6px 10px;cursor:pointer;user-select:none;white-space:nowrap;}
.data-table th:hover{background:#34495e;}
.data-table td{border-bottom:1px solid #eee;padding:4px 10px;white-space:nowrap;}
.label-cell{font-size:.78em;max-width:340px;overflow:hidden;text-overflow:ellipsis;}
.green-row td{background:#f0fdf4;}.red-row td{background:#fef2f2;}
.data-table tr:hover td{filter:brightness(.96);}
"""
    js = """
var sortDir={};
function sortTable(col){
  var tbody=document.getElementById('table-body');
  var rows=Array.from(tbody.querySelectorAll('tr'));
  var asc=!sortDir[col];sortDir={};sortDir[col]=asc;
  rows.sort(function(a,b){
    var av=a.cells[col].innerText.replace(/[$+,%\\s]/g,'');
    var bv=b.cells[col].innerText.replace(/[$+,%\\s]/g,'');
    var an=parseFloat(av),bn=parseFloat(bv);
    if(!isNaN(an)&&!isNaN(bn))return asc?an-bn:bn-an;
    return asc?av.localeCompare(bv):bv.localeCompare(av);
  });
  rows.forEach(function(r){tbody.appendChild(r);});
}
function filterTable(){
  var val=document.getElementById('size-filter').value;
  document.querySelectorAll('#table-body tr').forEach(function(r){
    r.style.display=(val===''||r.dataset.nsat===val)?'':'none';
  });
}
"""

    html = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<title>Powerset Capacitado — Resumen</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>{css}</style><script>{js}</script>
</head><body>
<div id="top-bar">
  <h1>Experimento Powerset — Modelo Capacitado (Propuesta A · costos desde Excel)</h1>
</div>
{kpi}
<div class="section">
  <h2>Variación % vs DC_only por satélites activos</h2>
  <p>Hover sobre cada punto para ver el desglose de costos. Línea punteada = frontera Pareto.</p>
  {scatter_html}
</div>
<div class="section">
  <h2>Distribución de Δ% por número de satélites activos</h2>
  {box_html}
</div>
{table_section}
</body></html>"""

    out = results_dir / "summary.html"
    out.write_text(html, encoding="utf-8")
    print(f"Summary → {out}")
    return out


if __name__ == "__main__":
    generate_summary_html()

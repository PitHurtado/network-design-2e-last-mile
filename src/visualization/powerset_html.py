"""Generate summary HTML for the powerset satellite experiment."""

from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go

from src.visualization.solution_map import generate_html

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"
PLOTLY_CFG = {"displayModeBar": True, "displaylogo": False}

RESULTS_DIR = Path("results/powerset_experiment")


# ── Individual HTMLs ──────────────────────────────────────────────────────────

def generate_individual_htmls(results_dir: Path = RESULTS_DIR) -> None:
    """Generate one HTML per powerset config JSON."""
    jsons = sorted(results_dir.glob("uncapacitated_*_None.json"))
    total = len(jsons)
    for idx, json_path in enumerate(jsons, start=1):
        label = _label_from_json(json_path)
        html_path = results_dir / f"{label}.html"
        if html_path.exists():
            print(f"[{idx}/{total}] Skip (exists): {html_path.name}")
            continue
        print(f"[{idx}/{total}] Generating: {html_path.name}")
        generate_html(json_path, html_path)


# ── Summary HTML ──────────────────────────────────────────────────────────────

def _label_from_json(json_path: Path) -> str:
    with open(json_path) as f:
        data = json.load(f)
    return data.get("configuration", json_path.stem)


def generate_summary_html(results_dir: Path = RESULTS_DIR) -> Path:
    """Generate summary.html with scatter, boxplot, sortable table + iframe."""
    import random
    jsons = sorted(results_dir.glob("uncapacitated_*_None.json"))
    if not jsons:
        raise FileNotFoundError(f"No result JSONs found in {results_dir}")

    # ── Load and enrich records ───────────────────────────────────────────────
    raw = []
    for json_path in jsons:
        with open(json_path) as f:
            data = json.load(f)
        label = data.get("configuration", json_path.stem)
        obj = data.get("objective", 0.0)
        raw.append((label, obj))

    baseline = next((obj for lb, obj in raw if lb == "DC_only"), max(obj for _, obj in raw))

    records = []
    for label, obj in raw:
        n_sat = 0 if label == "DC_only" else len(label.split("+"))
        d_abs = obj - baseline
        d_pct = (d_abs / baseline * 100) if baseline else 0.0
        records.append({
            "label": label, "objective": obj, "n_sat": n_sat,
            "delta_abs": d_abs, "delta_pct": d_pct,
        })

    records.sort(key=lambda r: r["objective"])

    best_global = records[0]
    best_per_size: dict[int, dict] = {}
    for r in records:
        k = r["n_sat"]
        if k not in best_per_size or r["objective"] < best_per_size[k]["objective"]:
            best_per_size[k] = r

    # ── Figure 1: Scatter # satellites vs Δ% ─────────────────────────────────
    sat_palette = [
        "#95a5a6", "#3498db", "#2ecc71", "#f39c12", "#e74c3c",
        "#9b59b6", "#1abc9c", "#e67e22", "#34495e", "#16a085",
    ]
    random.seed(42)
    scatter_traces = []
    for k in range(0, 10):
        grp = [r for r in records if r["n_sat"] == k]
        if not grp:
            continue
        jitter = [k + random.uniform(-0.35, 0.35) for _ in grp]
        scatter_traces.append(go.Scatter(
            x=jitter,
            y=[r["delta_pct"] for r in grp],
            mode="markers",
            name=f"{k} sat.",
            marker=dict(color=sat_palette[k % len(sat_palette)], size=7, opacity=0.75),
            text=[
                f"<b>{r['label']}</b><br>Objetivo: {r['objective']:,.2f}<br>"
                f"Δabs: {r['delta_abs']:+,.2f}<br>Δ%: {r['delta_pct']:+.2f}%"
                for r in grp
            ],
            hovertemplate="%{text}<extra></extra>",
        ))
    # Frontier line (best per size)
    frontier_x = sorted(best_per_size.keys())
    frontier_y = [best_per_size[k]["delta_pct"] for k in frontier_x]
    scatter_traces.append(go.Scatter(
        x=frontier_x, y=frontier_y,
        mode="lines+markers",
        name="Frontera (mejor por tamaño)",
        line=dict(color="#2c3e50", width=2, dash="dash"),
        marker=dict(size=9, color="#2c3e50"),
        text=[
            f"<b>Mejor con {k} sat.</b><br>{best_per_size[k]['label']}<br>"
            f"Δ%: {best_per_size[k]['delta_pct']:+.2f}%"
            for k in frontier_x
        ],
        hovertemplate="%{text}<extra></extra>",
    ))
    scatter_fig = go.Figure(scatter_traces)
    scatter_fig.add_hline(y=0, line_color="grey", line_dash="dot", opacity=0.5)
    scatter_fig.update_layout(
        xaxis=dict(title="Número de satélites activos", tickvals=list(range(10)),
                   ticktext=[str(k) for k in range(10)]),
        yaxis=dict(title="Variación % vs DC_only"),
        margin=dict(l=60, r=20, t=10, b=50), height=420,
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="#ccc", borderwidth=1),
    )

    # ── Figure 2: Box plot distribution per subset size ───────────────────────
    box_traces = []
    for k in range(0, 10):
        grp = [r for r in records if r["n_sat"] == k]
        if not grp:
            continue
        box_traces.append(go.Box(
            y=[r["delta_pct"] for r in grp],
            name=f"{k} sat.",
            marker_color=sat_palette[k % len(sat_palette)],
            boxmean=True,
        ))
    box_fig = go.Figure(box_traces)
    box_fig.add_hline(y=0, line_color="grey", line_dash="dot", opacity=0.5)
    box_fig.update_layout(
        xaxis=dict(title="Número de satélites activos"),
        yaxis=dict(title="Distribución Δ% vs DC_only"),
        margin=dict(l=60, r=20, t=10, b=50), height=400,
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False,
    )

    scatter_html = scatter_fig.to_html(full_html=False, include_plotlyjs=False, config=PLOTLY_CFG)
    box_html = box_fig.to_html(full_html=False, include_plotlyjs=False, config=PLOTLY_CFG)

    # ── KPI cards ─────────────────────────────────────────────────────────────
    best_per_size_rows = "".join(
        f"<tr><td>{k}</td><td>{best_per_size[k]['label']}</td>"
        f"<td>{best_per_size[k]['objective']:,.2f}</td>"
        f"<td>{best_per_size[k]['delta_abs']:+,.2f}</td>"
        f"<td>{best_per_size[k]['delta_pct']:+.2f}%</td></tr>"
        for k in sorted(best_per_size)
    )
    kpi_section = f"""
<div class="section">
  <h2>Resumen del experimento</h2>
  <p>
    <b>Configuraciones totales:</b> {len(records)} &nbsp;|&nbsp;
    <b>Baseline DC_only:</b> {baseline:,.2f} &nbsp;|&nbsp;
    <b>Mejor global:</b> {best_global['label']}
    ({best_global['objective']:,.2f}, {best_global['delta_pct']:+.2f}%)
  </p>
  <p><b>Mejor configuración por número de satélites:</b></p>
  <table class="kpi-table">
    <thead><tr><th># Sat.</th><th>Config</th><th>Objetivo</th><th>Δ Absoluto</th><th>Δ%</th></tr></thead>
    <tbody>{best_per_size_rows}</tbody>
  </table>
</div>"""

    # ── Sortable table ─────────────────────────────────────────────────────────
    size_filter_opts = '<option value="">Todos los tamaños</option>' + "".join(
        f'<option value="{k}">{k} satélites</option>' for k in range(0, 10)
    )
    table_rows = "".join(
        f'<tr data-nsat="{r["n_sat"]}" class="{"green-row" if r["delta_pct"] < 0 else ("red-row" if r["delta_pct"] > 0 else "")}">'
        f'<td>{i+1}</td><td class="label-cell">{r["label"]}</td><td>{r["n_sat"]}</td>'
        f'<td>{r["objective"]:,.2f}</td><td>{r["delta_abs"]:+,.2f}</td>'
        f'<td>{r["delta_pct"]:+.2f}%</td></tr>'
        for i, r in enumerate(records)
    )
    table_section = f"""
<div class="section">
  <h2>Tabla completa de resultados</h2>
  <p>Click en los encabezados para ordenar. Filtra por tamaño de subset:
    <select id="size-filter" onchange="filterTable()">
      {size_filter_opts}
    </select>
  </p>
  <div style="max-height:420px;overflow-y:auto;">
  <table id="results-table" class="data-table">
    <thead>
      <tr>
        <th onclick="sortTable(0)">Rank ↕</th>
        <th onclick="sortTable(1)">Config ↕</th>
        <th onclick="sortTable(2)"># Sat. ↕</th>
        <th onclick="sortTable(3)">Objetivo ↕</th>
        <th onclick="sortTable(4)">Δ Absoluto ↕</th>
        <th onclick="sortTable(5)">Δ% ↕</th>
      </tr>
    </thead>
    <tbody id="table-body">{table_rows}</tbody>
  </table>
  </div>
</div>"""

    # ── Dropdown for iframe ───────────────────────────────────────────────────
    dropdown_options = "\n".join(
        f'<option value="{r["label"]}">{r["label"]}  (Δ%: {r["delta_pct"]:+.2f}%)</option>'
        for r in records
    )

    # ── CSS ───────────────────────────────────────────────────────────────────
    css = """
body { font-family: sans-serif; margin: 0; padding: 0; background: #f4f6f8; }
h1 { color: #2c3e50; margin: 0; font-size: 1.3em; }
#top-bar { position: sticky; top: 0; z-index: 100; background: #2c3e50;
           padding: 10px 24px; display: flex; align-items: center; gap: 20px; flex-wrap: wrap; }
#top-bar label { color: white; font-size: 0.9em; font-weight: bold; }
#top-bar select { font-size: 0.95em; padding: 4px 8px; border-radius: 4px; border: none; min-width: 340px; }
.section { background: white; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
           margin: 20px 24px; padding: 16px; }
.section h2 { color: #2c3e50; margin: 0 0 6px; font-size: 1.1em; }
.section p  { color: #555; margin: 0 0 10px; font-size: 0.88em; }
iframe { width: 100%; height: 780px; border: none; border-radius: 4px; }
.kpi-table { border-collapse: collapse; font-size: 0.85em; width: auto; }
.kpi-table th, .kpi-table td { border: 1px solid #ddd; padding: 5px 10px; text-align: right; }
.kpi-table th { background: #2c3e50; color: white; }
.kpi-table td:nth-child(2) { text-align: left; }
.data-table { border-collapse: collapse; font-size: 0.82em; width: 100%; }
.data-table thead { position: sticky; top: 0; background: #2c3e50; color: white; }
.data-table th { padding: 6px 10px; cursor: pointer; user-select: none; white-space: nowrap; }
.data-table th:hover { background: #34495e; }
.data-table td { border-bottom: 1px solid #eee; padding: 4px 10px; white-space: nowrap; }
.label-cell { font-size: 0.78em; max-width: 420px; overflow: hidden; text-overflow: ellipsis; }
.green-row td { background: #f0fdf4; }
.red-row td { background: #fef2f2; }
.data-table tr:hover td { filter: brightness(0.95); }
"""

    # ── JS ────────────────────────────────────────────────────────────────────
    js = """
function updateIframe() {
  var label = document.getElementById('sel-config').value;
  document.getElementById('config-iframe').src = label + '.html';
}
document.addEventListener('DOMContentLoaded', function() { updateIframe(); });

var sortDir = {};
function sortTable(col) {
  var tbody = document.getElementById('table-body');
  var rows = Array.from(tbody.querySelectorAll('tr'));
  var asc = !sortDir[col];
  sortDir = {};
  sortDir[col] = asc;
  rows.sort(function(a, b) {
    var av = a.cells[col].innerText.replace(/[+,%,]/g, '');
    var bv = b.cells[col].innerText.replace(/[+,%,]/g, '');
    var an = parseFloat(av), bn = parseFloat(bv);
    if (!isNaN(an) && !isNaN(bn)) return asc ? an - bn : bn - an;
    return asc ? av.localeCompare(bv) : bv.localeCompare(av);
  });
  rows.forEach(function(r) { tbody.appendChild(r); });
}

function filterTable() {
  var val = document.getElementById('size-filter').value;
  var rows = document.querySelectorAll('#table-body tr');
  rows.forEach(function(r) {
    r.style.display = (val === '' || r.dataset.nsat === val) ? '' : 'none';
  });
}
"""

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Experimento Powerset — Satélites</title>
  <script src="{PLOTLY_CDN}"></script>
  <style>{css}</style>
  <script>{js}</script>
</head>
<body>
<div id="top-bar">
  <h1>Experimento Powerset de Satélites</h1>
  <label>Ver config:
    <select id="sel-config" onchange="updateIframe()">
{dropdown_options}
    </select>
  </label>
</div>
{kpi_section}
<div class="section">
  <h2>Variación % vs DC_only por número de satélites activos</h2>
  <p>Cada punto = una configuración. Línea punteada = frontera (mejor config por tamaño). Línea gris = baseline DC_only.</p>
  {scatter_html}
</div>
<div class="section">
  <h2>Distribución de Δ% por número de satélites activos</h2>
  <p>Boxplot de la variación porcentual dentro de cada tamaño de subset. La cruz (×) indica la media.</p>
  {box_html}
</div>
{table_section}
<div class="section">
  <h2>Solución de la configuración seleccionada</h2>
  <p>Usa los dropdowns de período y layer dentro del iframe para navegar la solución.</p>
  <iframe id="config-iframe" src=""></iframe>
</div>
</body>
</html>
"""

    output_path = results_dir / "summary.html"
    output_path.write_text(html, encoding="utf-8")
    print(f"Summary HTML guardado en: {output_path}")
    return output_path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate powerset experiment HTMLs.")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR), help="Directory with powerset JSONs.")
    parser.add_argument("--skip-individual", action="store_true", help="Skip generating individual config HTMLs.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if not args.skip_individual:
        print("=== Generating individual config HTMLs ===")
        generate_individual_htmls(results_dir)

    print("=== Generating summary HTML ===")
    generate_summary_html(results_dir)

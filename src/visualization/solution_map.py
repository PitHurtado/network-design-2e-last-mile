"""Generate interactive HTML map from uncapacitated SAA solution JSON."""

from __future__ import annotations

import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path

import math

import plotly.graph_objects as go

PLOTLY_CFG = {"displayModeBar": True, "displaylogo": False}
PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"
DC_LABEL = "DC"

# ── Data loading ──────────────────────────────────────────────────────────────

def load_solution(json_path: str | Path) -> dict:
    with open(json_path) as f:
        return json.load(f)


# ── Assignment logic ──────────────────────────────────────────────────────────

def _parse_key(key_str: str) -> tuple:
    """Convert string repr of tuple back to tuple."""
    return ast.literal_eval(key_str)


def compute_assignments(data: dict) -> dict[tuple[str, int], dict]:
    """
    For each (pixel_k, period_t), compute the mean assignment across scenarios.

    Returns:
        assignments[(k, t)] = {
            "serving_from": facility_id or "DC",
            "mean_cost": float,
            "mean_fleet_size": float,
            "mean_demand": float,
        }
    """
    scenarios = data["scenarios"]
    n_scenarios = len(scenarios)
    facilities = list(data["facility_info"].keys())
    pixels = list(data["pixel_info"].keys())
    periods = data["periods"]

    x_vals: dict[tuple, float] = {_parse_key(k): v for k, v in data["X"].items()}
    w_vals: dict[tuple, float] = {_parse_key(k): v for k, v in data["W"].items()}

    cost_fac: dict[tuple, float] = {_parse_key(k): v for k, v in data["cost_serving"]["facility"].items()}
    cost_dc: dict[tuple, float] = {_parse_key(k): v for k, v in data["cost_serving"]["dc"].items()}
    fleet_fac: dict[tuple, float] = {_parse_key(k): v for k, v in data["fleet_size_serving"]["facility"].items()}
    fleet_dc: dict[tuple, float] = {_parse_key(k): v for k, v in data["fleet_size_serving"]["dc"].items()}

    assignments = {}
    for k in pixels:
        demand_by_period = data["pixel_info"][k]["demand_by_period"]
        for t in range(periods):
            # Mean X across scenarios for each facility
            mean_x = {}
            for i in facilities:
                vals = [x_vals.get((i, k, t, n), 0.0) for n in scenarios]
                mean_x[i] = sum(vals) / n_scenarios

            # Mean W across scenarios
            mean_w = sum(w_vals.get((k, t, n), 0.0) for n in scenarios) / n_scenarios

            best_fac = max(mean_x, key=mean_x.get) if mean_x else None
            best_fac_val = mean_x[best_fac] if best_fac else 0.0

            if best_fac and best_fac_val >= mean_w:
                serving_from = best_fac
                mean_cost = sum(
                    cost_fac.get((best_fac, k, "small", t, n), 0.0) for n in scenarios
                ) / n_scenarios
                mean_fleet = sum(
                    fleet_fac.get((best_fac, k, "small", t, n), 0.0) for n in scenarios
                ) / n_scenarios
            else:
                serving_from = DC_LABEL
                mean_cost = sum(cost_dc.get((k, "large", t, n), 0.0) for n in scenarios) / n_scenarios
                mean_fleet = sum(fleet_dc.get((k, "large", t, n), 0.0) for n in scenarios) / n_scenarios

            mean_demand = demand_by_period[t] if t < len(demand_by_period) else 0.0

            assignments[(k, t)] = {
                "serving_from": serving_from,
                "mean_cost": round(mean_cost, 5),
                "mean_fleet_size": round(mean_fleet, 5),
                "mean_demand": round(mean_demand, 2),
            }

    return assignments


# ── Pixel totals ──────────────────────────────────────────────────────────────

def compute_pixel_totals(data: dict, assignments: dict) -> dict[str, dict]:
    """For each pixel k, compute aggregate metrics across all periods."""
    pixels = list(data["pixel_info"].keys())
    periods = data["periods"]
    totals = {}
    for k in pixels:
        total_demand = sum(assignments[(k, t)]["mean_demand"] for t in range(periods))
        total_cost = sum(assignments[(k, t)]["mean_cost"] for t in range(periods))
        totals[k] = {
            "total_demand": round(total_demand, 2),
            "total_cost": round(total_cost, 2),
        }
    return totals


# ── Figure generation ─────────────────────────────────────────────────────────

def _pixel_rect(lon: float, lat: float, area_km2: float = 1.0) -> tuple[list, list]:
    """Return (x_coords, y_coords) of a closed rectangle polygon for a pixel, plus None separator."""
    half = math.sqrt(area_km2) / 2
    dlat = half / 111.32
    dlon = half / (111.32 * math.cos(math.radians(lat)))
    x = [lon - dlon, lon + dlon, lon + dlon, lon - dlon, lon - dlon, None]
    y = [lat - dlat, lat - dlat, lat + dlat, lat + dlat, lat - dlat, None]
    return x, y


def _haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Approximate great-circle distance in km between two lon/lat points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def _categorical_colors(labels: list[str]) -> dict[str, str]:
    palette = [
        "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
        "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
        "#dcbeff", "#9A6324", "#fffac8", "#800000", "#aaffc3",
        "#808000", "#ffd8b1", "#000075", "#a9a9a9",
    ]
    return {label: palette[i % len(palette)] for i, label in enumerate(sorted(labels))}


def build_figure(
    data: dict,
    assignments: dict,
    pixel_totals: dict,
    period: int,
    layer: str,
) -> go.Figure:
    pixel_info = data["pixel_info"]
    facility_info = data["facility_info"]

    layer_pixels = {k: v for k, v in pixel_info.items() if v["layer"].upper() == layer.upper()}

    # Determine all unique serving labels for coloring
    all_labels = sorted({assignments[(k, period)]["serving_from"] for k in layer_pixels})
    colors = _categorical_colors(all_labels)

    # Group pixels by serving entity for traces (one trace per entity → legend)
    groups: dict[str, list] = defaultdict(list)
    for k, pinfo in layer_pixels.items():
        asgn = assignments[(k, period)]
        groups[asgn["serving_from"]].append((k, pinfo, asgn))

    # Precompute costs for all facilities per pixel for this period/scenario
    scenario_id = data["scenarios"][0]
    cost_fac_lookup = data["cost_serving"]["facility"]

    def _costs_all_facs(k: str) -> dict[str, float]:
        return {
            fac_id: cost_fac_lookup.get(str((fac_id, k, "small", period, scenario_id)), None)
            for fac_id in facility_info
        }

    # Grid spacing for 1km² grid
    all_lats = [v["lat"] for v in layer_pixels.values()]
    center_lat = sum(all_lats) / len(all_lats) if all_lats else 0.0
    dlat_1km = 1.0 / 111.32
    dlon_1km = 1.0 / (111.32 * math.cos(math.radians(center_lat)))

    traces = []
    hover_x, hover_y, hover_text = [], [], []  # invisible center-point hover layer

    for label, items in sorted(groups.items()):
        x_all, y_all = [], []
        for k, pinfo, asgn in items:
            assigned = asgn["serving_from"]

            # Aerial distances to every facility (may be empty for DC-only configs)
            aerial = {
                fid: _haversine(pinfo["lon"], pinfo["lat"], finfo["lon"], finfo["lat"])
                for fid, finfo in facility_info.items()
            }
            nearest = min(aerial, key=aerial.get) if aerial else None
            d_assigned = aerial.get(assigned, float("nan")) if assigned != DC_LABEL else float("nan")
            d_nearest = aerial[nearest] if nearest else float("nan")

            # Costs for all facilities this period
            costs = _costs_all_facs(k)
            valid = {f: c for f, c in costs.items() if c is not None}
            cheapest = min(valid, key=valid.get) if valid else None
            cost_assigned = asgn["mean_cost"]
            cost_nearest = valid.get(nearest) if nearest else None
            cost_cheapest = valid.get(cheapest) if cheapest else None

            mismatch = nearest is not None and assigned != nearest and assigned != DC_LABEL
            warning = "<br><b>⚠ Asignado difiere del más cercano (aéreo)</b>" if mismatch else ""

            nearest_line = (
                f"<br><b>Satelite más cercano (aéreo):</b> {nearest} ({d_nearest:.2f} km)"
                if nearest else ""
            )

            hover = (
                f"<b>Pixel:</b> {layer}-{pinfo['pixel_id']}<br>"
                f"<b>Satelite asignado:</b> {assigned}"
                + (f" (dist aérea: {d_assigned:.2f} km)" if not math.isnan(d_assigned) else "")
                + nearest_line + "<br>"
                + f"<b>Costo período:</b> {cost_assigned:.0f}"
                + (f" | <b>Costo más cercano:</b> {cost_nearest:.0f}" if cost_nearest is not None else "")
                + (f" | <b>Costo más barato:</b> {cheapest} ({cost_cheapest:.0f})" if cheapest and cheapest != assigned else "")
                + f"<br><b>Demanda período:</b> {asgn['mean_demand']:.2f} | <b>Fleet size:</b> {asgn['mean_fleet_size']:.2f}"
                + f"<br><b>Demanda total (12 per.):</b> {pixel_totals[k]['total_demand']:.2f} | <b>Costo total (12 per.):</b> {pixel_totals[k]['total_cost']:.0f}"
                + warning
            )
            px, py = _pixel_rect(pinfo["lon"], pinfo["lat"])
            x_all.extend(px)
            y_all.extend(py)
            # Collect center-point for the invisible hover layer
            hover_x.append(pinfo["lon"])
            hover_y.append(pinfo["lat"])
            hover_text.append(hover)

        # Filled polygon trace — no hover (hoverinfo="skip") to avoid conflicts
        traces.append(
            go.Scatter(
                x=x_all,
                y=y_all,
                mode="lines",
                fill="toself",
                fillcolor=colors[label],
                line=dict(color=colors[label], width=0.5),
                opacity=0.85,
                name=label,
                hoverinfo="skip",
            )
        )

    # Invisible center-point markers — carry the full hover for every pixel
    traces.append(
        go.Scatter(
            x=hover_x,
            y=hover_y,
            mode="markers",
            marker=dict(size=12, opacity=0, color="rgba(0,0,0,0)"),
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )

    # Facility markers — star colored as their pixel group, no legend entry
    for fac_id, finfo in facility_info.items():
        star_color = colors.get(fac_id, "black")
        traces.append(
            go.Scatter(
                x=[finfo["lon"]],
                y=[finfo["lat"]],
                mode="markers",
                marker=dict(size=16, color=star_color, symbol="star"),
                hovertemplate=f"<b>Facility:</b> {fac_id}<extra></extra>",
                showlegend=False,
            )
        )

    fig = go.Figure(traces)
    fig.update_layout(
        xaxis=dict(
            title="Longitud", showgrid=True, zeroline=False,
            dtick=dlon_1km, gridcolor="#d0d0d0", gridwidth=1,
        ),
        yaxis=dict(
            title="Latitud", showgrid=True, zeroline=False,
            dtick=dlat_1km, gridcolor="#d0d0d0", gridwidth=1,
            scaleanchor="x", scaleratio=1,
        ),
        margin=dict(l=40, r=20, t=20, b=40),
        height=600,
        plot_bgcolor="white",
        legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="#ccc", borderwidth=1),
    )
    return fig


# ── HTML assembly ─────────────────────────────────────────────────────────────

def generate_html(json_path: str | Path, output_path: str | Path | None = None) -> Path:
    json_path = Path(json_path)
    if output_path is None:
        output_path = json_path.with_suffix(".html")
    output_path = Path(output_path)

    data = load_solution(json_path)
    assignments = compute_assignments(data)
    pixel_totals = compute_pixel_totals(data, assignments)

    periods = data["periods"]
    layers = sorted({v["layer"].upper() for v in data["pixel_info"].values()})
    combos = [(t, layer) for t in range(periods) for layer in layers]

    # Pre-compute one figure per combo
    combo_html: dict[str, str] = {}
    for t, layer in combos:
        key = f"t{t}_layer{layer}"
        fig = build_figure(data, assignments, pixel_totals, t, layer)
        combo_html[key] = fig.to_html(full_html=False, include_plotlyjs=False, config=PLOTLY_CFG)

    # Period options (1-indexed for display)
    period_options = "\n".join(
        f'<option value="t{t}_layer__LAYER__">{t + 1}</option>' for t in range(periods)
    )
    layer_options = "\n".join(f'<option value="t__PERIOD___{layer}">{layer}</option>' for layer in layers)

    # Build period <select>
    period_select = "<select id='sel-period' onchange='updateView()'>\n"
    for t in range(periods):
        period_select += f'  <option value="{t}">Período {t + 1}</option>\n'
    period_select += "</select>"

    # Build layer <select>
    layer_select = "<select id='sel-layer' onchange='updateView()'>\n"
    for layer in layers:
        layer_select += f'  <option value="{layer}">Layer {layer}</option>\n'
    layer_select += "</select>"

    js_update = """
function updateView() {
  var t = document.getElementById('sel-period').value;
  var layer = document.getElementById('sel-layer').value;
  var key = 't' + t + '_layer' + layer;
  document.querySelectorAll('.combo-view').forEach(function(el) { el.style.display = 'none'; });
  var active = document.querySelector('.combo-view[data-key="' + key + '"]');
  if (active) {
    active.style.display = '';
    active.querySelectorAll('.js-plotly-plot').forEach(function(el) { Plotly.Plots.resize(el); });
  }
}
document.addEventListener('DOMContentLoaded', updateView);
"""

    css = """
body { font-family: sans-serif; margin: 0; padding: 0; background: #f4f6f8; }
h1 { color: #2c3e50; margin: 0; font-size: 1.3em; }
#top-bar { position: sticky; top: 0; z-index: 100; background: #2c3e50;
           padding: 10px 24px; display: flex; align-items: center; gap: 20px; flex-wrap: wrap; }
#top-bar label { color: white; font-size: 0.9em; font-weight: bold; }
#top-bar select { font-size: 0.95em; padding: 4px 8px; border-radius: 4px; border: none; }
.section { background: white; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
           margin: 20px 24px; padding: 16px; }
.section h2 { color: #2c3e50; margin: 0 0 4px; font-size: 1.1em; }
.section p  { color: #7f8c8d; margin: 0 0 12px; font-size: 0.88em; }
"""

    objective = data.get("objective", 0)
    cost_fac = data.get("cost_served_from_facilities", 0)
    cost_dc = data.get("cost_served_from_dc", 0)

    summary_html = (
        f"<div class='section'><h2>Resumen de solución</h2>"
        f"<p><b>Objetivo total:</b> {objective:,.2f} &nbsp;|&nbsp; "
        f"<b>Costo desde facilities:</b> {cost_fac:,.2f} &nbsp;|&nbsp; "
        f"<b>Costo desde DC:</b> {cost_dc:,.2f}</p></div>"
    )

    # Assemble HTML
    parts = [
        "<!DOCTYPE html><html lang='es'><head><meta charset='utf-8'>",
        "<title>Solución No Capacitada — Mapa</title>",
        f'<script src="{PLOTLY_CDN}"></script>',
        f"<style>{css}</style>",
        f"<script>{js_update}</script>",
        "</head><body>",
        "<div id='top-bar'>",
        "  <h1>Distribución óptima no capacitada</h1>",
        f"  <label>Período: {period_select}</label>",
        f"  <label>Layer: {layer_select}</label>",
        "</div>",
        summary_html,
        "<div class='section'><h2>Mapa de asignación de píxeles</h2>"
        "<p>Cada punto representa un píxel. El color indica el satélite que lo sirve (★ = ubicación de facility). "
        "Usa los dropdowns para filtrar por período y layer.</p>",
    ]

    first_key = f"t0_layer{layers[0]}"
    for key, fig_html in combo_html.items():
        display = "" if key == first_key else "display:none;"
        parts.append(f'<div class="combo-view" data-key="{key}" style="{display}">{fig_html}</div>')

    parts.append("</div>")  # close section
    parts.append("</body></html>")

    html = "\n".join(parts)
    output_path.write_text(html, encoding="utf-8")
    print(f"HTML guardado en: {output_path}")
    return output_path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate interactive solution map HTML.")
    parser.add_argument("--json", required=True, help="Path to uncapacitated solution JSON file.")
    parser.add_argument("--output", default=None, help="Output HTML path (default: same dir as JSON).")
    args = parser.parse_args()

    generate_html(args.json, args.output)

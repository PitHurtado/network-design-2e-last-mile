"""Report comparing an independent baseline with two spatially joint demand approaches.

The historical pixel-month panel is the reference: does the joint generator reproduce
the local co-movement that independent sampling necessarily removes?
"""

import base64
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.core.constants import GRID_DLAT, GRID_DLON, GRID_N_COLS, GRID_N_ROWS, N_PERIODS
from src.core.contract import DEPENDENCE_METHODS as METHODS
from src.core.grid import GRID_X, GRID_Y, cell_center, layer_of
from src.visualization.components.labels import METHOD_LABELS
from src.visualization.html import PLOTLY_CDN, fig_html


@dataclass
class ComparisonReportData:
    """The paired comparison of one regime, computed by `src.scenarios.reports`."""

    regime: str
    pixels: list[str]
    generated: dict[str, pd.DataFrame]
    metrics: pd.DataFrame
    neighbors: pd.DataFrame
    distance: pd.DataFrame
    footprints: dict[str, set[int]]
    grid_payload: dict
    historical_cv: float
    n_scenarios: int


CSS = """
body { font-family: sans-serif; margin: 0; padding: 0; background: #f4f6f8; }
#top-bar { background: #243447; padding: 14px 24px; color: white; }
#top-bar h1 { margin: 0; font-size: 1.35em; }
#top-bar p { color: #cbd5e1; margin: 5px 0 0; font-size: .88em; }
.global-period-bar { position: sticky; top: 0; z-index: 1000; display: flex; align-items: center;
                     gap: 10px; padding: 10px 24px; background: #ffffff;
                     border-bottom: 1px solid #dbe3ec; box-shadow: 0 2px 8px rgba(15,23,42,.10); }
.global-period-bar label { color: #243447; font-size: .84em; font-weight: 700; }
.global-period-bar select { min-width: 150px; padding: 7px 30px 7px 10px; border: 1px solid #94a3b8;
                            border-radius: 5px; background: #f8fafc; color: #243447; font-size: .84em; }
.global-period-bar select.layer-select { min-width: 110px; }
.global-period-bar .scope { color: #64748b; font-size: .78em; }
.section { background: white; border-radius: 8px; margin: 18px 24px; padding: 18px;
           box-shadow: 0 2px 8px rgba(0,0,0,.08); }
.section h2 { color: #243447; margin: 0 0 5px; font-size: 1.1em; }
.section p { color: #586474; font-size: .88em; margin: 0 0 12px; }
.subsection { border-top: 1px solid #e5e7eb; margin-top: 18px; padding-top: 12px; }
.subsection h3 { color: #334155; font-size: .98em; margin: 0 0 5px; }
.scale-note { display: inline-block; margin: 4px 0 12px; padding: 6px 9px; border-radius: 5px;
              background: #f8fafc; color: #475569; font-size: .82em; }
.grid-inspector { margin-top: 10px; padding: 12px 14px; border: 1px solid #dbe3ec;
                  border-radius: 7px; background: #fbfdff; }
.grid-inspector h4 { margin: 0 0 5px; color: #243447; font-size: .94em; }
.grid-inspector p { margin: 0 0 9px; }
.grid-inspector .hint { color: #64748b; font-size: .82em; }
.inspector-metrics { display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 8px; margin: 8px 0 12px; }
.inspector-metric { padding: 7px 9px; background: #ffffff; border-left: 3px solid #3b82f6; }
.inspector-metric span { display: block; color: #64748b; font-size: .72em; }
.inspector-metric strong { color: #243447; font-size: .95em; }
.grid-inspector table { font-size: .78em; }
.grid-inspector th { position: sticky; top: 0; z-index: 1; }
.grid-inspector .focus-row td { background: #eef6ff; font-weight: 650; }
.method-intro { max-width: 980px; line-height: 1.55; }
.method-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin: 14px 0 18px; }
.method-card { border: 1px solid #dbe3ec; border-top: 4px solid #64748b; padding: 13px 14px; background: #fbfdff; }
.method-card.baseline { border-top-color: #64748b; }
.method-card.parametric { border-top-color: #2563eb; }
.method-card.empirical { border-top-color: #d97706; }
.method-card h3 { color: #243447; font-size: .95em; margin: 0 0 7px; }
.method-card p { margin: 0; line-height: 1.5; }
.metric-guide { border-top: 1px solid #e5e7eb; padding-top: 14px; }
.metric-guide h3 { color: #334155; font-size: .98em; margin: 0 0 7px; }
.metric-guide p { max-width: 980px; line-height: 1.55; }
.metric-formula { display: inline-block; margin: 3px 0 9px; padding: 8px 10px; background: #f1f5f9; color: #243447; font-family: Georgia, serif; }
@media (max-width: 900px) { .method-grid { grid-template-columns: 1fr; } }
details { border-top: 1px solid #e5e7eb; padding: 14px 0; }
details:first-child { border-top: 0; }
summary { cursor: pointer; color: #243447; font-weight: 650; padding: 3px 0; }
details p { max-width: 900px; line-height: 1.55; margin: 9px 0 0; }
table { border-collapse: collapse; width: 100%; font-size: .84em; }
th, td { border-bottom: 1px solid #e5e7eb; padding: 6px 8px; text-align: right; }
th { background: #f8fafc; color: #243447; }
th:first-child, td:first-child { text-align: left; }
code { background: #eef2f7; padding: 1px 4px; border-radius: 3px; }
"""


def _robust_limits(values: list[np.ndarray] | np.ndarray, lower: float = 0.01, upper: float = 0.99) -> tuple[float, float]:
    """Use common percentile limits so a few extreme pixels do not flatten the map."""
    flattened = np.concatenate(values) if isinstance(values, list) else np.asarray(values)
    flattened = flattened[np.isfinite(flattened)]
    low, high = np.quantile(flattened, [lower, upper])
    if high <= low:
        high = low + max(abs(low) * 0.01, 1e-6)
    return float(low), float(high)


def _grid_heatmap_trace(
    values: dict[str, float],
    footprints: dict[str, set[int]],
    layer: str,
    period: int,
    method: str,
    metric: str,
    coloraxis: str,
    hover_label: str,
    visible: bool,
    custom_values: dict[str, dict] | None = None,
) -> go.Heatmap:
    """One compact cell-level heatmap; one trace replaces hundreds of polygons."""
    z = [[None for _ in range(GRID_N_COLS)] for _ in range(GRID_N_ROWS)]
    customdata = [[None for _ in range(GRID_N_COLS)] for _ in range(GRID_N_ROWS)]
    for pixel, cells in footprints.items():
        if layer_of(pixel) != layer or pixel not in values:
            continue
        value = values[pixel]
        metrics = (custom_values or {}).get(pixel, {})
        for cell in cells:
            row, col = divmod(int(cell), GRID_N_COLS)
            z[row][col] = value
            customdata[row][col] = [
                pixel,
                value,
                period,
                method,
                metric,
                metrics.get("stop"),
                metrics.get("drop"),
                metrics.get("demand"),
                metrics.get("cv"),
                layer,
            ]
    return go.Heatmap(
        x=GRID_X,
        y=GRID_Y,
        z=z,
        customdata=customdata,
        coloraxis=coloraxis,
        visible=visible,
        name=f"{method}-{metric}-{layer}",
        xgap=1,
        ygap=1,
        zsmooth=False,
        hoverongaps=False,
        hovertemplate=(
            "<b>Píxel %{customdata[0]}</b><br>"
            "Layer: %{customdata[9]}<br>"
            "Período: %{customdata[2]}<br>"
            "Promedio stop: %{customdata[5]:.2f}<br>"
            "Promedio drop: %{customdata[6]:.2f}<br>"
            "Promedio demanda: %{customdata[7]:.2f}<br>"
            "CV demanda: %{customdata[8]:.3f}<br>" + hover_label + ": %{customdata[1]:.3f}<extra></extra>"
        ),
        showscale=False,
    )


def _all_methods_grid_figure(
    generated: dict[str, pd.DataFrame],
    pixels: list[str],
    footprints: dict[str, set[int]],
    inspector_payload: dict | None = None,
) -> tuple[go.Figure, dict[str, int], dict[str, int]]:
    """3×3 dashboard: each row is a method; columns are demand, CV and neighbors."""
    layers = sorted({layer_of(pixel) for pixel in pixels})
    titles = [
        title
        for method in METHODS
        for title in (
            f"{METHOD_LABELS[method]} · demanda",
            f"{METHOD_LABELS[method]} · CV",
            f"{METHOD_LABELS[method]} · vecinos por CV",
        )
    ]
    fig = make_subplots(rows=3, cols=3, subplot_titles=titles, horizontal_spacing=0.045, vertical_spacing=0.085)
    summaries, all_mean, all_cv = {}, [], []
    for method in METHODS:
        frame = generated[method]
        grouped = frame.groupby(["period", "id_pixel"])
        mean_demand = grouped["demand"].mean()
        cv_demand = grouped["demand"].std() / mean_demand.replace(0, np.nan)
        summary = pd.DataFrame(
            {
                "demand": mean_demand,
                "cv": cv_demand,
                "stop": grouped["stop"].mean(),
                "drop": grouped["drop"].mean(),
            }
        ).reset_index()
        summaries[method] = summary.set_index(["period", "id_pixel"]).to_dict("index")
        all_mean.append(mean_demand.to_numpy())
        all_cv.append(cv_demand.replace([np.inf, -np.inf], np.nan).dropna().to_numpy())
    mean_min, mean_max = _robust_limits(all_mean)
    cv_min, cv_max = _robust_limits(all_cv)
    cv_min = max(cv_min, 0.0)
    relative_values = []
    if inspector_payload:
        for method in METHODS:
            for period_data in inspector_payload.get("relative_cv", {}).get(method, {}).values():
                relative_values.extend(value for values in period_data.values() for value in values if value is not None)
    relative_min, relative_max = _robust_limits([np.asarray(relative_values)]) if relative_values else (0.0, 1.0)
    relative_min = max(relative_min, 0.0)

    for period in range(N_PERIODS):
        for row, method in enumerate(METHODS, start=1):
            summary = summaries[method]
            for col, metric, axis, label in (
                (1, "demand", "coloraxis", "Demanda media"),
                (2, "cv", "coloraxis2", "CV demanda"),
            ):
                for layer in layers:
                    values = {
                        pixel: summary.get((period, pixel), {}).get(metric)
                        for pixel in pixels
                        if layer_of(pixel) == layer and summary.get((period, pixel), {}).get(metric) is not None
                    }
                    custom_values = {pixel: summary.get((period, pixel), {}) for pixel in values}
                    fig.add_trace(
                        _grid_heatmap_trace(
                            values,
                            footprints,
                            layer,
                            period,
                            method,
                            metric,
                            axis,
                            label,
                            period == 0,
                            custom_values,
                        ),
                        row=row,
                        col=col,
                    )

    # One local grid and one focus outline per method row. JavaScript only
    # rewrites these six small traces on hover, keeping the HTML compact.
    initial_pixel = pixels[0]
    initial_ids = list(pixels)
    local_trace_indices, focus_trace_indices = {}, {}
    for row, method in enumerate(METHODS, start=1):
        method_data = (inspector_payload or {}).get("methods", {}).get(method, {}).get("0", {})
        z = [[None for _ in range(GRID_N_COLS)] for _ in range(GRID_N_ROWS)]
        customdata = [[None for _ in range(GRID_N_COLS)] for _ in range(GRID_N_ROWS)]
        for pixel in initial_ids:
            metrics = method_data.get(pixel, {})
            relative = (inspector_payload or {}).get("relative_cv", {}).get(method, {}).get("0", {}).get(initial_pixel, [])
            relative_value = relative[(inspector_payload or {}).get("pixel_index", {}).get(pixel, 0)] if relative else None
            for cell in footprints.get(pixel, set()):
                grid_row, grid_col = divmod(int(cell), GRID_N_COLS)
                z[grid_row][grid_col] = relative_value
                customdata[grid_row][grid_col] = [
                    pixel,
                    relative_value,
                    0,
                    method,
                    "relative_cv",
                    metrics.get("stop"),
                    metrics.get("drop"),
                    metrics.get("demand"),
                    metrics.get("cv"),
                    layer_of(pixel),
                ]
        local_trace_indices[method] = len(fig.data)
        fig.add_trace(
            go.Heatmap(
                x=GRID_X,
                y=GRID_Y,
                z=z,
                customdata=customdata,
                coloraxis="coloraxis3",
                name=f"{method}-local",
                xgap=1,
                ygap=1,
                zsmooth=False,
                hoverongaps=False,
                hovertemplate="<b>Píxel %{customdata[0]}</b><br>CV relativo al focal: %{customdata[1]:.3f}<extra></extra>",
                showscale=False,
            ),
            row=row,
            col=3,
        )
        focus_trace_indices[method] = len(fig.data)
        center = np.asarray([cell_center(cell) for cell in footprints[initial_pixel]], dtype=float).mean(axis=0)
        fig.add_trace(
            go.Scatter(
                x=[center[0]],
                y=[center[1]],
                mode="markers",
                marker=dict(symbol="square-open", size=14, color="#111827", line=dict(width=2)),
                name=f"{method}-focus",
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=3,
        )

    fig.update_layout(
        title="Comparación de grilla · período 1",
        height=980,
        margin=dict(l=25, r=125, t=55, b=45),
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        coloraxis=dict(
            cmin=mean_min,
            cmax=mean_max,
            colorscale="Viridis",
            colorbar=dict(title="Demanda (p1–p99)", x=1.02, y=0.82, len=0.25, thickness=13),
        ),
        coloraxis2=dict(
            cmin=cv_min,
            cmax=cv_max,
            colorscale="YlOrRd",
            colorbar=dict(title="CV (p1–p99)", x=1.02, y=0.50, len=0.25, thickness=13),
        ),
        coloraxis3=dict(
            cmin=relative_min,
            cmax=relative_max,
            colorscale="YlOrRd",
            colorbar=dict(title="CV relativo al focal", x=1.02, y=0.18, len=0.25, thickness=13),
        ),
    )
    x_range = [GRID_X[0] - GRID_DLON / 2, GRID_X[-1] + GRID_DLON / 2]
    y_range = [GRID_Y[0] - GRID_DLAT / 2, GRID_Y[-1] + GRID_DLAT / 2]
    for index in range(1, 10):
        xaxis = "xaxis" if index == 1 else f"xaxis{index}"
        yaxis = "yaxis" if index == 1 else f"yaxis{index}"
        scaleanchor = "y" if index == 1 else f"y{index}"
        fig.layout[xaxis].update(showgrid=True, gridcolor="#e5e7eb", scaleanchor=scaleanchor, range=x_range)
        fig.layout[yaxis].update(showgrid=True, gridcolor="#e5e7eb", range=y_range)
    fig.layout.xaxis3.update(title="Longitud")
    fig.layout.yaxis3.update(title="Latitud")
    fig.layout.xaxis6.update(title="Longitud")
    fig.layout.yaxis6.update(title="Latitud")
    fig.layout.xaxis9.update(title="Longitud")
    fig.layout.yaxis9.update(title="Latitud")
    return fig, local_trace_indices, focus_trace_indices


def _grid_inspector_html(payload: dict) -> str:
    """Return the status and JS controller for the dynamic neighborhood subplot."""
    browser_payload = dict(payload)
    n_pixels = len(payload["pixel_order"])
    relative_matrix = np.full((len(METHODS), N_PERIODS, n_pixels, n_pixels), np.nan, dtype="<f4")
    for method_index, method in enumerate(METHODS):
        for period in range(N_PERIODS):
            for focal_index, focal in enumerate(payload["pixel_order"]):
                values = payload["relative_cv"][method][str(period)][focal]
                relative_matrix[method_index, period, focal_index, :] = np.asarray(
                    [np.nan if value is None else value for value in values], dtype="<f4"
                )
    browser_payload.pop("relative_cv", None)
    browser_payload["relative_cv_values"] = base64.b64encode(relative_matrix.tobytes()).decode("ascii")
    browser_payload["relative_shape"] = list(relative_matrix.shape)
    browser_payload["relative_method_index"] = {method: index for index, method in enumerate(METHODS)}
    encoded = json.dumps(browser_payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return (
        '<div id="grid-inspector" class="grid-inspector">'
        "<h4>Explorador visual de vecinos</h4>"
        '<p class="hint" id="grid-inspector-status">Pasa el cursor sobre un píxel: la tercera columna mostrará el CV relativo a ese píxel focal.</p>'
        '<div id="grid-inspector-body"></div>'
        '<p class="hint">La tercera columna conserva los mismos límites espaciales y pinta todos los píxeles del layer activo. El color representa el CV de la razón demanda_píxel / demanda_focal; los colores más cálidos son relaciones menos estables.</p>'
        "</div>"
        f'<script type="application/json" id="grid-inspector-data">{encoded}</script>'
        """
<script>
(function () {
  const data = JSON.parse(document.getElementById('grid-inspector-data').textContent);
  const binary = atob(data.relative_cv_values);
  const bytes = Uint8Array.from(binary, (char) => char.charCodeAt(0));
  data.relative_cv_values = new Float32Array(bytes.buffer);
  const plot = document.getElementById('grid-dashboard');
  const status = document.getElementById('grid-inspector-status');
  const body = document.getElementById('grid-inspector-body');
  const layerSelect = document.getElementById('global-layer');
  const periodSelect = document.getElementById('global-period');
  let lastHover = null;
  if (!plot || !plot.on) return;

  const esc = (value) => String(value ?? '—').replace(/[&<>"']/g, (char) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[char]));
  const fmt = (value, digits = 3) => value == null || !Number.isFinite(Number(value))
    ? '—' : Number(value).toFixed(digits);

  function relativeValue(method, period, focal, target) {
    const n = data.pixel_order.length;
    const focalIndex = data.pixel_index[focal];
    const targetIndex = data.pixel_index[target];
    const methodIndex = data.relative_method_index[method];
    const index = (((methodIndex * data.relative_shape[1] + Number(period)) * n + focalIndex) * n) + targetIndex;
    const value = data.relative_cv_values[index];
    return Number.isFinite(value) ? value : null;
  }

  function updateNeighborSubplot(pixel, period, method) {
    const selectedLayer = layerSelect ? layerSelect.value : 'all';
    const periodKey = String(period);
    const ids = Object.keys(data.layers)
      .filter((id) => selectedLayer === 'all' || data.layers[id] === selectedLayer);
    const grid = data.grid;
    const buildGrid = (name) => {
      const z = Array.from({ length: grid.rows }, () => Array(grid.cols).fill(null));
      const customdata = Array.from({ length: grid.rows }, () => Array(grid.cols).fill(null));
      const methodData = data.methods[name][periodKey] || {};
      ids.forEach((id) => {
        const metrics = methodData[id] || {};
        const relativeValueForPixel = relativeValue(name, period, pixel, id);
        (grid.pixel_cells[id] || []).forEach((cell) => {
          const row = Math.floor(Number(cell) / grid.cols);
          const col = Number(cell) % grid.cols;
          z[row][col] = relativeValueForPixel;
          customdata[row][col] = [id, relativeValueForPixel, period, name, 'relative_cv', metrics.stop, metrics.drop, metrics.demand, metrics.cv, data.layers[id]];
        });
      });
      return { z: z, customdata: customdata };
    };
    Object.keys(data.labels).forEach((name) => {
      const localIndex = data.local_trace_indices && data.local_trace_indices[name];
      const focusIndex = data.focus_trace_indices && data.focus_trace_indices[name];
      if (localIndex == null || focusIndex == null) return;
      const local = buildGrid(name);
      Plotly.restyle(plot, { z: [local.z], customdata: [local.customdata] }, [Number(localIndex)]);
      const center = data.grid.centers[pixel];
      const focusVisible = (selectedLayer === 'all' || data.layers[pixel] === selectedLayer) && center;
      Plotly.restyle(plot, {
        x: [focusVisible ? [center[0]] : [null]],
        y: [focusVisible ? [center[1]] : [null]],
      }, [Number(focusIndex)]);
    });
    ['xaxis3', 'xaxis6', 'xaxis9'].forEach((axis) => {
      Plotly.relayout(plot, { [axis + '.title.text']: 'CV relativo al focal · ' + (data.labels[method] || method) });
    });
  }

  function render(event) {
    const point = event && event.points && event.points[0];
    if (!point || !Array.isArray(point.customdata) || point.customdata.length < 4) return;
    lastHover = point;
    const [pixel, , periodValue, methodValue] = point.customdata;
    if (!data.methods[methodValue] || !data.methods[methodValue][String(periodValue)]) return;
    const period = Number(periodValue);
    const method = data.methods[methodValue] ? methodValue : 'spatial_joint';
    const periodKey = String(period);
    const selected = data.methods[method][periodKey] || {};
    const selectedMetrics = selected[pixel] || {};
    const methodLabel = data.labels[method] || method;
    status.innerHTML = '<strong>' + esc(pixel) + '</strong> · período ' + (period + 1)
      + ' · layer <strong>' + esc(data.layers[pixel]) + '</strong>'
      + ' · CV relativo al focal de <strong>' + esc(methodLabel) + '</strong>';
    body.innerHTML = '<div class="inspector-metrics">'
      + '<div class="inspector-metric"><span>Promedio stop · ' + esc(methodLabel) + '</span><strong>' + fmt(selectedMetrics.stop, 2) + '</strong></div>'
      + '<div class="inspector-metric"><span>Promedio drop · ' + esc(methodLabel) + '</span><strong>' + fmt(selectedMetrics.drop, 2) + '</strong></div>'
      + '<div class="inspector-metric"><span>Promedio demanda · ' + esc(methodLabel) + '</span><strong>' + fmt(selectedMetrics.demand, 2) + '</strong></div>'
      + '<div class="inspector-metric"><span>CV demanda · ' + esc(methodLabel) + '</span><strong>' + fmt(selectedMetrics.cv) + '</strong></div>'
      + '</div>';
    updateNeighborSubplot(pixel, period, method);
  }
  plot.on('plotly_hover', render);
  plot.on('plotly_unhover', function () {});
  if (layerSelect) layerSelect.addEventListener('change', function () {
    if (lastHover) render({ points: [lastHover] });
  });
  if (periodSelect) periodSelect.addEventListener('change', function () {
    if (!lastHover) return;
    const customdata = lastHover.customdata.slice();
    customdata[2] = Number(this.value);
    render({ points: [{ customdata: customdata }] });
  });
})();
</script>"""
    )


def _global_period_control_html(configs: list[dict], layers: list[str]) -> str:
    """Sticky period/layer selectors that control every period-based map."""
    encoded = json.dumps(configs, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    options = "".join(f'<option value="{period}">Período {period + 1}</option>' for period in range(N_PERIODS))
    layer_options = '<option value="all">Todas las layers</option>' + "".join(
        f'<option value="{layer}">{layer}</option>' for layer in layers
    )
    return (
        '<div class="global-period-bar">'
        '<label for="global-period">Período</label>'
        f'<select id="global-period" aria-label="Seleccionar período">{options}</select>'
        '<label for="global-layer">Layer</label>'
        f'<select id="global-layer" class="layer-select" aria-label="Seleccionar layer">{layer_options}</select>'
        '<span class="scope">Filtros globales: actualizan mapas y explorador local.</span>'
        "</div>"
        f'<script type="application/json" id="period-config">{encoded}</script>'
        """
<script>
(function () {
  function initializePeriodControl() {
    const select = document.getElementById('global-period');
    const layerSelect = document.getElementById('global-layer');
    const configs = JSON.parse(document.getElementById('period-config').textContent);
    if (!select || !window.Plotly) return;

    function setFilters(period, layer) {
      configs.forEach((config) => {
        const plot = document.getElementById(config.id);
        if (!plot || !plot.data) return;
        const totalTraces = config.totalTraces || config.nPeriods * config.tracesPerPeriod;
        const visible = Array(totalTraces).fill(false);
        const start = Number(period) * config.tracesPerPeriod;
        for (let offset = 0; offset < config.tracesPerPeriod; offset += 1) {
          visible[start + offset] = layer === 'all' || config.traceLayers[offset] === layer;
        }
        (config.alwaysVisible || []).forEach((index) => { visible[index] = true; });
        Plotly.restyle(plot, { visible: visible });
        Plotly.relayout(plot, { 'title.text': config.title + ' · período ' + (Number(period) + 1) });
      });
    }

    function refresh() { setFilters(select.value, layerSelect.value); }
    select.addEventListener('change', refresh);
    layerSelect.addEventListener('change', refresh);
    refresh();
  }
  window.addEventListener('load', initializePeriodControl);
})();
</script>"""
    )


def _delta_map_figure(
    generated: dict[str, pd.DataFrame],
    method: str,
    pixels: list[str],
    footprints: dict[str, set[int]],
    title: str,
) -> go.Figure:
    """Compact cell-level heatmaps of demand difference and relative difference."""
    baseline = generated["independent"].groupby(["period", "id_pixel"])["demand"].mean()
    alternative = generated[method].groupby(["period", "id_pixel"])["demand"].mean()
    delta = (alternative - baseline).rename("value")
    relative = (delta / baseline.replace(0, np.nan)).rename("value")
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Diferencia absoluta", "Diferencia relativa"), horizontal_spacing=0.08)
    layers = sorted({layer_of(pixel) for pixel in pixels})
    limit = float(np.nanquantile(np.abs(delta.to_numpy()), 0.98)) or 1.0
    rel_limit = float(np.nanquantile(np.abs(relative.to_numpy()), 0.98)) or 1.0
    for period in range(N_PERIODS):
        for col, values, coloraxis, label, metric in (
            (1, delta, "coloraxis", "Δ demanda", "delta"),
            (2, relative, "coloraxis2", "Δ relativa", "relative"),
        ):
            for layer in layers:
                layer_values = {
                    pixel: float(values.get((period, pixel), np.nan))
                    for pixel in pixels
                    if layer_of(pixel) == layer and np.isfinite(values.get((period, pixel), np.nan))
                }
                custom_values = {pixel: {"demand": value} for pixel, value in layer_values.items()}
                fig.add_trace(
                    _grid_heatmap_trace(
                        layer_values,
                        footprints,
                        layer,
                        period,
                        method,
                        metric,
                        coloraxis,
                        label,
                        period == 0,
                        custom_values,
                    ),
                    row=1,
                    col=col,
                )
    fig.update_layout(
        title=title,
        height=520,
        margin=dict(l=25, r=125, t=65, b=35),
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        coloraxis=dict(
            cmin=-limit, cmax=limit, colorscale="RdBu", colorbar=dict(title="Δ demanda", x=1.02, y=0.76, len=0.38, thickness=13)
        ),
        coloraxis2=dict(
            cmin=-rel_limit,
            cmax=rel_limit,
            colorscale="RdBu",
            colorbar=dict(title="Δ relativa", x=1.02, y=0.24, len=0.38, thickness=13),
        ),
    )
    fig.layout.xaxis.update(showgrid=True, gridcolor="#e5e7eb", scaleanchor="y")
    fig.layout.yaxis.update(showgrid=True, gridcolor="#e5e7eb")
    fig.layout.xaxis2.update(showgrid=True, gridcolor="#e5e7eb", scaleanchor="y2")
    fig.layout.yaxis2.update(showgrid=True, gridcolor="#e5e7eb")
    return fig


def _fig_period_cv(metrics: pd.DataFrame) -> go.Figure:
    rows = []
    for method, frame in metrics.items():
        aggregate = frame.groupby(["id_scenario", "period"])["demand"].sum().reset_index()
        summary = aggregate.groupby("period")["demand"].agg(mean="mean", std="std").reset_index()
        summary["cv"] = summary["std"] / summary["mean"]
        summary["method"] = method
        rows.append(summary)
    data = pd.concat(rows, ignore_index=True)
    fig = go.Figure()
    for method in METHODS:
        view = data[data["method"] == method]
        fig.add_trace(go.Scatter(x=view["period"] + 1, y=view["cv"], mode="lines+markers", name=METHOD_LABELS[method]))
    fig.update_layout(
        height=390,
        margin=dict(l=60, r=20, t=30, b=50),
        xaxis=dict(dtick=1, title="Período"),
        yaxis_title="CV de demanda agregada",
        plot_bgcolor="white",
    )
    return fig


def _fig_distance(distance: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for method in ("historical",) + METHODS:
        view = distance[distance["method"] == method]
        if not view.empty:
            fig.add_trace(
                go.Scatter(
                    x=view["distance_km"],
                    y=view["correlation"],
                    mode="lines+markers",
                    name="Histórico" if method == "historical" else METHOD_LABELS[method],
                )
            )
    fig.update_layout(
        height=390,
        margin=dict(l=60, r=20, t=30, b=50),
        xaxis_title="Distancia entre píxeles (km)",
        yaxis_title="Correlación de log-demanda normalizada",
        plot_bgcolor="white",
    )
    return fig


def _summary_table(metrics: pd.DataFrame, neighbors: pd.DataFrame) -> str:
    marginal = metrics[metrics["metric"] == "marginal"].copy()
    marginal["Método"] = marginal["method"].map(METHOD_LABELS)
    marginal["Variable"] = marginal["quantity"]
    marginal["Error medio"] = (marginal["relative_mean_error"] * 100).map(lambda x: f"{x:+.1f}%")
    marginal["Ratio σ"] = marginal["std_ratio"].map(lambda x: f"{x:.2f}x")
    table = marginal[["Variable", "Método", "Error medio", "Ratio σ"]].to_html(index=False, classes="summary")

    neighbor = neighbors[(neighbors["quantity"] == "demand") & (neighbors["ring"] == 1)].copy()
    neighbor["Método"] = neighbor["method"].replace({"historical": "Histórico", **METHOD_LABELS})
    neighbor["Correlación vecinal"] = neighbor["mean_correlation"].map(lambda x: f"{x:.3f}")
    neighbor["P(vecino ≥ p75 | foco ≥ p75)"] = neighbor["high_given_high"].map(lambda x: f"{x:.1%}")
    neighbor_table = neighbor[["Método", "Correlación vecinal", "P(vecino ≥ p75 | foco ≥ p75)"]].to_html(
        index=False, classes="summary"
    )
    cv = metrics[(metrics["metric"] == "aggregate_cv") & (metrics["quantity"] == "demand")].set_index("method")
    neighbor_demand = neighbors[(neighbors["quantity"] == "demand") & (neighbors["ring"] == 1)].set_index("method")
    baseline_cv = float(cv.loc["independent", "generated_cv"])
    baseline_corr = float(neighbor_demand.loc["independent", "mean_correlation"])
    insight_rows = []
    for method in METHODS[1:]:
        method_cv = float(cv.loc[method, "generated_cv"])
        method_corr = float(neighbor_demand.loc[method, "mean_correlation"])
        insight_rows.append(
            {
                "Método": METHOD_LABELS[method],
                "Ganancia correlación vecinal": f"{method_corr - baseline_corr:+.3f}",
                "Ganancia CV agregado": f"{(method_cv / baseline_cv - 1) * 100:+.1f}%",
                "Distancia a histórico (correlación)": f"{abs(method_corr - float(neighbor_demand.loc['historical', 'mean_correlation'])):.3f}",
            }
        )
    insight_table = pd.DataFrame(insight_rows).to_html(index=False, classes="summary")
    return (
        f"<h3>Marginales</h3>{table}<h3>Demanda en vecinos de adyacencia directa</h3>{neighbor_table}"
        f"<h3>Ganancia contra baseline independiente</h3>{insight_table}"
    )


def _fig_neighbor(neighbors: pd.DataFrame) -> go.Figure:
    subset = neighbors[neighbors["quantity"] == "demand"]
    fig = go.Figure()
    for method in ("historical",) + METHODS:
        data = subset[subset["method"] == method].sort_values("ring")
        if data.empty:
            continue
        fig.add_trace(
            go.Bar(
                x=["Anillo 1 · borde común" if int(r) == 1 else "Vecindad ≤2 saltos" for r in data["ring"]],
                y=data["mean_correlation"],
                name="Histórico" if method == "historical" else METHOD_LABELS[method],
            )
        )
    fig.update_layout(
        height=390,
        barmode="group",
        margin=dict(l=60, r=20, t=30, b=50),
        yaxis_title="Correlación de log-demanda normalizada",
        xaxis_title="Definición de vecindad",
        plot_bgcolor="white",
    )
    return fig


def _fig_cv(metrics: pd.DataFrame) -> go.Figure:
    subset = metrics[(metrics["metric"] == "aggregate_cv") & (metrics["quantity"] == "demand")]
    fig = go.Figure()
    for method in METHODS:
        data = subset[subset["method"] == method]
        if not data.empty:
            fig.add_trace(go.Bar(x=[METHOD_LABELS[method]], y=data["generated_cv"], name=METHOD_LABELS[method]))
    historical = subset[subset["method"] == "independent"]
    if not historical.empty:
        fig.add_hline(y=float(historical.iloc[0]["historical_cv"]), line_dash="dash", annotation_text="Histórico")
    fig.update_layout(
        height=360,
        showlegend=False,
        margin=dict(l=60, r=20, t=30, b=50),
        yaxis_title="CV de demanda agregada por período",
        plot_bgcolor="white",
    )
    return fig


def render(data: ComparisonReportData, output_path: Path) -> Path:
    regime, pixels, generated = data.regime, data.pixels, data.generated
    metrics, neighbors, distance, footprints = data.metrics, data.neighbors, data.distance, data.footprints
    hist_cv = data.historical_cv
    spatial_neighbor = neighbors[
        (neighbors["method"] == "spatial_joint") & (neighbors["quantity"] == "demand") & (neighbors["ring"] == 1)
    ]["mean_correlation"].iloc[0]
    independent_neighbor = neighbors[
        (neighbors["method"] == "independent") & (neighbors["quantity"] == "demand") & (neighbors["ring"] == 1)
    ]["mean_correlation"].iloc[0]

    grid_payload = {"labels": METHOD_LABELS, **data.grid_payload}
    grid_figure, local_trace_indices, focus_trace_indices = _all_methods_grid_figure(
        generated, pixels, footprints, inspector_payload=grid_payload
    )
    grid_payload["local_trace_indices"] = local_trace_indices
    grid_payload["focus_trace_indices"] = focus_trace_indices
    grid_figure_html = fig_html(grid_figure, div_id="grid-dashboard")
    layers = sorted({layer_of(pixel) for pixel in pixels})
    grid_trace_layers = [layer for method in METHODS for _metric in ("demand", "cv") for layer in layers]
    period_configs = [
        {
            "id": "grid-dashboard",
            "tracesPerPeriod": len(METHODS) * 2 * len(layers),
            "totalTraces": len(grid_figure.data),
            "traceLayers": grid_trace_layers,
            "alwaysVisible": list(grid_payload["local_trace_indices"].values())
            + list(grid_payload["focus_trace_indices"].values()),
            "nPeriods": N_PERIODS,
            "title": "Comparación de grilla",
        }
    ]
    map_sections = [
        f'<div class="subsection"><h3>Los tres métodos: demanda media y CV por píxel</h3>'
        f'<span class="scale-note">Escalas comparables en los seis paneles: demanda usa <strong>Viridis</strong> y CV usa <strong>YlOrRd</strong>, con límites comunes p1–p99; los extremos se saturan para que no dominen el mapa.</span>'
        f"{grid_figure_html}{_grid_inspector_html(grid_payload)}</div>"
    ]
    for method in METHODS[1:]:
        delta_id = f"delta-{method}"
        period_configs.append(
            {
                "id": delta_id,
                "tracesPerPeriod": 2 * len(layers),
                "traceLayers": [layer for _metric in ("delta", "relative") for layer in layers],
                "nPeriods": N_PERIODS,
                "title": METHOD_LABELS[method] + " — diferencia vs baseline",
            }
        )
        map_sections.append(
            f'<div class="subsection"><h3>Diferencia contra baseline: {METHOD_LABELS[method]}</h3>'
            f'{fig_html(_delta_map_figure(generated, method, pixels, footprints, METHOD_LABELS[method] + " — diferencia vs baseline"), div_id=delta_id)}</div>'
        )
    period_control = _global_period_control_html(period_configs, layers)

    methods_section = """
<div class="section"><h2>Cómo se generan los escenarios y cómo leer las métricas</h2>
<p class="method-intro">Un <strong>escenario</strong> es una posible realización de la demanda para los 12 períodos y todos los píxeles. Para generar un escenario partimos del nivel esperado de cada píxel y agregamos un <strong>shock aleatorio</strong>: una variación positiva aumenta la demanda de ese período y una variación negativa la reduce. La diferencia entre los métodos está en si esos shocks se generan por separado o si se comparten entre píxeles.</p>
<div class="method-grid">
<article class="method-card baseline"><h3>Baseline: muestreo independiente</h3><p>Se sortea un shock distinto para cada píxel. Si un píxel recibe un shock positivo, eso no hace más probable que su vecino también lo reciba. Conserva la demanda media y la variabilidad de cada píxel, pero elimina el movimiento conjunto espacial. Es la referencia mínima de correlación.</p></article>
<article class="method-card parametric"><h3>Enfoque paramétrico: cópula espacial</h3><p>Primero se genera un campo de shocks relacionados: píxeles cercanos tienden a recibir shocks parecidos y píxeles lejanos se relacionan menos. La intensidad de esa relación se calibra con la historia y decae con la distancia. Mantiene las distribuciones marginales y agrega dependencia espacial de forma controlada.</p></article>
<article class="method-card empirical"><h3>Enfoque empírico: bootstrap histórico conjunto</h3><p>Se elige un mes histórico y se reutiliza su vector completo de shocks para todos los píxeles. Así se conserva directamente el patrón espacial que ocurrió en la realidad. Es fácil de interpretar, pero solo puede repetir combinaciones parecidas a las observadas.</p></article>
</div>
<div class="metric-guide"><h3>Qué significa cada métrica</h3>
<p><strong>Demanda media:</strong> promedio de la demanda simulada entre escenarios. <strong>CV por píxel:</strong> desviación estándar dividida por la media; mide cuánto varía cada píxel en términos relativos. <strong>CV agregado:</strong> aplica el mismo cálculo a la suma de demanda de todos los píxeles; muestra cuánto riesgo total se conserva o se cancela. <strong>Correlación vecinal:</strong> mide si las desviaciones de dos píxeles suben y bajan juntas. <strong>Probabilidad de co-excedencia:</strong> mide si un vecino también entra en un nivel alto cuando el píxel focal entra en un nivel alto.</p>
<p><strong>¿Por qué p75?</strong> El p75 es el valor que deja al 75% de las observaciones por debajo y al 25% por encima. Usarlo convierte “demanda alta” en una regla comparable para píxeles densos y menos densos: cada píxel se evalúa contra su propio nivel habitual, no contra un umbral absoluto que favorecería a los píxeles grandes. El 25% deja suficientes eventos para estimar la métrica sin definir “alto” con casos demasiado extremos.</p>
<p class="metric-formula"><strong>P(vecino ≥ p75 | foco ≥ p75)</strong> = casos en que ambos están altos / casos en que el foco está alto</p>
<p>“Foco” es el píxel que estamos condicionando y “vecino” es el píxel del par analizado. Por ejemplo, un resultado de 40% significa: entre las ocasiones en que el foco estuvo en su 25% superior, el vecino también estuvo en su 25% superior en 4 de cada 10 ocasiones. Si los píxeles fueran independientes, esperaríamos aproximadamente 25%; un valor mayor indica co-excedencia espacial. No significa que el vecino tenga 40% de demanda, sino 40% de probabilidad de superar <em>su propio</em> p75 bajo esa condición.</p>
<p><strong>Ganancia frente al baseline:</strong> se reporta cuánto aumenta la correlación vecinal, el CV de la demanda agregada o la probabilidad de co-excedencia respecto del muestreo independiente. Es una ganancia estadística: indica cuánto movimiento conjunto recuperamos, pero todavía no es una ganancia monetaria ni una reducción de costos.</p>
</div>
<details><summary>Definiciones técnicas y supuestos</summary><p>La correlación se calcula sobre log-demanda centrada por píxel y la probabilidad se acumula sobre todos los pares del anillo y sus observaciones comparables. El p75 se calcula por píxel, por lo que la métrica compara eventos relativos. La referencia de 25% es exacta solo como intuición bajo independencia y puede variar levemente por el tamaño finito de la muestra y la forma de agregar los pares.</p></details>
</div>
"""

    html = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8"><title>Comparación de enfoques de demanda espacial</title>
<script src="{PLOTLY_CDN}"></script><style>{CSS}</style></head><body>
<div id="top-bar"><h1>Comparación: independencia vs dependencia espacial</h1>
<p>Régimen {regime} · {len(pixels)} píxeles · {N_PERIODS} períodos ·
{data.n_scenarios} escenarios pareados · panel histórico usado como referencia.</p></div>
{period_control}
{methods_section}
<div class="section"><p>Se comparan tres formas de generar escenarios: un baseline sin dependencia espacial, un modelo paramétrico que calibra una cópula espacial y un modelo empírico que remuestrea configuraciones históricas completas. El baseline es estrictamente independiente entre píxeles; la varianza del shock común se incorpora a la dispersión individual para conservar la marginal.</p>{_summary_table(metrics, neighbors)}</div>
<div class="section"><h2>Mapas por píxel y período</h2><p>Usa el selector global fijo de arriba para cambiar simultáneamente todos los mapas. La demanda muestra el promedio entre escenarios; el CV por píxel mide la variación entre escenarios para ese período. Los mapas de diferencia muestran cuánto cambia cada píxel respecto del baseline independiente.</p>{''.join(map_sections)}</div>
<div class="section"><h2>¿Los vecinos se mueven juntos?</h2><p>La correlación se calcula sobre log-demanda centrada por píxel. El <strong>anillo 1</strong> contiene píxeles cuyas huellas comparten directamente un borde de celda en la grilla. La <strong>vecindad de hasta 2 saltos</strong> contiene esos vecinos directos más los píxeles que se alcanzan pasando por un píxel intermedio; por eso es una vecindad acumulada, no un anillo exacto que excluya el primero.</p>{fig_html(_fig_neighbor(neighbors))}<p>En el anillo 1, el baseline independiente obtiene {independent_neighbor:.3f} y el enfoque paramétrico obtiene {spatial_neighbor:.3f}; el histórico entrega la referencia empírica disponible.</p></div>
<div class="section"><h2>Correlograma continuo</h2><p>La correlación se calcula para todos los pares y se agrupa por distancia entre centroides. Permite verificar si el efecto se concentra en vecinos cercanos o permanece a escala urbana.</p>{fig_html(_fig_distance(distance))}</div>
<div class="section"><h2>Riesgo de demanda agregada</h2><p>El CV mide cuánto varía la suma de demanda de los píxeles por período. Un modelo independiente tiende a cancelar el ruido específico de cada píxel.</p>{fig_html(_fig_cv(metrics))}<p>CV histórico: {hist_cv:.3f}. Los valores exactos quedan en <code>metrics.csv</code>.</p></div>
<div class="section"><h2>CV por período</h2><p>La línea muestra si las alternativas cambian el riesgo agregado de manera distinta según el período del horizonte.</p>{fig_html(_fig_period_cv(generated))}</div>
<div class="section"><h2>Archivos para análisis externo</h2><p><code>comparison_long.csv</code> contiene cada método, escenario, píxel, período, stop, drop y demand. <code>pixel_period_metrics.csv</code> contiene media, desviación, cuantiles y CV por píxel-período. <code>neighbor_metrics.csv</code> contiene correlaciones y probabilidades de co-excedencia por anillo. <code>distance_metrics.csv</code> contiene el correlograma continuo. <code>summary.json</code> resume la comparación pareada y las ganancias contra el baseline.</p></div>
</body></html>"""
    output_path.write_text(html)
    return output_path

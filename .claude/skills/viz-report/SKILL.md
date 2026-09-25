---
name: viz-report
description: Build an interactive self-contained HTML report for this project's experiment output — solution maps, sweep summaries, factor analyses. Use when asked to visualize a result JSON, compare configurations visually, add a section to an existing report, or produce a figure for the paper. Triggers: "genera el HTML", "mapa de la solución", "reporte de resultados", "summary del powerset", "gráfico de", "visualiza".
---

# Reportes HTML

This project's reports follow one house style, established by
`analysis/ca_factor_analysis.py` (multi-section analysis),
`visualization/solution_map.py` (single-solution map) and
`visualization/powerset_html.py` (sweep summary + iframe viewer). Match it —
these reports get read side by side and shared with coauthors.

**For chart design decisions** (chart type, color assignment, axis and legend
treatment, palette), read the `dataviz` skill first. This skill covers the report
*shell* and this project's specific conventions; `dataviz` covers what goes in the
plot. Do not pick chart colors ad hoc.

## Non-negotiables

1. **One HTML file per report, Plotly from CDN.** Never embed the Plotly bundle
   per figure — that produces 3 MB files and there are hundreds of them.
   ```python
   PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"
   PLOTLY_CFG = {"displayModeBar": True, "displaylogo": False}

   def to_html(fig):
       return fig.to_html(full_html=False, include_plotlyjs=False, config=PLOTLY_CFG)
   ```
   The CDN `<script>` goes once in `<head>`; every figure is inlined with
   `include_plotlyjs=False`.

2. **Every section carries a method box and an insight box.** The structure is
   fixed: title → subtitle → 📐 method → figure → 💡 insight. Reuse the helpers
   verbatim rather than reinventing the markup:
   ```python
   def insight_box(text): return f'<div class="insight"><b>💡 Por qué ocurre esto:</b> {text}</div>'
   def method_box(text):  return f'<div class="method"><b>📐 Datos y construcción:</b> {text}</div>'

   def section(title, subtitle, fig_html, insight="", method=""):
       return f"""
   <div class="section">
     <h2>{title}</h2>
     <p>{subtitle}</p>
     {method_box(method) if method else ""}
     {fig_html}
     {insight_box(insight) if insight else ""}
   </div>"""
   ```

3. **Insights are computed from the data, never hardcoded.** This is the rule that
   matters most. `ca_factor_analysis.py` has a `# ── Compute insights from data ──`
   block that derives the dominant cost component, the largest elasticity and so
   on, then interpolates the numbers into the prose. A hardcoded claim silently
   becomes false the next time the instance changes. Write
   `f"...{dominant} domina con {share:.0%}..."`, not `"line-haul domina"`.
   The method box states provenance — which instance, how many observations,
   which vehicle, euclidean vs road distances.

4. **Top bar states the scope of the run.** Dark header with the title and a
   one-line count of what was analyzed:
   ```
   <p>{n_obs:,} observaciones · {n_sat} satélites · {n_pix} píxeles · vehículo "small" · distancia euclidiana</p>
   ```

5. **Report language is Spanish** — titles, subtitles, labels, boxes. Code,
   identifiers and docstrings stay English.

## CSS

Reuse the existing block so reports are visually consistent. Palette:
background `#f4f6f8`, cards white with `border-radius: 8px` and a soft shadow,
top bar `#2c3e50`, insight box blue (`#eef6fb` / border `#3498db` / bold `#2980b9`),
method box green (`#f0fdf4` / border `#27ae60` / bold `#1e8449`). Copy the `css`
string from `analysis/ca_factor_analysis.py` rather than retyping it; if you change
it, change it there too.

## Reading result JSONs

Keys are stringified Python tuples. Always:

```python
import ast
def _parse_key(key_str: str) -> tuple:
    return ast.literal_eval(key_str)

x_vals = {_parse_key(k): v for k, v in data["X"].items()}
```

`X` is `(satellite, pixel, period, scenario)` and `W` is `(pixel, period, scenario)`,
with the scenario index a **string**. Solution maps average across scenarios —
`mean_x[i] = sum(x_vals.get((i, k, t, n), 0.0) for n in scenarios) / n_scenarios` —
because a pixel can be split across satellites in different scenarios; a report
showing a single assignment must say it is the scenario mean.

Cost components do not sum to `objective` (the `1/N` applies only to
scenario-dependent terms). If a figure decomposes cost, reconstruct the objective
and check it before rendering, and state the scaling in the method box.

## Interaction patterns

- **Solution map** (`solution_map.py`): Plotly `updatemenus` dropdowns for period
  and layer, pixels as rectangles via `_pixel_rect`, plus an *invisible
  center-point marker layer* carrying the full hover text for every pixel (demand,
  cost, assigned satellite, fleet size, and the cost of every alternative
  satellite). Keep the invisible hover layer — rectangle-only hover is unreliable.
- **Sweep summary** (`powerset_html.py`): scatter of Δ% objective vs. active
  satellite count with the efficiency frontier, boxplot by subset size, sortable
  filterable table, and a `<select>` that swaps an `<iframe>` src to the
  per-config map. Individual maps are written next to `summary.html` and named by
  the `+`-joined satellite label, so the iframe can address them as
  `label + '.html'`. Support a `--skip-individual` flag to rebuild only the summary.

## Output and mechanics

Split every report in two, following `src/visualization/`:
- a **renderer** in `src/visualization/{scenarios,results}/<name>_report.py`: a
  dataclass with what it shows plus `render(data, output_path) -> Path`. It imports only
  `src.tools` / `src.core` and the toolkit in `src/visualization/html.py`
  (`fig_html`, `section`, `method_box`, `insight_box`, `page`, `top_bar`, `BASE_CSS`)
  and `components/labels.py`;
- a **builder** in `src/scenarios/reports.py` or `src/optimization/reports.py` that loads
  the artifacts, computes the metrics worth persisting, and calls the renderer.

Reports go under the artifact they describe, in its `reports/` directory (e.g.
`results/runs/r1/reports/`), which is excluded from the artifact's content digest.
Expose them through the package CLI (`optimize report <run>`, `scenarios explore <v>`),
not a module-level `__main__`. Never commit generated HTML.

After generating, report the output path and offer `open <path>` — do not open a
browser unprompted.

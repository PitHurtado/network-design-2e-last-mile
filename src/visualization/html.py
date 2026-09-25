"""HTML building blocks shared by every report."""

import plotly.graph_objects as go

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"
PLOTLY_CFG = {"displayModeBar": True, "displaylogo": False}

METHOD_LABEL = "📐 Datos y construcción:"
INSIGHT_LABEL = "💡 Por qué ocurre esto:"

BASE_CSS = """
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


def fig_html(fig: go.Figure, config: dict | None = None, div_id: str | None = None) -> str:
    """A figure as an embeddable div; Plotly itself is loaded once from the CDN."""
    kwargs = {"full_html": False, "include_plotlyjs": False, "config": config or PLOTLY_CFG}
    if div_id:
        kwargs["div_id"] = div_id
    return fig.to_html(**kwargs)


def insight_box(text: str, label: str | None = INSIGHT_LABEL) -> str:
    return f'<div class="insight"><b>{label}</b> {text}</div>' if label else f'<div class="insight">💡 {text}</div>'


def method_box(text: str, label: str | None = METHOD_LABEL) -> str:
    return f'<div class="method"><b>{label}</b> {text}</div>' if label else f'<div class="method">📐 {text}</div>'


def section(title: str, subtitle: str, body: str, insight: str = "", method: str = "", labelled: bool = True) -> str:
    """One report section: title, subtitle, method box, content, insight box.

    `labelled=False` renders the boxes with their icon only, without the bold caption.
    """
    method_html = method_box(method, METHOD_LABEL if labelled else None) if method else ""
    insight_html = insight_box(insight, INSIGHT_LABEL if labelled else None) if insight else ""
    return f"""
<div class="section">
  <h2>{title}</h2>
  <p>{subtitle}</p>
  {method_html}
  {body}
  {insight_html}
</div>"""


def top_bar(title: str, subtitle: str = "") -> str:
    return f'<div id="top-bar"><h1>{title}</h1><p>{subtitle}</p></div>'


def page(title: str, body: str, css: str = BASE_CSS, head_extra: str = "") -> str:
    """A complete self-contained document."""
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <script src="{PLOTLY_CDN}"></script>
  <style>{css}</style>{head_extra}
</head>
<body>
{body}
</body>
</html>"""

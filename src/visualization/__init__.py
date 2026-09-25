"""Interactive HTML reports: a shared Plotly/HTML toolkit and one renderer per report.

`src.visualization` imports only `src.tools` and `src.core`. Producers
(`src.scenarios.reports`, `src.optimization.reports`) load their artifacts, compute the
metrics worth persisting, and hand the renderer plain DataFrames and dataclasses; a
renderer may still aggregate for display, but never reads study artifacts itself.

House style (`.claude/skills/viz-report`): Plotly via CDN, a method box and an insight
box per section, insights computed from the data rather than written by hand.
"""

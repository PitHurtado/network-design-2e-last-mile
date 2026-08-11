"""Generate interactive HTML with pixel maps and fleet capacity analysis.

Sections:
  1. Pixel demand map (per layer)
  2. Satellite assignment map (per layer, Euclidean distance)
  3. Fleet analysis — how many vehicles each satellite needs

Run from project root:
    poetry run python notebooks/mvp_parametros/generate_pixel_maps.py
"""

import ast
import json
import logging
import math
import os
import sys

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.io as pio

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pixel_maps.html")

# src/ imports for Continuous Approximation
sys.path.insert(0, ROOT)
from src.data.etl import get_facilities, get_scenario, get_vehicles
from src.routing_tools.continuous_approximation import ContinuousApproximation
from src.utils.scenario import Scenario

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#fffac8", "#800000", "#aaffc3",
]

OPACITY_FILL    = 0.5
MAP_STYLE       = "open-street-map"
ZOOM            = 11
N_SCENARIOS     = 500
VEHICLE_TYPE    = "small"
CAPACITY_LEVELS = [2, 4, 6, 8, 10, 12]
BATCH_SIZE      = 50    # scenarios per CA batch (memory management)
THETAS          = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]  # % DC direct service

MONTHS = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
          "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("Loading geojson...")
with open(os.path.join(DATA, "pixels/grid_pixels.geojson")) as f:
    geojson_data = json.load(f)

pixel_coords: dict[int, list] = {}
for feat in geojson_data["features"]:
    pid = int(feat["properties"]["pixel"])
    pixel_coords[pid] = feat["geometry"]["coordinates"][0]

all_pids = sorted(pixel_coords.keys())

print("Loading pixels metadata...")
pixels_df = pd.read_excel(os.path.join(DATA, "pixels/input_pixels.xlsx"), engine="openpyxl")
pixels_df["pixel"] = pixels_df["pixel"].astype(int)
pixels_df["layer"] = pixels_df["layer"].str.upper()

print("Loading distance matrix...")
dist_df = pd.read_excel(
    os.path.join(DATA, "distances/input_matrix_distance_facilities_pixels.xlsx"),
    engine="openpyxl",
)
dist_df["layer"]       = dist_df["layer"].str.upper()
dist_df["id_facility"] = dist_df["id_facility"].str.upper()
dist_df["pixel"]       = dist_df["pixel"].astype(int)

# Extract DC distances before removing DC rows
_dc_rows = dist_df[dist_df["id_facility"] == "DC"].copy()
_dc_rows["pixel_id"] = _dc_rows["layer"] + "-" + _dc_rows["pixel"].astype(str)
dc_dist_dict: dict[str, float] = dict(zip(_dc_rows["pixel_id"], _dc_rows["distance"]))

dist_df = dist_df[dist_df["id_facility"] != "DC"]

print("Loading facilities...")
fac_df = pd.read_excel(os.path.join(DATA, "facilities/input_facilities.xlsx"), engine="openpyxl")
fac_df["id_facility"] = fac_df["id_facility"].str.upper()


# ---------------------------------------------------------------------------
# 2. Satellite assignment — Euclidean (Haversine)
# Road distance matrix had 5 anomalous entries (now corrected in Excel), but
# Euclidean is used here to maintain visual consistency with the maps.
# ---------------------------------------------------------------------------
def _haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


fac_pos = {row["id_facility"]: (row["lon"], row["lat"]) for _, row in fac_df.iterrows()}

pix_latlon = {
    (row["layer"], int(row["pixel"])): (row["lon"], row["lat"])
    for _, row in pixels_df.iterrows()
}

assignment: dict[tuple[str, int], str] = {}
for (layer, pid), (plon, plat) in pix_latlon.items():
    nearest = min(fac_pos, key=lambda s: _haversine(plon, plat, fac_pos[s][0], fac_pos[s][1]))
    assignment[(layer, pid)] = nearest

satellites = sorted(fac_df["id_facility"].tolist())
sat_color  = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(satellites)}

# Assigned pixels per satellite (both layers combined, format "A-148")
assigned_by_sat: dict[str, set] = {}
for (layer, pid), sat in assignment.items():
    assigned_by_sat.setdefault(sat, set()).add(f"{layer}-{pid}")


# ---------------------------------------------------------------------------
# 3. Map helpers
# ---------------------------------------------------------------------------
def poly_coords(pids: list[int]) -> tuple[list, list]:
    lons, lats = [], []
    for pid in pids:
        if pid not in pixel_coords:
            continue
        ring = pixel_coords[pid]
        lons.extend([c[0] for c in ring] + [None])
        lats.extend([c[1] for c in ring] + [None])
    return lons, lats


road_dist: dict[tuple[str, int], dict[str, float]] = {}
for (layer, pixel), grp in dist_df.groupby(["layer", "pixel"]):
    road_dist[(layer, int(pixel))] = dict(zip(grp["id_facility"], grp["distance"]))

pix_centroid = {
    (row["layer"], int(row["pixel"])): (row["lon"], row["lat"])
    for _, row in pixels_df.iterrows()
}


def _centroid(pid: int) -> tuple[float, float]:
    ring = pixel_coords[pid]
    r = ring[:-1]
    return sum(c[0] for c in r) / len(r), sum(c[1] for c in r) / len(r)


ANOMALOUS = {
    ("A", 148, "LLOJETA"), ("B", 147, "LLOJETA"),
    ("A", 140, "COTA_COTA"), ("A", 147, "LLOJETA"),
    ("A", 192, "ZONA_CEMENTERIO"),
}

DC_LON = -68.076151
DC_LAT = -16.512483


def _hover_markers(layer: str, mode: str) -> go.Scattermap:
    demand_set = set(pixels_df[pixels_df["layer"] == layer]["pixel"].tolist())
    hover_lons, hover_lats, texts = [], [], []
    for pid in sorted(demand_set):
        clon, clat = pix_centroid.get((layer, pid), _centroid(pid))
        sat = assignment.get((layer, pid), "N/A")
        sat_label = sat.replace("_", " ").title() if sat != "N/A" else "N/A"
        eucl = round(_haversine(clon, clat, fac_pos[sat][0], fac_pos[sat][1]), 2) if sat != "N/A" else 0
        rd   = road_dist.get((layer, pid), {}).get(sat)
        rd_str = f"{rd:.1f} km" if rd is not None else "—"
        warn = "<br>⚠️ dist. vial anómala en datos" if (layer, pid, sat) in ANOMALOUS else ""
        if mode == "demand":
            text = f"<b>Pixel {layer}-{pid}</b><br>Con demanda"
        else:
            text = (f"<b>Pixel {layer}-{pid}</b><br>"
                    f"Satélite: {sat_label}<br>"
                    f"Dist. eucl.: {eucl} km<br>"
                    f"Dist. vial: {rd_str}{warn}")
        hover_lons.append(clon)
        hover_lats.append(clat)
        texts.append(text)
    return go.Scattermap(
        lon=hover_lons, lat=hover_lats, mode="markers",
        marker=dict(size=10, opacity=0.001, color="rgba(0,0,0,0)"),
        hovertemplate="%{text}<extra></extra>", text=texts,
        showlegend=False, name="_hover",
    )


def _hex_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _scatter(lons, lats, color, name, show_legend, opacity=OPACITY_FILL):
    return go.Scattermap(
        lon=lons, lat=lats, mode="lines", fill="toself",
        fillcolor=_hex_rgba(color, opacity),
        line=dict(color=color, width=0.8),
        name=name, showlegend=show_legend, hoverinfo="skip",
    )


def _bg_trace(bg_pids):
    lons, lats = poly_coords(bg_pids)
    return go.Scattermap(
        lon=lons, lat=lats, mode="lines", fill="toself",
        fillcolor="rgba(180,180,180,0.08)", line=dict(color="#bbbbbb", width=0.5),
        opacity=1.0, hoverinfo="skip", showlegend=False, name="_bg",
    )


def _sat_marker(sat_name, lon, lat, color):
    label = sat_name.replace("_", " ").title()
    return go.Scattermap(
        lon=[lon, lon], lat=[lat, lat], mode="markers+text",
        marker=dict(size=14, color="white", opacity=1.0),
        text=["", label], textposition="top right",
        textfont=dict(size=11, color="#2c3e50"),
        hovertemplate=f"<b>{label}</b><extra></extra>",
        showlegend=False, name=f"_sat_{sat_name}",
    )


def _sat_marker_dot(sat_name, lon, lat, color):
    return go.Scattermap(
        lon=[lon], lat=[lat], mode="markers",
        marker=dict(size=9, color=color, opacity=1.0),
        hoverinfo="skip", showlegend=False, name=f"_dot_{sat_name}",
    )


def _dc_markers() -> list:
    """DC: white outer ring + dark fill + label."""
    return [
        go.Scattermap(
            lon=[DC_LON, DC_LON], lat=[DC_LAT, DC_LAT],
            mode="markers+text",
            marker=dict(size=18, color="white", opacity=1.0),
            text=["", "DC"],
            textposition="top right",
            textfont=dict(size=12, color="#2c3e50"),
            hovertemplate="<b>Centro de Distribución (DC)</b><extra></extra>",
            showlegend=False, name="_dc",
        ),
        go.Scattermap(
            lon=[DC_LON], lat=[DC_LAT],
            mode="markers",
            marker=dict(size=11, color="#2c3e50", opacity=1.0),
            hoverinfo="skip", showlegend=False, name="_dc_dot",
        ),
    ]


# ---------------------------------------------------------------------------
# 4. Map figures
# ---------------------------------------------------------------------------
center_lat = pixels_df["lat"].mean()
center_lon = pixels_df["lon"].mean()
map_layout = dict(style=MAP_STYLE, center=dict(lat=center_lat, lon=center_lon), zoom=ZOOM)
fig_layout = dict(
    height=560, margin=dict(l=0, r=0, t=36, b=0),
    paper_bgcolor="#f4f6f8", plot_bgcolor="#f4f6f8",
    font=dict(family="sans-serif", color="#2c3e50"),
    showlegend=True,
    legend=dict(bgcolor="rgba(255,255,255,0.88)", bordercolor="#cccccc",
                borderwidth=1, font=dict(size=11), x=0.01, y=0.99,
                xanchor="left", yanchor="top"),
)


def make_demand_fig(layer: str) -> go.Figure:
    demand_set  = set(pixels_df[pixels_df["layer"] == layer]["pixel"].tolist())
    no_demand   = [p for p in all_pids if p not in demand_set]
    demand_pids = [p for p in all_pids if p in demand_set]
    fig = go.Figure()
    fig.add_trace(_bg_trace(no_demand))
    lons, lats = poly_coords(demand_pids)
    fig.add_trace(_scatter(lons, lats, "#1565C0", f"Con demanda ({len(demand_pids)} píxeles)", True))
    fig.add_trace(_hover_markers(layer, "demand"))
    for t in _dc_markers():
        fig.add_trace(t)
    fig.update_layout(**fig_layout,
                      title=dict(text=f"Píxeles con demanda — Capa {layer}", font=dict(size=13), x=0.5),
                      map=map_layout)
    return fig


def make_assignment_fig(layer: str) -> go.Figure:
    demand_set = set(pixels_df[pixels_df["layer"] == layer]["pixel"].tolist())
    no_demand  = [p for p in all_pids if p not in demand_set]
    fig = go.Figure()
    fig.add_trace(_bg_trace(no_demand))
    for sat in satellites:
        sat_pids = [p for p in demand_set if assignment.get((layer, p)) == sat]
        if not sat_pids:
            continue
        lons, lats = poly_coords(sat_pids)
        fig.add_trace(_scatter(lons, lats, sat_color[sat],
                               f"{sat.replace('_',' ').title()} ({len(sat_pids)})", True))
    fig.add_trace(_hover_markers(layer, "assign"))
    for sat in satellites:
        row_s = fac_df[fac_df["id_facility"] == sat].iloc[0]
        fig.add_trace(_sat_marker(sat, row_s["lon"], row_s["lat"], sat_color[sat]))
        fig.add_trace(_sat_marker_dot(sat, row_s["lon"], row_s["lat"], sat_color[sat]))
    for t in _dc_markers():
        fig.add_trace(t)
    fig.update_layout(**fig_layout,
                      title=dict(text=f"Asignación geográfica — Capa {layer}", font=dict(size=13), x=0.5),
                      map=map_layout)
    return fig


print("Building map figures...")
map_figs = {
    ("A", "demand"): make_demand_fig("A"),
    ("A", "assign"): make_assignment_fig("A"),
    ("B", "demand"): make_demand_fig("B"),
    ("B", "assign"): make_assignment_fig("B"),
}


# ---------------------------------------------------------------------------
# 5. Fleet analysis via Continuous Approximation
# ---------------------------------------------------------------------------
def compute_fleet_df(n: int) -> pd.DataFrame:
    """Run CA for n scenarios and return per-(satellite, period, scenario) fleet."""
    vehicles   = get_vehicles()
    facilities = get_facilities()

    logging.disable(logging.INFO)   # suppress verbose per-scenario logs

    records = []
    for batch_start in range(0, n, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n)
        batch_ids = [str(i) for i in range(batch_start + 1, batch_end + 1)]

        scenarios = {}
        for sid in batch_ids:
            px = get_scenario(id_scenario=sid)
            scenarios[sid] = Scenario(id_scenario=sid, pixels=px, periods=12)

        ca = ContinuousApproximation(
            scenarios=scenarios, facilities=facilities,
            vehicles=vehicles, use_euclidean_distance=False,
        )
        scenarios = ca.run_continuous_approximation()

        for w, sc in scenarios.items():
            fs = sc.fleet_size["facility"]
            for sat in satellites:
                for t in range(12):
                    total = sum(
                        fs.get((sat, j, VEHICLE_TYPE, t, w), 0)
                        for j in assigned_by_sat.get(sat, set())
                    )
                    records.append({
                        "scenario": w,
                        "satellite": sat,
                        "period": t + 1,
                        "fleet": total,
                        "fleet_ceil": math.ceil(total),
                    })

        print(f"  CA progress: {batch_end}/{n} scenarios")
        del scenarios

    logging.disable(logging.NOTSET)
    return pd.DataFrame(records)


def compute_pixel_fleet_df(n: int) -> pd.DataFrame:
    """Per-pixel fleet: satellite small vans + DC large trucks, per scenario/period."""
    vehicles   = get_vehicles()
    facilities = get_facilities()
    pix_to_sat = {j: sat for sat, pxs in assigned_by_sat.items() for j in pxs}

    logging.disable(logging.INFO)
    records = []
    for batch_start in range(0, n, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n)
        batch_ids = [str(i) for i in range(batch_start + 1, batch_end + 1)]

        scenarios = {}
        for sid in batch_ids:
            px = get_scenario(id_scenario=sid)
            scenarios[sid] = Scenario(id_scenario=sid, pixels=px, periods=12)

        ca = ContinuousApproximation(
            scenarios=scenarios, facilities=facilities,
            vehicles=vehicles, use_euclidean_distance=False,
        )
        scenarios = ca.run_continuous_approximation()

        for w, sc in scenarios.items():
            sat_fs = sc.fleet_size["facility"]
            dc_fs  = sc.fleet_size["dc"]
            for (j, _v, t, _w), dc_fleet in dc_fs.items():
                sat = pix_to_sat.get(j)
                if sat is None:
                    continue
                sat_fleet = sat_fs.get((sat, j, VEHICLE_TYPE, t, w), 0)
                records.append({
                    "scenario": w, "pixel_id": j, "satellite": sat,
                    "period": t + 1,
                    "sat_fleet": sat_fleet,
                    "sat_ceil": math.ceil(sat_fleet),
                    "dc_fleet": dc_fleet,
                    "dc_ceil": math.ceil(dc_fleet),
                })

        print(f"  Pixel CA progress: {batch_end}/{n} scenarios")
        del scenarios

    logging.disable(logging.NOTSET)
    return pd.DataFrame(records)


_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"fleet_cache_{N_SCENARIOS}.csv")
if os.path.exists(_CACHE):
    print(f"\nLoading fleet data from cache ({_CACHE})...")
    fleet_df = pd.read_csv(_CACHE)
else:
    print(f"\nRunning CA for {N_SCENARIOS} scenarios (vehicle: {VEHICLE_TYPE})...")
    fleet_df = compute_fleet_df(N_SCENARIOS)
    fleet_df.to_csv(_CACHE, index=False)
    print(f"  Fleet data cached to {_CACHE}")

# Peak fleet per (satellite, scenario): max over 12 periods
peak_df = fleet_df.groupby(["satellite", "scenario"])["fleet_ceil"].max().reset_index()

# Summary statistics
def _p(arr, q): return int(math.ceil(np.percentile(arr, q)))

stats_df = (
    peak_df.groupby("satellite")["fleet_ceil"]
    .agg(
        p50=lambda x: _p(x, 50),
        p75=lambda x: _p(x, 75),
        p90=lambda x: _p(x, 90),
        p95=lambda x: _p(x, 95),
        max_fleet="max",
    )
    .reset_index()
)

def _recommend(p90: int) -> int:
    for lvl in CAPACITY_LEVELS:
        if lvl >= p90:
            return lvl
    return CAPACITY_LEVELS[-1]

stats_df["recommended_cap"] = stats_df["p90"].apply(_recommend)

n_pixels_per_sat = (
    pd.DataFrame([{"satellite": v} for v in assignment.values()])
    .value_counts()
    .reset_index(name="n_pixels")
)
stats_df = stats_df.merge(n_pixels_per_sat, on="satellite", how="left").fillna(0)
stats_df["n_pixels"] = stats_df["n_pixels"].astype(int)

# P90 by (satellite × period) for heatmap
p90_period = (
    fleet_df.groupby(["satellite", "period"])["fleet_ceil"]
    .apply(lambda x: _p(x, 90))
    .reset_index(name="p90")
)

print("\n=== Fleet capacity stats (peak period, across scenarios) ===")
print(stats_df[["satellite", "n_pixels", "p50", "p75", "p90", "p95", "max_fleet", "recommended_cap"]]
      .to_string(index=False))


# ---------------------------------------------------------------------------
# 5b-cap. Capacity level proposals per satellite
# ---------------------------------------------------------------------------
_extra_pct = (
    peak_df.groupby("satellite")["fleet_ceil"]
    .agg(p10=lambda x: _p(x, 10), p25=lambda x: _p(x, 25))
    .reset_index()
)
cap_df = stats_df.merge(_extra_pct, on="satellite")


def _rup2(x: float) -> int:
    """Round x up to the next even integer."""
    c = math.ceil(x)
    return c + (c % 2)


def _levels_A(row) -> list[int]:
    """Proposal A — full useful range.
    From 2 vehicles below the P25 floor (min=2) up to 2 above P95, step 2.
    Captures the realistic operational spread without tiny unusable levels.
    """
    lo = max(2, _rup2(row["p25"]) - 2)
    hi = _rup2(row["p95"]) + 2
    return list(range(lo, hi + 1, 2))


def _levels_B(row) -> list[int]:
    """Proposal B — 3 essential levels only.
    c_min: covers P50  (minimum viable operation — demand in half the scenarios)
    c_rec: covers P90  (recommended — demand in 9 out of 10 scenarios)
    c_max: covers 100% (safety ceiling — worst observed scenario)
    Deduplicates when levels coincide (e.g. if P50 == P90).
    """
    c_min = _rup2(row["p50"])
    c_rec = _rup2(row["p90"])
    c_max = _rup2(row["max_fleet"])
    return sorted(set([c_min, c_rec, c_max]))


cap_df["levels_A"] = cap_df.apply(_levels_A, axis=1)
cap_df["levels_B"] = cap_df.apply(_levels_B, axis=1)
cap_df["n_levels_A"] = cap_df["levels_A"].apply(len)
cap_df["n_levels_B"] = cap_df["levels_B"].apply(len)

print("\n=== Propuesta A — Rango operacional completo ===")
for _, r in cap_df.sort_values("p90", ascending=False).iterrows():
    print(f"  {r['satellite']:20s}  P25={r['p25']:2d}  P90={r['p90']:2d}  "
          f"P95={r['p95']:2d}  →  {r['levels_A']}")

print("\n=== Propuesta B — 3 niveles esenciales ===")
for _, r in cap_df.sort_values("p90", ascending=False).iterrows():
    print(f"  {r['satellite']:20s}  P50={r['p50']:2d}  P90={r['p90']:2d}  "
          f"max={r['max_fleet']:2d}  →  {r['levels_B']}")

# ---------------------------------------------------------------------------
# OPEX cost table (from user data) + extrapolation for levels > 12
# Extrapolation: uses increment from last two known levels (10→12 = $879/2veh)
# This is conservative — actual negotiated cost could be lower at scale.
# ---------------------------------------------------------------------------
OPEX_TABLE: dict[int, int] = {2: 1463, 4: 2332, 6: 3271, 8: 3561, 10: 4891, 12: 5770}
_opex_increment = OPEX_TABLE[12] - OPEX_TABLE[10]  # 879 per 2 additional vehicles

# Installation (fixed) cost per capacity level — read directly from
# input_facilities.xlsx → cost_installation field (same values for all satellites).
# Levels beyond the Excel maximum plateau at the highest known cost.
_raw_ci = ast.literal_eval(fac_df.iloc[0]["cost_installation"])
_INSTALL_TABLE: dict[int, float] = {int(k): float(v) for k, v in _raw_ci.items()}
_INSTALL_PLATEAU: float = float(max(_INSTALL_TABLE.values()))  # cost at max known level


def install_cost(level: int) -> float:
    """Return installation (fixed opening) cost for a given capacity level."""
    return _INSTALL_TABLE.get(level, _INSTALL_PLATEAU)
for _lvl in range(14, 26, 2):
    OPEX_TABLE[_lvl] = OPEX_TABLE[_lvl - 2] + _opex_increment


def _opex(level: int) -> int:
    """OPEX for an exact capacity level (must be even and in OPEX_TABLE)."""
    return OPEX_TABLE.get(level, 0)


def _coverage(sat: str, level: int) -> float:
    """% of 500 scenarios where peak fleet ≤ level for this satellite."""
    data = peak_df[peak_df["satellite"] == sat]["fleet_ceil"].values
    return float((data <= level).sum()) / len(data) * 100


# Pre-compute coverage and OPEX for every level in both proposals
for col in ("levels_A", "levels_B"):
    cap_df[f"opex_{col}"] = cap_df.apply(
        lambda r: [_opex(lvl) for lvl in r[col]], axis=1
    )
    cap_df[f"coverage_{col}"] = cap_df.apply(
        lambda r: [round(_coverage(r["satellite"], lvl), 1) for lvl in r[col]], axis=1
    )

# Recommended level per proposal = first level with coverage ≥ 90%
def _rec_level(row, col):
    for lvl, cov in zip(row[col], row[f"coverage_{col}"]):
        if cov >= 90.0:
            return lvl
    return row[col][-1]  # fallback: max level

cap_df["rec_A"] = cap_df.apply(lambda r: _rec_level(r, "levels_A"), axis=1)
cap_df["rec_B"] = cap_df.apply(lambda r: _rec_level(r, "levels_B"), axis=1)
cap_df["opex_rec_A"] = cap_df["rec_A"].apply(_opex)
cap_df["opex_rec_B"] = cap_df["rec_B"].apply(_opex)
cap_df["opex_p50"] = cap_df["p50"].apply(lambda x: _opex(_rup2(x)))  # min to cover P50
cap_df["opex_max"] = cap_df["max_fleet"].apply(lambda x: _opex(_rup2(x)))

# "Cost of robustness" = extra monthly spend to cover P90 vs P50
cap_df["cost_of_robustness"] = cap_df["opex_rec_A"] - cap_df["opex_p50"]

total_A_p90  = cap_df["opex_rec_A"].sum()
total_B_p90  = cap_df["opex_rec_B"].sum()
total_p50    = cap_df["opex_p50"].sum()
total_max    = cap_df["opex_max"].sum()

print(f"\n=== Network OPEX comparison (monthly) ===")
print(f"  Costo mínimo (P50):   ${total_p50:,}")
print(f"  Propuesta A (P90):    ${total_A_p90:,}  (+${total_A_p90-total_p50:,} vs P50)")
print(f"  Propuesta B (P90):    ${total_B_p90:,}  (+${total_B_p90-total_p50:,} vs P50)")
print(f"  Cobertura máx:        ${total_max:,}  (+${total_max-total_p50:,} vs P50)")


# ---------------------------------------------------------------------------
# 5b. Pixel-level fleet: DC large trucks + satellite small vans
# ---------------------------------------------------------------------------
_PIX_CACHE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), f"pixel_fleet_cache_{N_SCENARIOS}.csv"
)
if os.path.exists(_PIX_CACHE):
    print(f"\nLoading pixel fleet data from cache...")
    pixel_fleet_df = pd.read_csv(_PIX_CACHE)
else:
    print(f"\nRunning CA for pixel-level fleet ({N_SCENARIOS} scenarios)...")
    pixel_fleet_df = compute_pixel_fleet_df(N_SCENARIOS)
    pixel_fleet_df.to_csv(_PIX_CACHE, index=False)
    print(f"  Pixel fleet cached to {_PIX_CACHE}")

# Sort each satellite's pixels by DC distance (ascending = closest to DC first)
sat_pixels_by_dc: dict[str, list] = {
    sat: sorted(
        list(assigned_by_sat.get(sat, set())),
        key=lambda j: dc_dist_dict.get(j, float("inf"))
    )
    for sat in satellites
}

# Pre-compute theta statistics
# theta_data[theta] = {"satellites": {sat: {dc_p90, sat_p90, n_dc, n_sat}},
#                      "total_dc_p90": int, "total_sat_p90": int}
print("\nComputing theta analysis...")
theta_data: dict[int, dict] = {}
for theta in THETAS:
    all_dc_pix: set    = set()
    all_sat_rows: list = []
    sat_stats: dict    = {}

    for sat in satellites:
        pxs  = sat_pixels_by_dc[sat]
        n_dc = int(len(pxs) * theta / 100)
        dc_pix  = set(pxs[:n_dc])
        sat_pix = set(pxs[n_dc:])
        all_dc_pix |= dc_pix

        # Satellite P90 — sum raw floats first, ceil after summing
        sf = pixel_fleet_df[
            (pixel_fleet_df["satellite"] == sat) &
            (pixel_fleet_df["pixel_id"].isin(sat_pix))
        ]
        if len(sf) > 0:
            sp = (sf.groupby(["scenario", "period"])["sat_fleet"]
                    .sum().reset_index()
                    .assign(sat_ceil=lambda d: d["sat_fleet"].apply(math.ceil))
                    .groupby("scenario")["sat_ceil"].max())
            sat_p90 = _p(sp.values, 90)
            all_sat_rows.append(sf[["scenario", "period", "sat_fleet"]])
        else:
            sat_p90 = 0

        # DC P90 for this satellite's DC pixels
        df_dc = pixel_fleet_df[pixel_fleet_df["pixel_id"].isin(dc_pix)]
        if len(df_dc) > 0:
            dp = (df_dc.groupby(["scenario", "period"])["dc_fleet"]
                       .sum().reset_index()
                       .assign(dc_ceil=lambda d: d["dc_fleet"].apply(math.ceil))
                       .groupby("scenario")["dc_ceil"].max())
            dc_p90_sat = _p(dp.values, 90)
        else:
            dc_p90_sat = 0

        sat_stats[sat] = {
            "dc_p90": dc_p90_sat, "sat_p90": sat_p90,
            "n_dc": len(dc_pix), "n_sat": len(sat_pix),
        }

    # Total DC fleet — all DC pixels together
    total_dc = pixel_fleet_df[pixel_fleet_df["pixel_id"].isin(all_dc_pix)]
    if len(total_dc) > 0:
        tdp = (total_dc.groupby(["scenario", "period"])["dc_fleet"]
                       .sum().reset_index()
                       .assign(dc_ceil=lambda d: d["dc_fleet"].apply(math.ceil))
                       .groupby("scenario")["dc_ceil"].max())
        total_dc_p90 = _p(tdp.values, 90)
    else:
        total_dc_p90 = 0

    # Total satellite fleet — P90 of sum over ALL sat pixels (not sum of P90s)
    if all_sat_rows:
        all_sat_df = pd.concat(all_sat_rows, ignore_index=True)
        tsp = (all_sat_df.groupby(["scenario", "period"])["sat_fleet"]
                         .sum().reset_index()
                         .assign(sat_ceil=lambda d: d["sat_fleet"].apply(math.ceil))
                         .groupby("scenario")["sat_ceil"].max())
        total_sat_p90 = _p(tsp.values, 90)
    else:
        total_sat_p90 = 0

    theta_data[theta] = {
        "satellites":    sat_stats,
        "total_dc_p90":  total_dc_p90,
        "total_sat_p90": total_sat_p90,
    }
    print(f"  θ={theta}%: DC large={total_dc_p90}  Sat small={total_sat_p90}")


# ---------------------------------------------------------------------------
# 6. Fleet figures
# ---------------------------------------------------------------------------
_chart_layout = dict(
    paper_bgcolor="#f4f6f8", plot_bgcolor="white",
    font=dict(family="sans-serif", color="#2c3e50", size=12),
    margin=dict(l=10, r=10, t=44, b=10),
)

# ── 6A. Box plots — distribution per satellite ──────────────────────────────
fig_box = go.Figure()
for sat in satellites:
    data_sat = fleet_df[fleet_df["satellite"] == sat]["fleet_ceil"].tolist()
    fig_box.add_trace(go.Box(
        y=data_sat,
        name=sat.replace("_", " ").title(),
        marker_color=sat_color[sat],
        line_color=sat_color[sat],
        boxpoints="outliers",
        marker_size=3,
        hovertemplate=(
            f"<b>{sat.replace('_',' ').title()}</b><br>"
            "Vehículos: %{y}<extra></extra>"
        ),
    ))

for lvl in CAPACITY_LEVELS:
    fig_box.add_hline(
        y=lvl, line_dash="dot", line_color="#aaaaaa", line_width=1,
        annotation_text=f" {lvl}", annotation_position="right",
        annotation_font=dict(size=10, color="#888888"),
    )

fig_box.update_layout(
    **_chart_layout,
    height=400,
    title=dict(text="Distribución de flota requerida por satélite", x=0.5, font=dict(size=13)),
    yaxis=dict(title="Vehículos (van)", gridcolor="#eeeeee", zeroline=False),
    xaxis=dict(title=""),
    showlegend=False,
)

# ── 6B. Heatmap — P90 fleet per satellite × period ─────────────────────────
# Order satellites by their annual P90 (descending)
sat_order = stats_df.sort_values("p90", ascending=False)["satellite"].tolist()

z_matrix, y_labels = [], []
for sat in sat_order:
    row_vals = []
    for t in range(1, 13):
        val = p90_period[(p90_period["satellite"] == sat) & (p90_period["period"] == t)]["p90"].values
        row_vals.append(int(val[0]) if len(val) > 0 else 0)
    z_matrix.append(row_vals)
    y_labels.append(sat.replace("_", " ").title())

fig_heatmap = go.Figure(go.Heatmap(
    z=z_matrix,
    x=MONTHS,
    y=y_labels,
    colorscale=[[0, "#f9f9f9"], [0.3, "#ffcdd2"], [0.7, "#e53935"], [1.0, "#7f0000"]],
    text=[[str(v) for v in row] for row in z_matrix],
    texttemplate="%{text}",
    textfont=dict(size=11),
    showscale=True,
    colorbar=dict(title="Vehículos", thickness=14, len=0.8),
    hovertemplate="<b>%{y}</b><br>%{x}: %{z} vehículos<extra></extra>",
))
_heatmap_layout = {**_chart_layout, "margin": dict(l=130, r=80, t=44, b=60)}
fig_heatmap.update_layout(
    **_heatmap_layout,
    height=420,
    title=dict(text="Flota P90 por satélite y período", x=0.5, font=dict(size=13)),
    xaxis=dict(
        side="bottom",
        tickmode="array",
        tickvals=list(range(12)),
        ticktext=MONTHS,
        tickfont=dict(size=11),
    ),
    yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
)

# ── 6C. Horizontal bar — P90 peak + recommended capacity ───────────────────
bar_df = stats_df.sort_values("p90", ascending=True)

fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(
    y=[s.replace("_", " ").title() for s in bar_df["satellite"]],
    x=bar_df["p90"],
    orientation="h",
    marker_color=[sat_color[s] for s in bar_df["satellite"]],
    text=bar_df["p90"].astype(str),
    textposition="outside",
    hovertemplate=(
        "<b>%{y}</b><br>P90 pico: %{x} vehículos<extra></extra>"
    ),
    showlegend=False,
))

for lvl in CAPACITY_LEVELS:
    fig_bar.add_vline(
        x=lvl, line_dash="dot", line_color="#888888", line_width=1.2,
        annotation_text=f" Cap.{lvl}", annotation_position="top",
        annotation_font=dict(size=9, color="#666666"),
    )

# Mark recommended capacity per satellite
for _, row in bar_df.iterrows():
    sat = row["satellite"]
    fig_bar.add_trace(go.Scatter(
        x=[row["recommended_cap"]],
        y=[sat.replace("_", " ").title()],
        mode="markers",
        marker=dict(symbol="diamond", size=10, color=sat_color[sat],
                    line=dict(color="white", width=1.5)),
        hovertemplate=(
            f"<b>{sat.replace('_',' ').title()}</b><br>"
            f"Cap. recomendada: {row['recommended_cap']}<extra></extra>"
        ),
        showlegend=False,
        name=f"_rec_{sat}",
    ))

fig_bar.update_layout(
    **_chart_layout,
    height=380,
    title=dict(text="P90 flota pico y capacidad recomendada (◆)", x=0.5, font=dict(size=13)),
    xaxis=dict(title="Vehículos (van)", gridcolor="#eeeeee", range=[0, CAPACITY_LEVELS[-1] + 1]),
    yaxis=dict(title=""),
    barmode="overlay",
)

# ── 6D. Summary table ───────────────────────────────────────────────────────
tbl_sats = stats_df.sort_values("p90", ascending=False)


def _hex_rgba(hex_color: str, alpha: float = 0.13) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


row_fill = [_hex_rgba(sat_color[s]) for s in tbl_sats["satellite"]]

fig_table = go.Figure(go.Table(
    columnwidth=[160, 80, 60, 60, 80, 60, 60, 130],
    header=dict(
        values=["<b>Satélite</b>", "<b>Píxeles</b>",
                "<b>P50</b>", "<b>P75</b>", "<b>P90</b>",
                "<b>P95</b>", "<b>Máx</b>", "<b>Cap. recomendada</b>"],
        fill_color="#2c3e50",
        font=dict(color="white", size=12),
        align="left",
        height=32,
    ),
    cells=dict(
        values=[
            [s.replace("_", " ").title() for s in tbl_sats["satellite"]],
            tbl_sats["n_pixels"].tolist(),
            tbl_sats["p50"].tolist(),
            tbl_sats["p75"].tolist(),
            tbl_sats["p90"].tolist(),
            tbl_sats["p95"].tolist(),
            tbl_sats["max_fleet"].astype(int).tolist(),
            tbl_sats["recommended_cap"].tolist(),
        ],
        fill_color=[row_fill] * 8,
        align="left",
        font=dict(size=12, color="#2c3e50"),
        height=28,
    ),
))
fig_table.update_layout(
    paper_bgcolor="#f4f6f8",
    margin=dict(l=0, r=0, t=0, b=0),
    height=len(tbl_sats) * 28 + 80,
)


# ---------------------------------------------------------------------------
# 6c. Capacity proposal figures
# ---------------------------------------------------------------------------
_cap_order = cap_df.sort_values("p90", ascending=False)["satellite"].tolist()

# ── Coverage color helper ─────────────────────────────────────────────────
def _cov_color(cov: float) -> str:
    if cov >= 90:  return "#27ae60"  # green  — P90+
    if cov >= 75:  return "#f39c12"  # orange — P75-P90
    if cov >= 50:  return "#e67e22"  # amber  — P50-P75
    return "#e74c3c"                 # red    — <P50

# ── Range chart: P10–P90 band + both proposals ────────────────────────────
fig_cap = go.Figure()

# P10–P90 fleet range per satellite (horizontal band)
for sat in _cap_order:
    r = cap_df[cap_df["satellite"] == sat].iloc[0]
    y = sat.replace("_", " ").title()
    fig_cap.add_trace(go.Scatter(
        x=[r["p10"], r["p90"]], y=[y, y],
        mode="lines",
        line=dict(color="rgba(150,150,150,0.35)", width=12),
        showlegend=False, hoverinfo="skip",
    ))
    # P50 tick
    fig_cap.add_trace(go.Scatter(
        x=[r["p50"]], y=[y],
        mode="markers",
        marker=dict(color="#555", size=10, symbol="line-ns-open", line_width=2),
        showlegend=False,
        hovertemplate=f"<b>{y}</b><br>P50 = {r['p50']}<extra></extra>",
    ))

# Proposal A levels (blue squares)
_Ax, _Ay, _Ahover = [], [], []
for sat in _cap_order:
    r = cap_df[cap_df["satellite"] == sat].iloc[0]
    y = sat.replace("_", " ").title()
    for lvl in r["levels_A"]:
        _Ax.append(lvl); _Ay.append(y)
        _Ahover.append(f"<b>{y}</b><br>Propuesta A: {lvl} veh.<extra></extra>")
fig_cap.add_trace(go.Scatter(
    x=_Ax, y=_Ay, mode="markers",
    marker=dict(color="#1565C0", size=13, symbol="square",
                line=dict(color="white", width=1)),
    name="Propuesta A",
    hovertemplate="%{customdata}",
    customdata=_Ahover,
))

# Proposal B levels (red diamonds)
_Bx, _By, _Bhover = [], [], []
for sat in _cap_order:
    r = cap_df[cap_df["satellite"] == sat].iloc[0]
    y = sat.replace("_", " ").title()
    for lvl in r["levels_B"]:
        _Bx.append(lvl); _By.append(y)
        _Bhover.append(f"<b>{y}</b><br>Propuesta B: {lvl} veh.<extra></extra>")
fig_cap.add_trace(go.Scatter(
    x=_Bx, y=_By, mode="markers",
    marker=dict(color="#c0392b", size=14, symbol="diamond",
                line=dict(color="white", width=1.2)),
    name="Propuesta B",
    hovertemplate="%{customdata}",
    customdata=_Bhover,
))

# P95 reference (dotted line per satellite would clutter; use scatter instead)
_P95x, _P95y = [], []
for sat in _cap_order:
    r = cap_df[cap_df["satellite"] == sat].iloc[0]
    _P95x.append(r["p95"]); _P95y.append(sat.replace("_", " ").title())
fig_cap.add_trace(go.Scatter(
    x=_P95x, y=_P95y, mode="markers",
    marker=dict(color="rgba(200,100,0,0.7)", size=8, symbol="triangle-right"),
    name="P95 demanda",
    hovertemplate="<b>%{y}</b><br>P95 = %{x}<extra></extra>",
))

fig_cap.update_layout(
    **{**_chart_layout, "margin": dict(l=130, r=20, t=44, b=40)},
    height=420,
    title=dict(text="Propuestas de niveles de capacidad por satélite", x=0.5, font=dict(size=13)),
    xaxis=dict(title="Vehículos", gridcolor="#eeeeee", zeroline=False, dtick=2),
    yaxis=dict(title=""),
    legend=dict(x=0.98, y=0.02, xanchor="right", yanchor="bottom"),
)

# ── Summary table: both proposals side by side ────────────────────────────
_cap_rows = cap_df.sort_values("p90", ascending=False)

def _fmt(levels): return " · ".join(str(x) for x in levels)

fig_cap_table = go.Figure(go.Table(
    columnwidth=[150, 55, 55, 55, 55, 180, 120],
    header=dict(
        values=["<b>Satélite</b>", "<b>P25</b>", "<b>P50</b>",
                "<b>P90</b>", "<b>P95</b>",
                "<b>Propuesta A</b><br><i>rango completo</i>",
                "<b>Propuesta B</b><br><i>3 niveles</i>"],
        fill_color="#2c3e50",
        font=dict(color="white", size=11),
        align="center", height=36,
    ),
    cells=dict(
        values=[
            [s.replace("_", " ").title() for s in _cap_rows["satellite"]],
            _cap_rows["p25"].tolist(),
            _cap_rows["p50"].tolist(),
            _cap_rows["p90"].tolist(),
            _cap_rows["p95"].tolist(),
            [_fmt(r["levels_A"]) for _, r in _cap_rows.iterrows()],
            [_fmt(r["levels_B"]) for _, r in _cap_rows.iterrows()],
        ],
        fill_color=[[_hex_rgba(sat_color[s]) for s in _cap_rows["satellite"]]] * 7,
        align=["left", "center", "center", "center", "center", "center", "center"],
        font=dict(size=11, color="#2c3e50"),
        height=28,
    ),
))
fig_cap_table.update_layout(
    paper_bgcolor="#f4f6f8",
    margin=dict(l=0, r=0, t=0, b=0),
    height=len(_cap_rows) * 28 + 90,
)


# ── 6c-cost-1. Dot chart: OPEX vs Coverage per level (both proposals) ─────
# Each dot = one capacity level for one satellite.
# X = monthly OPEX, Y = satellite, color = coverage tier, shape = proposal.
fig_cost_dots = go.Figure()

_added_legend: set = set()
for sat in _cap_order:
    r = cap_df[cap_df["satellite"] == sat].iloc[0]
    y = sat.replace("_", " ").title()

    # Proposal A dots (squares)
    for lvl, opex_val, cov in zip(r["levels_A"], r["opex_levels_A"], r["coverage_levels_A"]):
        c = _cov_color(cov)
        leg_key = f"A_{c}"
        show_leg = leg_key not in _added_legend
        _added_legend.add(leg_key)
        fig_cost_dots.add_trace(go.Scatter(
            x=[opex_val], y=[y], mode="markers",
            marker=dict(symbol="square", size=11, color=c, line=dict(color="white", width=1)),
            name=f"■ A — cob. {'>90%' if c=='#27ae60' else '75-90%' if c=='#f39c12' else '50-75%' if c=='#e67e22' else '<50%'}",
            showlegend=show_leg, legendgroup=leg_key,
            hovertemplate=f"<b>{y}</b> · Prop.A<br>Capacidad: {lvl} veh.<br>OPEX: ${opex_val:,}/mes<br>Cobertura: {cov}%<extra></extra>",
        ))

    # Proposal B dots (diamonds)
    for lvl, opex_val, cov in zip(r["levels_B"], r["opex_levels_B"], r["coverage_levels_B"]):
        c = _cov_color(cov)
        leg_key = f"B_{c}"
        show_leg = leg_key not in _added_legend
        _added_legend.add(leg_key)
        fig_cost_dots.add_trace(go.Scatter(
            x=[opex_val], y=[y], mode="markers",
            marker=dict(symbol="diamond", size=13, color=c, line=dict(color="white", width=1.2)),
            name=f"◆ B — cob. {'>90%' if c=='#27ae60' else '75-90%' if c=='#f39c12' else '50-75%' if c=='#e67e22' else '<50%'}",
            showlegend=show_leg, legendgroup=leg_key,
            hovertemplate=f"<b>{y}</b> · Prop.B<br>Capacidad: {lvl} veh.<br>OPEX: ${opex_val:,}/mes<br>Cobertura: {cov}%<extra></extra>",
        ))

fig_cost_dots.update_layout(
    **{**_chart_layout, "margin": dict(l=140, r=20, t=44, b=50)},
    height=430,
    title=dict(text="OPEX mensual vs cobertura de escenarios — cada punto es un nivel de capacidad",
               x=0.5, font=dict(size=13)),
    xaxis=dict(title="OPEX mensual ($)", gridcolor="#eeeeee", zeroline=False,
               tickformat="$,.0f"),
    yaxis=dict(title=""),
    legend=dict(x=1.01, y=1, xanchor="left", font=dict(size=10)),
)

# ── 6c-cost-2. Stacked bar: total network OPEX per scenario (P50 / P90 / max) ──
_scen_labels = ["P50 (mínimo viable)", "P90 — Prop.A", "P90 — Prop.B", "Máximo histórico"]
_scen_totals = [total_p50, total_A_p90, total_B_p90, total_max]
_scen_colors = ["#f39c12", "#1565C0", "#c0392b", "#555555"]
_scen_pct    = [f"${v:,}<br>({v/total_p50*100:.0f}% vs P50)" for v in _scen_totals]

fig_cost_total = go.Figure()
for sat in _cap_order[::-1]:
    r = cap_df[cap_df["satellite"] == sat].iloc[0]
    y_sat = sat.replace("_", " ").title()
    _vals = [_opex(_rup2(r["p50"])), r["opex_rec_A"], r["opex_rec_B"], _opex(_rup2(r["max_fleet"]))]
    for i, (lbl, val, col) in enumerate(zip(_scen_labels, _vals, _scen_colors)):
        fig_cost_total.add_trace(go.Bar(
            name=lbl, x=[lbl], y=[val],
            marker_color=col, opacity=0.85,
            showlegend=(sat == _cap_order[-1]),
            legendgroup=lbl,
            hovertemplate=f"<b>{y_sat}</b><br>{lbl}<br>${val:,}/mes<extra></extra>",
            text=y_sat if val > 200 else "",
            textposition="inside", textfont=dict(size=9, color="white"),
        ))

fig_cost_total.update_layout(
    **_chart_layout,
    height=420,
    barmode="stack",
    title=dict(text="OPEX mensual total de la red por escenario de demanda", x=0.5, font=dict(size=13)),
    yaxis=dict(title="OPEX mensual total ($)", gridcolor="#eeeeee", tickformat="$,.0f"),
    xaxis=dict(title=""),
    legend=dict(x=0.98, y=0.99, xanchor="right"),
    annotations=[dict(
        x=lbl, y=total + 200, text=f"<b>${total:,.0f}</b>",
        showarrow=False, font=dict(size=11),
        xref="x", yref="y",
    ) for lbl, total in zip(_scen_labels, _scen_totals)],
)

# ── 6c-cost-3. Robustness table: per satellite, cost breakdown ────────────
_rob_rows = cap_df.sort_values("p90", ascending=False).copy()

# ---------------------------------------------------------------------------
# Read actual capacity and cost_installation per satellite from Excel
# ---------------------------------------------------------------------------
import ast as _ast

_fac_raw = pd.read_excel(os.path.join(DATA, "facilities/input_facilities.xlsx"), engine="openpyxl")
_fac_raw["id_facility"] = _fac_raw["id_facility"].str.upper()

_excel_cap: dict[str, dict[int, int]]   = {}   # sat -> {level_int: vehicles}
_excel_ci:  dict[str, dict[int, float]] = {}   # sat -> {level_int: install_cost}

for _, _fr in _fac_raw.iterrows():
    _sid = _fr["id_facility"]
    _cap_raw = _ast.literal_eval(_fr["capacity"])
    _ci_raw  = _ast.literal_eval(_fr["cost_installation"])
    # keys are strings in the Excel; convert to int
    _excel_cap[_sid] = {int(k): int(v)   for k, v in _cap_raw.items()}
    _excel_ci[_sid]  = {int(k): float(v) for k, v in _ci_raw.items()}


def _coverage_at(sat: str, level: int) -> float:
    """% of 500 scenarios where peak fleet ≤ level."""
    data = peak_df[peak_df["satellite"] == sat]["fleet_ceil"].values
    return float((data <= level).sum()) / len(data) * 100 if len(data) > 0 else 0.0


def _fmt_level_cost(levels, costs, coverages, cost_label="$"):
    parts = []
    for lvl, cost_val, cov in zip(levels, costs, coverages):
        marker = "✓" if cov >= 90 else "~" if cov >= 75 else "✗"
        parts.append(f"{marker} {lvl}v → {cost_label}{cost_val:,.0f} ({cov:.0f}%)")
    return "<br>".join(parts)


def _gap_pct(new_val: float, base_val: float) -> str:
    if base_val == 0:
        return "—"
    pct = (new_val - base_val) / base_val * 100
    sign = "▼" if pct < 0 else ("▲" if pct > 0 else "=")
    return f"{sign} {abs(pct):.1f}%"


# Enrich _rob_rows with actual (Excel) capacity info per satellite
# "Actual recommended" = smallest available level in Excel capacity ≥ P90
# (same logic as recommended_cap, but using the Excel levels explicitly)
def _excel_rec_level(sat: str, p90: int) -> int:
    levels = sorted(k for k in _excel_cap.get(sat, {}) if k > 0)  # exclude level 0
    for lvl in levels:
        if lvl >= p90:
            return lvl
    return levels[-1] if levels else 12   # fallback to max if nothing covers P90


_rob_rows["excel_avail_levels"]    = _rob_rows["satellite"].apply(
    lambda s: sorted(k for k in _excel_cap.get(s, {}) if k > 0)
)
_rob_rows["excel_rec_level"]       = _rob_rows.apply(
    lambda r: _excel_rec_level(r["satellite"], r["p90"]), axis=1
)
_rob_rows["excel_install_rec"]     = _rob_rows.apply(
    lambda r: _excel_ci.get(r["satellite"], {}).get(r["excel_rec_level"], 45417), axis=1
)
_rob_rows["excel_coverage_rec"]    = _rob_rows.apply(
    lambda r: round(_coverage_at(r["satellite"], r["excel_rec_level"]), 1), axis=1
)
# Install cost at proposals' recommended levels (plateau for >12)
_rob_rows["install_rec_A"]     = _rob_rows["rec_A"].apply(install_cost)
_rob_rows["install_rec_B"]     = _rob_rows["rec_B"].apply(install_cost)
# Coverage at proposed recommended levels
_rob_rows["cov_rec_A"]         = _rob_rows.apply(
    lambda r: round(_coverage_at(r["satellite"], r["rec_A"]), 1), axis=1
)
_rob_rows["cov_rec_B"]         = _rob_rows.apply(
    lambda r: round(_coverage_at(r["satellite"], r["rec_B"]), 1), axis=1
)
# Coverage of each level in proposals (for the multi-line cell)
_rob_rows["cov_levels_A"]      = _rob_rows.apply(
    lambda r: [round(_coverage_at(r["satellite"], lvl), 1) for lvl in r["levels_A"]], axis=1
)
_rob_rows["cov_levels_B"]      = _rob_rows.apply(
    lambda r: [round(_coverage_at(r["satellite"], lvl), 1) for lvl in r["levels_B"]], axis=1
)
# Install costs per level in proposals
_rob_rows["install_levels_A"]  = _rob_rows["levels_A"].apply(lambda lvls: [install_cost(l) for l in lvls])
_rob_rows["install_levels_B"]  = _rob_rows["levels_B"].apply(lambda lvls: [install_cost(l) for l in lvls])
# GAP% costo fijo: proposals vs current Excel recommended
_rob_rows["gap_install_A"]     = _rob_rows.apply(
    lambda r: _gap_pct(r["install_rec_A"], r["excel_install_rec"]), axis=1
)
_rob_rows["gap_install_B"]     = _rob_rows.apply(
    lambda r: _gap_pct(r["install_rec_B"], r["excel_install_rec"]), axis=1
)
# GAP cobertura (pp): proposals vs current Excel recommended coverage
def _gap_cov(new_cov: float, base_cov: float) -> str:
    diff = new_cov - base_cov
    if abs(diff) < 0.5:
        return "= 0 pp"
    sign = "▲" if diff > 0 else "▼"
    return f"{sign} {abs(diff):.1f} pp"

_rob_rows["gap_cov_A"]         = _rob_rows.apply(
    lambda r: _gap_cov(r["cov_rec_A"], r["excel_coverage_rec"]), axis=1
)
_rob_rows["gap_cov_B"]         = _rob_rows.apply(
    lambda r: _gap_cov(r["cov_rec_B"], r["excel_coverage_rec"]), axis=1
)


def _levels_costs_cell(levels, costs, coverages, rec_level) -> str:
    """Format levels + costs vertically; mark recommended with arrow."""
    parts = []
    for lvl, cost, cov in zip(levels, costs, coverages):
        rec_mark = " ◄" if lvl == rec_level else ""
        cov_str  = f"{cov:.0f}%"
        parts.append(f"{lvl}v → ${cost:,.0f} ({cov_str}){rec_mark}")
    return "<br>".join(parts)


def _actual_cell(row) -> str:
    levels = row["excel_avail_levels"]
    costs  = [_INSTALL_TABLE.get(l, _INSTALL_PLATEAU) for l in levels]
    covs   = [round(_coverage_at(row["satellite"], l), 1) for l in levels]
    rec    = row["excel_rec_level"]
    return _levels_costs_cell(levels, costs, covs, rec)


def _prop_cell(row, suffix) -> str:
    levels = row[f"levels_{suffix}"]
    costs  = row[f"install_levels_{suffix}"]
    covs   = row[f"cov_levels_{suffix}"]
    rec    = row[f"rec_{suffix}"]
    return _levels_costs_cell(levels, costs, covs, rec)


# Row height: 6 Excel levels at ~14px each + padding
_ROW_H = max(6, max(_rob_rows["excel_avail_levels"].apply(len))) * 15 + 10

fig_cost_rob_table = go.Figure(go.Table(
    columnwidth=[120, 40, 40, 200, 170, 140],
    header=dict(
        values=[
            "<b>Satélite</b>",
            "<b>P50<br>flota</b>",
            "<b>P90<br>flota</b>",
            "<b>Actual (Excel)</b><br>"
            "<i>nivel · costo fijo · cobertura</i><br>"
            "<i>◄ = nivel que cubre P90</i>",
            "<b>Propuesta A</b><br>"
            "<i>nivel · costo fijo · cobertura</i><br>"
            "<i>◄ = nivel recomendado</i>",
            "<b>Propuesta B</b><br>"
            "<i>nivel · costo fijo · cobertura</i><br>"
            "<i>◄ = nivel recomendado</i>",
        ],
        fill_color="#2c3e50", font=dict(color="white", size=11),
        align="center", height=52,
    ),
    cells=dict(
        values=[
            [s.replace("_", " ").title() for s in _rob_rows["satellite"]],
            _rob_rows["p50"].tolist(),
            _rob_rows["p90"].tolist(),
            [_actual_cell(r) for _, r in _rob_rows.iterrows()],
            [_prop_cell(r, "A") for _, r in _rob_rows.iterrows()],
            [_prop_cell(r, "B") for _, r in _rob_rows.iterrows()],
        ],
        fill_color=[[_hex_rgba(sat_color[s]) for s in _rob_rows["satellite"]]] * 6,
        align=["left", "center", "center", "left", "left", "left"],
        font=dict(size=11, color="#2c3e50"),
        height=_ROW_H,
    ),
))
fig_cost_rob_table.update_layout(
    paper_bgcolor="#f4f6f8",
    margin=dict(l=0, r=0, t=0, b=0),
    height=len(_rob_rows) * _ROW_H + 120,
)


# ---------------------------------------------------------------------------
# 6b. Theta (DC split) figures
# ---------------------------------------------------------------------------

# ── 6b-1. Line chart: total DC + total Sat fleet vs theta ──────────────────
fig_theta_line = go.Figure()
fig_theta_line.add_trace(go.Scatter(
    x=THETAS,
    y=[theta_data[t]["total_dc_p90"] for t in THETAS],
    mode="lines+markers",
    name="DC — camiones grandes (P90)",
    line=dict(color="#2c3e50", width=2.5),
    marker=dict(size=8, symbol="circle"),
    hovertemplate="θ=%{x}%<br>DC grandes: %{y}<extra></extra>",
))
fig_theta_line.add_trace(go.Scatter(
    x=THETAS,
    y=[theta_data[t]["total_sat_p90"] for t in THETAS],
    mode="lines+markers",
    name="Satélites — vans pequeñas (P90 total)",
    line=dict(color="#e74c3c", width=2.5, dash="dot"),
    marker=dict(size=8, symbol="diamond"),
    hovertemplate="θ=%{x}%<br>Sat vans: %{y}<extra></extra>",
))
fig_theta_line.update_layout(
    **_chart_layout,
    height=360,
    title=dict(text="Flota total P90 vs % demanda atendida por DC", x=0.5, font=dict(size=13)),
    xaxis=dict(
        title="% píxeles más cercanos al DC atendidos por él",
        tickvals=THETAS, ticktext=[f"{t}%" for t in THETAS],
        gridcolor="#eeeeee",
    ),
    yaxis=dict(title="Vehículos P90 (peak period)", gridcolor="#eeeeee", zeroline=False),
    legend=dict(x=0.5, y=1.12, xanchor="center", orientation="h"),
)

# ── 6b-2. Geographic map: satellite colors preserved + hatch overlay for DC ─

def _dc_pids_at_theta(theta: int) -> set[int]:
    """Pixel integers served by DC at this theta (closest to DC per satellite)."""
    dc_set: set[int] = set()
    for sat in satellites:
        pxs  = sat_pixels_by_dc[sat]
        n_dc = int(len(pxs) * theta / 100)
        for j in pxs[:n_dc]:
            dc_set.add(int(j.split("-")[1]))
    return dc_set


def _hatch_lines(ring: list, spacing: float = 0.004) -> tuple[list, list]:
    """45° diagonal hatch lines clipped to polygon bounding box."""
    lons_r = [c[0] for c in ring[:-1]]
    lats_r = [c[1] for c in ring[:-1]]
    lon_min, lon_max = min(lons_r), max(lons_r)
    lat_min, lat_max = min(lats_r), max(lats_r)

    h_lons: list = []
    h_lats: list = []
    c = lat_min - lon_max          # first diagonal touching bottom-right corner
    c_end = lat_max - lon_min      # last diagonal touching top-left corner

    while c <= c_end:
        x0, y0 = lon_min, lon_min + c
        x1, y1 = lon_max, lon_max + c

        if y0 < lat_min:           # clip start to bottom edge
            x0, y0 = lat_min - c, lat_min
        elif y0 > lat_max:
            c += spacing
            continue

        if y1 > lat_max:           # clip end to top edge
            x1, y1 = lat_max - c, lat_max
        elif y1 < lat_min:
            c += spacing
            continue

        if abs(x1 - x0) > 1e-10 or abs(y1 - y0) > 1e-10:
            h_lons += [x0, x1, None]
            h_lats += [y0, y1, None]
        c += spacing

    return h_lons, h_lats


# All demand pixel integers (union of both layers)
_demand_pids_int = sorted({int(j.split("-")[1]) for pxs in assigned_by_sat.values() for j in pxs})
_no_demand_pids  = [p for p in all_pids if p not in set(_demand_pids_int)]

fig_theta_map = go.Figure()

# ── Static traces (always visible) ────────────────────────────────────────
# Background grid
lons_bg, lats_bg = poly_coords(_no_demand_pids)
fig_theta_map.add_trace(go.Scattermap(
    lon=lons_bg, lat=lats_bg, mode="lines", fill="toself",
    fillcolor="rgba(180,180,180,0.08)", line=dict(color="#bbbbbb", width=0.5),
    hoverinfo="skip", showlegend=False, name="_bg",
))

# Satellite polygon fills — all demand pixels, always colored (don't change with theta)
for sat in satellites:
    sat_pids_int = sorted({int(j.split("-")[1]) for j in assigned_by_sat.get(sat, set())})
    lons_s, lats_s = poly_coords(sat_pids_int)
    fig_theta_map.add_trace(go.Scattermap(
        lon=lons_s, lat=lats_s, mode="lines", fill="toself",
        fillcolor=_hex_rgba(sat_color[sat], OPACITY_FILL),
        line=dict(color=sat_color[sat], width=0.8),
        name=sat.replace("_", " ").title(),
        showlegend=True, hoverinfo="skip",
    ))

_n_static = 1 + len(satellites)  # bg + 9 satellite polygon traces

# ── Hatch overlay traces (one per theta, toggled by dropdown) ─────────────
for i_t, theta in enumerate(THETAS):
    visible = (theta == 0)
    dc_pids_theta = _dc_pids_at_theta(theta)

    all_h_lons: list = []
    all_h_lats: list = []
    for pid in sorted(dc_pids_theta):
        if pid in pixel_coords:
            hl, hla = _hatch_lines(pixel_coords[pid])
            all_h_lons += hl
            all_h_lats += hla

    fig_theta_map.add_trace(go.Scattermap(
        lon=all_h_lons, lat=all_h_lats,
        mode="lines",
        line=dict(color="#1a1a2e", width=1.3),
        opacity=0.80,
        visible=visible,
        showlegend=(theta == 0),
        name="DC directo 🚛",
        hoverinfo="skip",
    ))

_n_hatch = len(THETAS)        # one hatch trace per theta

# ── Markers (always visible) ──────────────────────────────────────────────
for sat in satellites:
    row_s = fac_df[fac_df["id_facility"] == sat].iloc[0]
    fig_theta_map.add_trace(_sat_marker(sat, row_s["lon"], row_s["lat"], sat_color[sat]))
    fig_theta_map.add_trace(_sat_marker_dot(sat, row_s["lon"], row_s["lat"], sat_color[sat]))
for t in _dc_markers():
    fig_theta_map.add_trace(t)

_n_markers = len(satellites) * 2 + 2  # sat rings + sat dots + dc ring + dc dot

# ── Dropdown: toggle only the hatch traces ───────────────────────────────
_n_total = len(fig_theta_map.data)
_theta_buttons = []
for i_t, theta in enumerate(THETAS):
    vis = (
        [True] * _n_static +                     # static polygons
        [i2 == i_t for i2 in range(_n_hatch)] +  # only this theta's hatch
        [True] * _n_markers                       # markers
    )
    _theta_buttons.append(dict(
        label=f"θ = {theta}%",
        method="update",
        args=[{"visible": vis},
              {"title": {"text": f"Asignación por satélite — θ={theta}% atendido por DC (rayado = DC directo)",
                         "x": 0.5, "font": {"size": 13}}}],
    ))

_theta_map_layout = {k: v for k, v in fig_layout.items() if k not in ("height", "legend")}
fig_theta_map.update_layout(
    **_theta_map_layout,
    height=580,
    title=dict(text="Asignación por satélite — θ=0% al DC (rayado = DC directo)", x=0.5, font=dict(size=13)),
    map=map_layout,
    updatemenus=[dict(
        buttons=_theta_buttons,
        direction="down",
        showactive=True,
        x=0.01, xanchor="left",
        y=1.07, yanchor="top",
        bgcolor="white", bordercolor="#aaa",
        font=dict(size=11),
    )],
    legend=dict(bgcolor="rgba(255,255,255,0.88)", bordercolor="#cccccc",
                borderwidth=1, font=dict(size=10), x=0.01, y=0.99,
                xanchor="left", yanchor="top"),
)

# ── 6b-3. Summary table: per satellite at key theta values ─────────────────
_key_thetas = [0, 30, 60, 90]
_tbl_sats_theta = sorted(satellites, key=lambda s: theta_data[0]["satellites"][s]["sat_p90"], reverse=True)

_hdr = ["<b>Satélite</b>", "<b>Píxeles</b>"]
for th in _key_thetas:
    _hdr += [f"<b>θ={th}%<br>DC 🚛</b>", f"<b>θ={th}%<br>Sat 🚐</b>"]

_cell_vals = [[s.replace("_", " ").title() for s in _tbl_sats_theta]]
_cell_vals.append([len(sat_pixels_by_dc[s]) for s in _tbl_sats_theta])
for th in _key_thetas:
    sd = theta_data[th]["satellites"]
    _cell_vals.append([sd[s]["dc_p90"]  for s in _tbl_sats_theta])
    _cell_vals.append([sd[s]["sat_p90"] for s in _tbl_sats_theta])

_row_fill_theta = [_hex_rgba(sat_color[s]) for s in _tbl_sats_theta]

fig_theta_table = go.Figure(go.Table(
    columnwidth=[160, 70] + [60, 60] * len(_key_thetas),
    header=dict(
        values=_hdr,
        fill_color="#2c3e50",
        font=dict(color="white", size=11),
        align="center", height=36,
    ),
    cells=dict(
        values=_cell_vals,
        fill_color=[_row_fill_theta] * len(_cell_vals),
        align="center",
        font=dict(size=11, color="#2c3e50"),
        height=28,
    ),
))
fig_theta_table.update_layout(
    paper_bgcolor="#f4f6f8",
    margin=dict(l=0, r=0, t=0, b=0),
    height=len(_tbl_sats_theta) * 28 + 90,
)


# ---------------------------------------------------------------------------
# 6d. Operational cost proposal based on scenarios
# ---------------------------------------------------------------------------
# Methodology:
#   base_opex(q)      = monthly total OPEX from the user's tariff table (OPEX_TABLE)
#   seasonal_factor[sat][t] = relative demand in month t vs annual average
#                           derived from 500-scenario fleet analysis
#   cost_operation[q][t] = base_opex(q) × seasonal_factor[sat][t]
#
# Justification: operational costs (staff hours, fuel, vehicle wear) scale with
# actual utilization. OPEX_TABLE gives the cost at average utilization; the
# seasonal factor adjusts for months with higher/lower demand.
# Fixed fraction (α=0.70) keeps most costs stable; variable fraction (1-α=0.30)
# follows demand seasonality.

ALPHA_FIXED = 0.70   # fraction of OPEX that is fixed (rent, base staff, insurance)

# 1. Real seasonal factors from raw historical data (raw_pixels_with_drop_demand.csv)
#    Scenarios have nearly flat seasonality because Beta params don't vary enough between
#    periods. The raw data shows the true demand pattern (Dec=+42%, Feb=-30%).
_raw_demand = pd.read_csv(os.path.join(DATA, "pixels/raw_pixels_with_drop_demand.csv"))

_layer_factors: dict[str, list[float]] = {}
for _layer in ["A", "B"]:
    _mf = _raw_demand[_raw_demand["layer"] == _layer].groupby("month")["demand"].sum()
    _mf = _mf.reindex(range(1, 13), fill_value=_mf.mean())
    _layer_factors[_layer] = (_mf / _mf.mean()).round(4).tolist()

# Per-satellite weighted factors: weighted by layer A/B pixel mix for that satellite
_sat_seasonal: dict[str, list[float]] = {}
for _sat in satellites:
    _pix_ids = list(assigned_by_sat.get(_sat, set()))
    _n_A = sum(1 for j in _pix_ids if j.startswith("A-"))
    _n_B = sum(1 for j in _pix_ids if j.startswith("B-"))
    _total = _n_A + _n_B
    if _total == 0:
        _sat_seasonal[_sat] = [1.0] * 12
        continue
    _w_A, _w_B = _n_A / _total, _n_B / _total
    _sat_seasonal[_sat] = [
        round(_w_A * _layer_factors["A"][t] + _w_B * _layer_factors["B"][t], 4)
        for t in range(12)
    ]

print("\n=== Real seasonal factors per satellite (from historical data) ===")
for _sat in sorted(_sat_seasonal):
    _f = _sat_seasonal[_sat]
    print(f"  {_sat:20s}: min={min(_f):.3f} (mes {MONTHS[_f.index(min(_f))]}) "
          f"max={max(_f):.3f} (mes {MONTHS[_f.index(max(_f))]}) "
          f"var={(max(_f)-min(_f))/1.0*100:.1f}pp")

# 2. Proposed cost_operation[sat][q][t] for each satellite and Proposal-A level
proposed_opex: dict[str, dict[int, list[float]]] = {}
for _sat in satellites:
    proposed_opex[_sat] = {}
    _levels = cap_df[cap_df["satellite"] == _sat]["levels_A"].iloc[0]
    for _q in _levels:
        _base = OPEX_TABLE.get(_q, OPEX_TABLE[12] + (_q - 12) // 2 * _opex_increment)
        proposed_opex[_sat][_q] = [
            round(_base * (ALPHA_FIXED + (1 - ALPHA_FIXED) * _sat_seasonal[_sat][t]), 2)
            for t in range(12)
        ]

# Summary stats per satellite at recommended capacity
_opex_summary: list[dict] = []
for _sat in satellites:
    _rec = int(cap_df[cap_df["satellite"] == _sat]["rec_A"].iloc[0])
    _monthly = proposed_opex[_sat].get(_rec, [0.0] * 12)
    _opex_summary.append({
        "satellite":  _sat,
        "rec_level":  _rec,
        "monthly":    _monthly,
        "avg_monthly": round(sum(_monthly) / 12, 0),
        "annual":     round(sum(_monthly), 0),
        "peak_month": MONTHS[_monthly.index(max(_monthly))],
        "low_month":  MONTHS[_monthly.index(min(_monthly))],
        "variation":  round((max(_monthly) - min(_monthly)) / (sum(_monthly)/12) * 100, 1),
    })

print("\n=== Proposed operational costs at recommended capacity ===")
for r in sorted(_opex_summary, key=lambda x: -x["annual"]):
    print(f"  {r['satellite']:20s}  q={r['rec_level']:2d}v  "
          f"avg=${r['avg_monthly']:,.0f}/mes  annual=${r['annual']:,.0f}  "
          f"peak={r['peak_month']}  var={r['variation']:.1f}%")

# ── 6d-1. Heatmap: proposed monthly OPEX at recommended level ──────────────
_heat_z, _heat_text, _heat_y = [], [], []
for r in sorted(_opex_summary, key=lambda x: -x["annual"]):
    _heat_z.append(r["monthly"])
    _heat_text.append([f"${v:,.0f}" for v in r["monthly"]])
    _heat_y.append(r["satellite"].replace("_", " ").title())

fig_opex_heatmap = go.Figure(go.Heatmap(
    z=_heat_z, x=MONTHS, y=_heat_y,
    text=_heat_text, texttemplate="%{text}", textfont=dict(size=10),
    colorscale=[[0, "#f0f9ff"], [0.5, "#3498db"], [1.0, "#1a5276"]],
    colorbar=dict(title="$/mes", thickness=14, len=0.8),
    hovertemplate="<b>%{y}</b><br>%{x}: %{text}<extra></extra>",
))
fig_opex_heatmap.update_layout(
    **{k: v for k, v in _chart_layout.items() if k != "margin"},
    margin=dict(l=140, r=80, t=44, b=60),
    height=420,
    title=dict(text="Costo operacional mensual propuesto — nivel recomendado P90", x=0.5, font=dict(size=13)),
    xaxis=dict(tickfont=dict(size=11)),
    yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
)

# ── 6d-2. Line chart: monthly OPEX profile per satellite ──────────────────
fig_opex_lines = go.Figure()
for r in sorted(_opex_summary, key=lambda x: -x["annual"]):
    _sat = r["satellite"]
    fig_opex_lines.add_trace(go.Scatter(
        x=MONTHS, y=r["monthly"],
        mode="lines+markers",
        name=f"{_sat.replace('_',' ').title()} ({r['rec_level']}v)",
        line=dict(color=sat_color[_sat], width=2),
        marker=dict(size=6),
        hovertemplate=f"<b>{_sat.replace('_',' ').title()}</b><br>%{{x}}: $%{{y:,.0f}}<extra></extra>",
    ))
fig_opex_lines.update_layout(
    **{k: v for k, v in _chart_layout.items() if k != "margin"},
    height=400,
    margin=dict(l=80, r=170, t=44, b=40),
    title=dict(text="Perfil mensual de costo operacional — nivel recomendado por satélite", x=0.5, font=dict(size=13)),
    yaxis=dict(title="$/mes", gridcolor="#eeeeee", tickformat="$,.0f"),
    xaxis=dict(title=""),
    legend=dict(x=1.01, y=1, xanchor="left", font=dict(size=10)),
)

# ── 6d-3. Comparison: proposed vs current cost_operation ──────────────────
_fac_raw_co = pd.read_excel(os.path.join(DATA, "facilities/input_facilities.xlsx"), engine="openpyxl")
_fac_raw_co["id_facility"] = _fac_raw_co["id_facility"].str.upper()

_current_annual: dict[str, float] = {}
for _, _fr in _fac_raw_co.iterrows():
    _co = ast.literal_eval(_fr["cost_operation"])
    _rec_str = str(cap_df[cap_df["satellite"] == _fr["id_facility"]]["rec_A"].iloc[0]) \
               if _fr["id_facility"] in satellites else "12"
    _monthly_curr = _co.get(_rec_str, _co.get("12", [0]*12))
    _current_annual[_fr["id_facility"]] = sum(_monthly_curr)

fig_opex_compare = go.Figure()
_bar_sats = [r["satellite"] for r in sorted(_opex_summary, key=lambda x: -x["annual"])]
_bar_labels = [s.replace("_", " ").title() for s in _bar_sats]

fig_opex_compare.add_trace(go.Bar(
    name="Actual Excel",
    x=_bar_labels,
    y=[_current_annual.get(s, 0) for s in _bar_sats],
    marker_color="#aaaaaa", opacity=0.8,
    hovertemplate="<b>%{x}</b><br>Actual: $%{y:,.0f}/año<extra></extra>",
))
fig_opex_compare.add_trace(go.Bar(
    name="Propuesto (OPEX tabla + estacionalidad)",
    x=_bar_labels,
    y=[r["annual"] for r in sorted(_opex_summary, key=lambda x: -x["annual"])],
    marker_color=[sat_color[s] for s in _bar_sats], opacity=0.85,
    hovertemplate="<b>%{x}</b><br>Propuesto: $%{y:,.0f}/año<extra></extra>",
))
fig_opex_compare.update_layout(
    **_chart_layout,
    height=380,
    barmode="group",
    title=dict(text="Costo operacional anual: actual en Excel vs propuesto", x=0.5, font=dict(size=13)),
    yaxis=dict(title="$/año", gridcolor="#eeeeee", tickformat="$,.0f"),
    xaxis=dict(title=""),
    legend=dict(x=0.98, y=0.99, xanchor="right"),
)

# ── 6d-4. Summary table with proposed cost_operation values ───────────────
def _fmt_opex_list(monthly: list[float]) -> str:
    return "[" + ", ".join(f"{v:,.0f}" for v in monthly) + "]"

fig_opex_table = go.Figure(go.Table(
    columnwidth=[130, 55, 75, 75, 65, 65, 310],
    header=dict(
        values=["<b>Satélite</b>", "<b>Nivel<br>rec.</b>",
                "<b>OPEX base<br>$/mes (avg)</b>", "<b>Costo anual<br>propuesto</b>",
                "<b>Mes<br>pico</b>", "<b>Var.<br>estac.</b>",
                "<b>cost_operation[q] — lista 12 meses ($)</b>"],
        fill_color="#2c3e50", font=dict(color="white", size=10),
        align="center", height=38,
    ),
    cells=dict(
        values=[
            [r["satellite"].replace("_", " ").title() for r in sorted(_opex_summary, key=lambda x: -x["annual"])],
            [f"{r['rec_level']}v" for r in sorted(_opex_summary, key=lambda x: -x["annual"])],
            [f"${r['avg_monthly']:,.0f}" for r in sorted(_opex_summary, key=lambda x: -x["annual"])],
            [f"${r['annual']:,.0f}" for r in sorted(_opex_summary, key=lambda x: -x["annual"])],
            [r["peak_month"] for r in sorted(_opex_summary, key=lambda x: -x["annual"])],
            [f"{r['variation']:.1f}%" for r in sorted(_opex_summary, key=lambda x: -x["annual"])],
            [_fmt_opex_list(r["monthly"]) for r in sorted(_opex_summary, key=lambda x: -x["annual"])],
        ],
        fill_color=[[_hex_rgba(sat_color[r["satellite"]]) for r in sorted(_opex_summary, key=lambda x: -x["annual"])]] * 7,
        align=["left", "center", "right", "right", "center", "center", "left"],
        font=dict(size=10, color="#2c3e50"),
        height=26,
    ),
))
fig_opex_table.update_layout(
    paper_bgcolor="#f4f6f8",
    margin=dict(l=0, r=0, t=0, b=0),
    height=len(satellites) * 26 + 90,
)


# ---------------------------------------------------------------------------
# 7. Serialize all figures to HTML divs
# ---------------------------------------------------------------------------
def fig_div(fig: go.Figure, div_id: str) -> str:
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, div_id=div_id)


_plotly_js_path = os.path.join(os.path.dirname(plotly.__file__), "package_data", "plotly.min.js")
with open(_plotly_js_path, encoding="utf-8") as _f:
    _plotly_js_content = _f.read()
plotly_js_tag = f'<script type="text/javascript">{_plotly_js_content}</script>'

div_A_demand    = fig_div(map_figs[("A", "demand")], "fig_A_demand")
div_A_assign    = fig_div(map_figs[("A", "assign")], "fig_A_assign")
div_B_demand    = fig_div(map_figs[("B", "demand")], "fig_B_demand")
div_B_assign    = fig_div(map_figs[("B", "assign")], "fig_B_assign")
div_fleet_box        = fig_div(fig_box,          "fig_fleet_box")
div_fleet_hmap       = fig_div(fig_heatmap,      "fig_fleet_hmap")
div_fleet_bar        = fig_div(fig_bar,           "fig_fleet_bar")
div_fleet_table      = fig_div(fig_table,         "fig_fleet_table")
div_cap_chart        = fig_div(fig_cap,              "fig_cap_chart")
div_cap_table        = fig_div(fig_cap_table,        "fig_cap_table")
div_cost_dots        = fig_div(fig_cost_dots,        "fig_cost_dots")
div_cost_total       = fig_div(fig_cost_total,       "fig_cost_total")
div_cost_rob_table   = fig_div(fig_cost_rob_table,   "fig_cost_rob_table")
div_theta_line       = fig_div(fig_theta_line,    "fig_theta_line")
div_theta_map        = fig_div(fig_theta_map,     "fig_theta_map")
div_theta_table       = fig_div(fig_theta_table,    "fig_theta_table")
div_opex_heatmap      = fig_div(fig_opex_heatmap,   "fig_opex_heatmap")
div_opex_lines        = fig_div(fig_opex_lines,     "fig_opex_lines")
div_opex_compare      = fig_div(fig_opex_compare,   "fig_opex_compare")
div_opex_table        = fig_div(fig_opex_table,     "fig_opex_table")


# ---------------------------------------------------------------------------
# 8. Assemble HTML
# ---------------------------------------------------------------------------
legend_items_html = "".join(
    f'<span style="display:inline-flex;align-items:center;margin-right:14px;">'
    f'<span style="width:14px;height:14px;background:{sat_color[s]};'
    f'border-radius:2px;margin-right:5px;flex-shrink:0;"></span>'
    f'{s.replace("_", " ").title()}</span>'
    for s in satellites
)

HTML = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Análisis de Píxeles y Flota</title>
  {plotly_js_tag}
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body   {{ font-family: sans-serif; margin: 0; padding: 0; background: #f4f6f8; }}
    h1     {{ color: #ffffff; margin: 0; font-size: 1.2em; font-weight: 600; }}
    #top-bar {{
      position: sticky; top: 0; z-index: 100; background: #2c3e50;
      padding: 10px 20px; display: flex; align-items: center; gap: 20px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.25);
    }}
    #top-bar label {{ color: #cde; font-size: 0.88em; font-weight: bold; }}
    #top-bar select {{
      font-size: 0.95em; padding: 5px 10px;
      border-radius: 4px; border: none; cursor: pointer;
    }}
    .page-body {{ padding: 16px; }}
    .row {{ display: flex; gap: 14px; margin-bottom: 14px; }}
    .section {{
      flex: 1; background: white; border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.10);
      padding: 14px 14px 6px; overflow: hidden;
    }}
    .section.full {{ flex: none; width: 100%; }}
    .section h2 {{ color: #2c3e50; margin: 0 0 3px; font-size: 1.05em; }}
    .section p  {{ color: #7f8c8d; margin: 0 0 10px; font-size: 0.84em; }}
    .divider {{
      border: none; border-top: 2px solid #2c3e50;
      margin: 20px 0 16px;
    }}
    .section-title {{
      color: #2c3e50; font-size: 1.1em; font-weight: 700;
      margin: 0 0 4px;
    }}
    .legend-bar {{
      background: white; border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.10);
      padding: 10px 16px; margin-bottom: 14px;
      font-size: 0.88em; color: #2c3e50;
      display: flex; flex-wrap: wrap; align-items: center; gap: 4px;
    }}
    .legend-bar strong {{ margin-right: 10px; }}
    .layer-view {{ display: none; }}
    .layer-view.active {{ display: block; }}
  </style>
</head>
<body>

<div id="top-bar">
  <h1>Análisis de Píxeles y Flota</h1>
  <div>
    <label for="layer-select">CAPA &nbsp;</label>
    <select id="layer-select" onchange="switchLayer(this.value)">
      <option value="A">Capa A (97 píxeles)</option>
      <option value="B">Capa B (64 píxeles)</option>
    </select>
  </div>
</div>

<div class="page-body">

  <!-- ── Sección 1 & 2: Mapas por capa ── -->
  <div id="view-A" class="layer-view active">
    <div class="row">
      <div class="section">
        <h2>Píxeles con demanda</h2>
        <p>Azul = con demanda. Gris = sin demanda. Grid de 1 km².</p>
        {div_A_demand}
      </div>
      <div class="section">
        <h2>Asignación por satélite</h2>
        <p>Color = satélite más cercano (euclidiana). ● = ubicación del satélite.</p>
        {div_A_assign}
      </div>
    </div>
  </div>
  <div id="view-B" class="layer-view">
    <div class="row">
      <div class="section">
        <h2>Píxeles con demanda</h2>
        <p>Azul = con demanda. Gris = sin demanda. Grid de 1 km².</p>
        {div_B_demand}
      </div>
      <div class="section">
        <h2>Asignación por satélite</h2>
        <p>Color = satélite más cercano (euclidiana). ● = ubicación del satélite.</p>
        {div_B_assign}
      </div>
    </div>
  </div>

  <div class="legend-bar">
    <strong>Satélites:</strong> {legend_items_html}
  </div>

  <hr class="divider">

  <!-- ── Sección 3: Análisis de Flota ── -->
  <p class="section-title">Análisis de Flota por Satélite</p>
  <p style="color:#7f8c8d;font-size:0.85em;margin:0 0 14px;">
    {N_SCENARIOS} escenarios de demanda &nbsp;·&nbsp;
    Vehículo: van pequeña (cap. 115 ítems) &nbsp;·&nbsp;
    Asignación euclidiana &nbsp;·&nbsp;
    Líneas punteadas = niveles de capacidad disponibles
  </p>

  <!-- Box plots (ancho completo) -->
  <div class="row">
    <div class="section full">
      <h2>Distribución de flota requerida (todos los períodos y escenarios)</h2>
      <p>Cada caja muestra la variabilidad de vehículos necesarios en el satélite a lo largo de los 12 meses y {N_SCENARIOS} escenarios de demanda.</p>
      {div_fleet_box}
    </div>
  </div>

  <!-- Heatmap + Bar chart -->
  <div class="row">
    <div class="section">
      <h2>Flota P90 por período</h2>
      <p>En el 90% de los escenarios, el satélite necesita ≤ este número de vehículos en ese mes.</p>
      {div_fleet_hmap}
    </div>
    <div class="section">
      <h2>P90 pico y capacidad recomendada</h2>
      <p>Barra = P90 del período más exigente. ◆ = nivel de capacidad mínimo que lo cubre.</p>
      {div_fleet_bar}
    </div>
  </div>

  <!-- Tabla resumen -->
  <div class="row">
    <div class="section full">
      <h2>Tabla resumen — capacidad recomendada por satélite</h2>
      <p>P50 / P75 / P90 / P95 corresponden al período pico (máximo mensual) de cada escenario. Capacidad recomendada = nivel disponible más pequeño que cubre el P90.</p>
      {div_fleet_table}
    </div>
  </div>

  <hr class="divider">

  <!-- ── Sección 4: Propuestas de capacidad ── -->
  <p class="section-title">Propuestas de niveles de capacidad por satélite</p>
  <p style="color:#7f8c8d;font-size:0.85em;margin:0 0 14px;">
    Basado en la distribución de flota P90 por período pico y {N_SCENARIOS} escenarios.
    &nbsp;·&nbsp; Banda gris = rango P10–P90. &nbsp;·&nbsp; Línea vertical = P50. &nbsp;·&nbsp;
    ▶ = P95.
  </p>

  <div style="display:flex;gap:16px;margin-bottom:12px;">
    <div style="flex:1;background:#eaf2fb;border-left:4px solid #1565C0;border-radius:6px;padding:12px 16px;">
      <strong style="color:#1565C0;">■ Propuesta A — Rango operacional completo</strong>
      <p style="margin:6px 0 0;font-size:0.85em;color:#2c3e50;">
        Niveles desde 2 vehículos por debajo del P25 hasta 2 por encima del P95, en pasos de 2.
        Da al modelo flexibilidad para optimizar dentro del rango realista de demanda.
        Descarta niveles que nunca serían elegidos (por debajo del P25).
        Útil cuando el tiempo de resolución del modelo es bajo o se quiere explorar más opciones.
      </p>
    </div>
    <div style="flex:1;background:#fdedec;border-left:4px solid #c0392b;border-radius:6px;padding:12px 16px;">
      <strong style="color:#c0392b;">◆ Propuesta B — 3 niveles esenciales</strong>
      <p style="margin:6px 0 0;font-size:0.85em;color:#2c3e50;">
        Solo 3 niveles por satélite: <b>mínimo</b> (cubre P50 — operación básica),
        <b>recomendado</b> (cubre P90 — 9 de cada 10 escenarios) y
        <b>máximo</b> (cubre el 100% observado).
        Modelo más compacto y rápido de resolver. Ideal como punto de partida.
      </p>
    </div>
  </div>

  <div class="row">
    <div class="section full">
      <h2>Comparación de propuestas sobre distribución de demanda</h2>
      <p>Banda gris = rango de flota P10–P90 por satélite. Los marcadores muestran los niveles de cada propuesta en relación a la distribución real de demanda.</p>
      {div_cap_chart}
    </div>
  </div>

  <div class="row">
    <div class="section full">
      <h2>Tabla resumen — niveles propuestos por satélite</h2>
      <p>Propuesta A ofrece el rango completo útil. Propuesta B reduce a 3 niveles clave. Ambas evitan niveles innecesariamente bajos.</p>
      {div_cap_table}
    </div>
  </div>

  <!-- Subsección: análisis de costos -->
  <p style="color:#2c3e50;font-size:1.0em;font-weight:700;margin:18px 0 4px;">
    Análisis de costos y robustez de las propuestas
  </p>
  <p style="color:#7f8c8d;font-size:0.85em;margin:0 0 12px;">
    OPEX mensual desde tabla de tarifas. Niveles &gt;12 veh. extrapolados con incremento de ${_opex_increment:,}/2 veh (tendencia 10→12).
    &nbsp;·&nbsp; ✓ = cubre ≥90% · ~ = cubre 75-90% · ✗ = cubre &lt;75%.
  </p>

  <div style="display:flex;gap:12px;margin-bottom:12px;font-size:0.84em;color:#2c3e50;">
    <div style="background:#eaf7ef;border-left:3px solid #27ae60;padding:8px 12px;border-radius:4px;flex:1;">
      <b style="color:#27ae60;">■/◆ Verde ≥90%</b> — nivel robusto: cubre el 90% o más de los escenarios
    </div>
    <div style="background:#fef9e7;border-left:3px solid #f39c12;padding:8px 12px;border-radius:4px;flex:1;">
      <b style="color:#f39c12;">■/◆ Naranja 75-90%</b> — nivel aceptable: falla en 1 de cada 4-10 escenarios
    </div>
    <div style="background:#fdedec;border-left:3px solid #e74c3c;padding:8px 12px;border-radius:4px;flex:1;">
      <b style="color:#e74c3c;">■/◆ Rojo &lt;75%</b> — nivel insuficiente: falla frecuentemente (excluido en Prop.B)
    </div>
  </div>

  <div class="row">
    <div class="section full">
      <h2>OPEX mensual vs cobertura de escenarios — cada punto es un nivel posible</h2>
      <p>■ = Propuesta A (rango completo). ◆ = Propuesta B (3 niveles). El color indica qué fracción de los 500 escenarios queda cubierta. Solo los puntos verdes son robustos.</p>
      {div_cost_dots}
    </div>
  </div>

  <div class="row">
    <div class="section">
      <h2>OPEX total de la red por escenario de demanda</h2>
      <p>Comparación del gasto mensual total (9 satélites) si se dimensiona al P50, P90 (ambas propuestas) o al máximo histórico.</p>
      {div_cost_total}
    </div>
  </div>

  <div class="row">
    <div class="section full">
      <h2>Tabla de niveles de capacidad disponibles — Actual vs Propuestas</h2>
      <p>
        Por cada satélite se listan los niveles de capacidad disponibles con su
        <b>costo fijo de instalación</b> (de <code>input_facilities.xlsx</code>) y el
        <b>% de escenarios cubiertos</b> a ese nivel de flota.
        <b>◄</b> marca el nivel que cubre el P90 de demanda (nivel recomendado).
        Las propuestas A y B eliminan niveles que nunca serían óptimos,
        reduciendo el espacio de decisión del modelo.
      </p>
      {div_cost_rob_table}
    </div>
  </div>

  <hr class="divider">

  <!-- ── Sección 5: Análisis DC vs Satélites (θ) ── -->
  <p class="section-title">Análisis DC directo vs Satélites — Parámetro θ</p>
  <p style="color:#7f8c8d;font-size:0.85em;margin:0 0 14px;">
    θ = % de píxeles más cercanos al DC atendidos directamente por él (camión grande 🚛). &nbsp;·&nbsp;
    El resto va por satélite (van pequeña 🚐). &nbsp;·&nbsp;
    {N_SCENARIOS} escenarios · Valores = Flota P90 período pico.
  </p>

  <!-- Línea evolución total -->
  <div class="row">
    <div class="section full">
      <h2>Tradeoff total: flota DC vs flota satélites al variar θ</h2>
      <p>Al aumentar θ, el DC necesita más camiones grandes y los satélites necesitan menos vans.</p>
      {div_theta_line}
    </div>
  </div>

  <!-- Mapa geográfico de asignación con θ -->
  <div class="row">
    <div class="section full">
      <h2>Mapa de asignación según θ <span style="font-weight:normal;font-size:0.9em">(usar dropdown para cambiar θ)</span></h2>
      <p>Píxeles de color = atendidos por su satélite. Píxeles gris oscuro = transferidos al DC directo. Usar dropdown para ver cómo cambia la asignación.</p>
      {div_theta_map}
    </div>
  </div>

  <!-- Tabla resumen θ -->
  <div class="row">
    <div class="section full">
      <h2>Tabla resumen — DC 🚛 y Satélite 🚐 por escenario θ</h2>
      <p>Muestra flota P90 (período pico) para θ = 0%, 30%, 60% y 90%. 🚛 = camiones grandes DC · 🚐 = vans satélite.</p>
      {div_theta_table}
    </div>
  </div>

  <hr class="divider">

  <!-- ── Sección 6: Costos operacionales propuestos ── -->
  <p class="section-title">Propuesta de costos operacionales — basada en datos históricos reales</p>

  <!-- Caja de fórmula y metodología -->
  <div style="display:flex;gap:14px;margin-bottom:14px;flex-wrap:wrap;">

    <div style="flex:2;background:#f8f9ff;border:1px solid #d0d7f5;border-radius:8px;padding:16px 20px;">
      <p style="margin:0 0 8px;font-weight:700;color:#2c3e50;font-size:1.0em;">📐 Fórmula</p>
      <div style="background:#2c3e50;color:#ecf0f1;border-radius:6px;padding:12px 16px;font-family:monospace;font-size:0.9em;margin-bottom:10px;">
        cost_operation[q][t] = OPEX(q) &times; [ 0.70 + 0.30 &times; factor_real(sat, t) ]
      </div>
      <table style="font-size:0.83em;border-collapse:collapse;width:100%;">
        <tr><td style="padding:3px 8px;font-weight:600;">OPEX(q)</td>
            <td style="padding:3px 8px;">Costo mensual base para q vehículos (tabla de tarifas del usuario)</td></tr>
        <tr style="background:#f4f6f8;"><td style="padding:3px 8px;font-weight:600;">0.70 (α fijo)</td>
            <td style="padding:3px 8px;">Arriendo, dotación base, seguros, administración — no varía con el volumen</td></tr>
        <tr><td style="padding:3px 8px;font-weight:600;">0.30 (1-α variable)</td>
            <td style="padding:3px 8px;">Combustible, horas extra, desgaste vehículos — escala con la demanda real</td></tr>
        <tr style="background:#f4f6f8;"><td style="padding:3px 8px;font-weight:600;">factor_real(sat, t)</td>
            <td style="padding:3px 8px;">Factor estacional del satélite en el mes t, ponderado por mezcla de capas A/B</td></tr>
      </table>
    </div>

    <div style="flex:1;background:#fff8e1;border:1px solid #ffe082;border-radius:8px;padding:16px 20px;">
      <p style="margin:0 0 8px;font-weight:700;color:#e65100;font-size:1.0em;">⚠️ Por qué no usar los escenarios</p>
      <p style="font-size:0.83em;color:#333;margin:0 0 8px;">
        Los escenarios (Beta) muestran variación mensual <b>&lt;2%</b> porque el parámetro
        <code>loc</code> apenas cambia entre períodos (0.978–0.993 en capa A).
      </p>
      <p style="font-size:0.83em;color:#333;margin:0 0 8px;">
        Los datos históricos reales (<code>raw_pixels_with_drop_demand.csv</code>) muestran:
      </p>
      <div style="font-size:0.82em;font-family:monospace;background:#fff3cd;border-radius:4px;padding:8px 10px;">
        Feb: 0.699 → −30% vs promedio<br>
        Oct: 1.261 → +26% vs promedio<br>
        Dic: 1.415 → +42% vs promedio<br>
        <b>Variación total: ~22% entre meses</b>
      </div>
    </div>

    <div style="flex:1;background:#e8f5e9;border:1px solid #a5d6a7;border-radius:8px;padding:16px 20px;">
      <p style="margin:0 0 8px;font-weight:700;color:#1b5e20;font-size:1.0em;">✅ Ejemplo: Sopocachi (q=18v)</p>
      <p style="font-size:0.83em;color:#333;margin:0 0 6px;">OPEX base: <b>$8,407/mes</b></p>
      <div style="font-size:0.82em;font-family:monospace;background:#f1f8e9;border-radius:4px;padding:8px 10px;">
        Febrero: 8,407×[0.70+0.30×0.695]<br>
        &nbsp;&nbsp;= <b>$7,931/mes</b> (mínimo)<br><br>
        Diciembre: 8,407×[0.70+0.30×1.421]<br>
        &nbsp;&nbsp;= <b>$9,470/mes</b> (máximo)<br><br>
        Diferencia: <b>$1,539/mes</b><br>
        Variación anual: <b>±9.2%</b>
      </div>
    </div>

  </div>

  <!-- Heatmap OPEX mensual -->
  <div class="row">
    <div class="section full">
      <h2>Costo operacional mensual propuesto por satélite (nivel recomendado P90)</h2>
      <p>Cada celda = costo mensual propuesto en el nivel de capacidad que cubre el P90 de demanda.
         La variación refleja la estacionalidad de la demanda del satélite.</p>
      {div_opex_heatmap}
    </div>
  </div>

  <!-- Perfiles mensuales + comparación -->
  <div class="row">
    <div class="section">
      <h2>Perfil mensual por satélite</h2>
      <p>Variación del costo mes a mes. Satélites con mayor variación = demanda más estacional.</p>
      {div_opex_lines}
    </div>
    <div class="section">
      <h2>Comparación: actual (Excel) vs propuesto</h2>
      <p>El costo actual en el Excel es significativamente inferior al OPEX real de la tabla de tarifas.</p>
      {div_opex_compare}
    </div>
  </div>

  <!-- Tabla de valores -->
  <div class="row">
    <div class="section full">
      <h2>Valores propuestos para <code>cost_operation[q][t]</code> — usar para actualizar input_facilities.xlsx</h2>
      <p>Lista de 12 valores mensuales ($/mes) en el nivel recomendado. Variación estacional = diferencia
         porcentual entre mes pico y mes valle. Nivel recomendado = menor nivel Propuesta A que cubre P90.</p>
      {div_opex_table}
    </div>
  </div>

</div>

<script>
  function switchLayer(layer) {{
    document.querySelectorAll('.layer-view').forEach(function(el) {{
      el.classList.remove('active');
    }});
    document.getElementById('view-' + layer).classList.add('active');
  }}
</script>
</body>
</html>
"""

with open(OUTPUT, "w", encoding="utf-8") as f:
    f.write(HTML)

print(f"\nOutput written to: {OUTPUT}")

"""Build the monthly demand panel from the raw delivery-event file.

`base_customers_all_years_z7.csv` is one row per delivery event, but it was
materialized through a join that fanned every event out once per active year of
the customer (verified: row multiplicity equals the customer's distinct-year count
for 100% of customers, with a hard ceiling at 3 and a leftover `_merge` column).
Deduplicating is therefore mandatory: without it total demand is inflated from
3,924,851 to 10,694,636.

The output panel is one row per `(layer, pixel, year, month)` with the three
quantities the scenario generator needs: active customers (`stop`), items, and
items per visit (`drop`).
"""

import numpy as np
import pandas as pd

from src.core.constants import EXCLUDED_YEAR_MONTHS, EXPECTED_FANOUT_SHARE, PATH_PANEL_MONTHLY, PATH_RAW_DEMAND
from src.scenarios.crosswalk import assign_customers, load_manual_crosswalk
from src.tools.logging import get_logger

logger = get_logger("DemandPanel")

RAW_COLUMNS = ["cod_customer", "demand", "c_ter_act", "lon", "lat", "date", "cluster_targets", "_merge"]


def load_events(fanout_tolerance: float = 0.05) -> pd.DataFrame:
    """Load the raw file, undo the join fan-out and net returns.

    Aborts if the observed fan-out share departs from what was verified for this
    file: a silently different duplication pattern would mean every demand total
    downstream is wrong, which is worse than failing here.
    """
    raw = pd.read_csv(PATH_RAW_DEMAND)
    n_raw = len(raw)

    events = raw.drop_duplicates()
    share = 1.0 - len(events) / n_raw
    logger.info(f"Raw rows {n_raw:,} -> {len(events):,} after dedup ({share:.1%} were fan-out).")
    if abs(share - EXPECTED_FANOUT_SHARE) > fanout_tolerance:
        raise ValueError(
            f"Fan-out share {share:.3f} departs from the verified {EXPECTED_FANOUT_SHARE:.3f}. "
            "The raw file changed shape — re-verify before trusting any demand total."
        )

    # Cross-check the mechanism itself, not just the magnitude.
    years = events["date"].str[:4]
    mult = raw.groupby(RAW_COLUMNS, dropna=False).size().rename("m").reset_index()
    max_mult = mult.groupby("cod_customer")["m"].max()
    n_years = events.assign(y=years).groupby("cod_customer")["y"].nunique()
    agreement = (max_mult.reindex(n_years.index) == n_years).mean()
    logger.info(f"Row multiplicity equals the customer's active-year count for {agreement:.1%} of customers.")
    if agreement < 0.99:
        raise ValueError(
            f"Fan-out mechanism does not hold ({agreement:.1%} agreement). Deduplication may be discarding real events."
        )

    events = events.drop(columns=["_merge", "cluster_targets", "c_ter_act"])
    events["year"] = years.astype(int)
    events["month"] = events["date"].str[5:7].astype(int)

    # Net returns against same-day deliveries instead of discarding them.
    n_negative = int((events["demand"] < 0).sum())
    netted = events.groupby(["cod_customer", "date", "year", "month"], as_index=False).agg(
        demand=("demand", "sum"), lon=("lon", "first"), lat=("lat", "first")
    )
    before = len(netted)
    netted = netted[netted["demand"] > 0]
    logger.info(
        f"Netted {n_negative:,} negative rows into customer-days; "
        f"{before - len(netted):,} customer-days were left non-positive and dropped."
    )
    return netted


def customer_drop(events: pd.DataFrame) -> pd.DataFrame:
    """Per-customer items, visits and drop size (items per visit)."""
    g = events.groupby("cod_customer").agg(
        items=("demand", "sum"),
        visits=("date", "nunique"),
        lon=("lon", "first"),
        lat=("lat", "first"),
    )
    g["drop"] = g["items"] / g["visits"]
    return g.reset_index()


def build_panel(events: pd.DataFrame | None = None) -> pd.DataFrame:
    """Aggregate delivery events into the monthly `(layer, pixel)` panel."""
    if events is None:
        events = load_events()

    customers = customer_drop(events)
    crosswalk = load_manual_crosswalk()
    assigned = assign_customers(customers, crosswalk)

    placed = assigned[assigned["id_pixel"].notna()]
    dropped_demand = assigned.loc[assigned["id_pixel"].isna(), "items"].sum()
    logger.info(
        f"Off-grid customers dropped: {len(assigned) - len(placed)} "
        f"({dropped_demand / assigned['items'].sum():.1%} of demand)."
    )

    keys = placed.set_index("cod_customer")[["id_pixel", "layer", "pixel", "layer_source"]]
    ev = events.join(keys, on="cod_customer", how="inner")

    panel = ev.groupby(["id_pixel", "layer", "pixel", "year", "month"], as_index=False).agg(
        n_customers=("cod_customer", "nunique"),
        items=("demand", "sum"),
        n_visits=("date", "nunique"),
        n_customer_days=("demand", "size"),
    )
    # `drop` is items per customer-visit: a customer served twice in the month
    # contributes two visits, matching how the CA consumes `drop`.
    panel["drop"] = panel["items"] / panel["n_customer_days"]

    # Demand in *model* units: one representative delivery round in the period,
    # `stop * drop`. This is the quantity the scenario contract carries and the one
    # the regime targets are calibrated against — not `items`, which is the raw
    # monthly count and is ~3x larger because customers repeat within a month.
    panel["model_demand"] = panel["n_customers"] * panel["drop"]

    # Share of the pixel-month's demand whose layer came from the crosswalk.
    src = (
        ev.assign(from_cw=(ev["layer_source"] == "crosswalk").astype(float) * ev["demand"])
        .groupby(["id_pixel", "year", "month"], as_index=False)
        .agg(cw_items=("from_cw", "sum"), tot_items=("demand", "sum"))
    )
    src["crosswalk_share"] = src["cw_items"] / src["tot_items"]
    panel = panel.merge(src[["id_pixel", "year", "month", "crosswalk_share"]], on=["id_pixel", "year", "month"])

    excluded = np.zeros(len(panel), dtype=bool)
    for year, month in EXCLUDED_YEAR_MONTHS:
        excluded |= (panel["year"] == year) & (panel["month"] == month)
    panel["excluded"] = excluded
    logger.info(f"Flagged {int(excluded.sum())} pixel-months as excluded (extraction gaps): {EXCLUDED_YEAR_MONTHS}.")

    panel = panel.sort_values(["id_pixel", "year", "month"]).reset_index(drop=True)
    logger.info(
        f"Panel: {len(panel):,} rows | {panel['id_pixel'].nunique()} pixels | "
        f"{panel.groupby(['year', 'month']).ngroups} year-months | items {panel['items'].sum():,.0f}"
    )
    return panel


def save_panel(panel: pd.DataFrame) -> None:
    """Write the panel cache."""
    PATH_PANEL_MONTHLY.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(PATH_PANEL_MONTHLY, index=False)
    logger.info(f"Panel written to {PATH_PANEL_MONTHLY}")


def load_panel() -> pd.DataFrame:
    """Read the panel cache, building it first if absent."""
    if not PATH_PANEL_MONTHLY.exists():
        panel = build_panel()
        save_panel(panel)
        return panel
    return pd.read_csv(PATH_PANEL_MONTHLY)

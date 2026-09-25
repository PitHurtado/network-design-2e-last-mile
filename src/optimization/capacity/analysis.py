"""Satellite capacity analysis on the `capacity` scenario set of a version.

For every regime and scenario of the set:

1. each pixel is assigned to its nearest satellite (haversine between the pixel's service
   point and the satellite), the procedure of `OLD/notebooks/mvp_parametros`;
2. the CA gives the small-vehicle fleet each pixel-period needs from that satellite; a
   satellite's fleet in a period is the ceiling of the sum over its pixels;
3. its *peak fleet* in the scenario is the maximum over the periods.

The peaks of all regimes and scenarios, pooled, are the distribution a `LevelMethod`
turns into capacity levels, so one table spans the demand range of every regime. Costs
come from the tariff table; the operating cost of level q in period t is
`OPEX(q) × [α + (1 − α) × s(t)]`, where `s(t)` is the satellite's seasonal factor: the
demand of its pixels in the version's expected scenario, relative to its annual mean.
"""

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.core.constants import N_PERIODS, REGIMES
from src.core.contract import ScenarioLayout
from src.core.grid import haversine_km
from src.core.inputs import get_facilities, get_pixels
from src.optimization.capacity.levels import level_method
from src.optimization.capacity.table import save_table
from src.optimization.capacity.tariffs import PATH_TARIFFS, Tariffs
from src.optimization.instance import InstanceBuilder, InstanceSpec
from src.tools.io import read_json, write_json
from src.tools.logging import get_logger

logger = get_logger("CapacityAnalysis")

VEHICLE = "small"
SEASONAL_REGIME = "normal"  # a regime is a scalar on the level, so every regime has the same seasonal shape


@dataclass(frozen=True)
class CapacityConfig:
    scenarios: str
    levels: str = "percentiles-a"
    regimes: tuple[str, ...] = REGIMES
    alpha_fixed: float | None = None  # None: the tariff table's
    use_euclidean_distance: bool = True  # as the optimization runs
    batch_size: int = 10


def nearest_satellite_assignment() -> dict[str, str]:
    """Pixel id -> nearest satellite id."""
    facilities = get_facilities()
    assignment = {}
    for id_pixel, pixel in get_pixels().items():
        point = pixel.geo_point
        assignment[id_pixel] = min(
            facilities,
            key=lambda i: haversine_km(point.lon, point.lat, facilities[i].geo_point.lon, facilities[i].geo_point.lat),
        )
    return assignment


class CapacityAnalysis:
    def __init__(self, layout: ScenarioLayout, version: str, config: CapacityConfig, tariffs: Tariffs):
        self.layout = layout
        self.version = version
        self.config = config
        self.tariffs = tariffs
        self.builder = InstanceBuilder(layout, version=version)
        self.assignment = nearest_satellite_assignment()
        self.satellites = sorted(get_facilities())

    def peak_fleets(self) -> pd.DataFrame:
        """One row per (regime, scenario, satellite): peak over periods of the ceiled fleet."""
        rows = []
        pixels_of = {i: [j for j, sat in self.assignment.items() if sat == i] for i in self.satellites}
        for regime in self.config.regimes:
            ids = read_json(self.layout.set_manifest(regime, "capacity"))["scenario_ids"]
            for start in range(0, len(ids), self.config.batch_size):
                batch = tuple(ids[start : start + self.config.batch_size])
                instance = self.builder.build(
                    InstanceSpec(
                        n_scenarios=len(batch),
                        regime=regime,
                        scenario_set="capacity",
                        scenario_ids=batch,
                        use_euclidean_distance=self.config.use_euclidean_distance,
                        id_instance="capacity",
                    )
                )
                for scenario_id, scenario in instance.scenarios.items():
                    fleet = scenario.get_fleet_size("facility")
                    for satellite in self.satellites:
                        per_period = [
                            math.ceil(
                                sum(
                                    fleet[(satellite, j, VEHICLE, t, scenario_id)]
                                    for j in pixels_of[satellite]
                                    if (satellite, j, VEHICLE, t, scenario_id) in fleet
                                )
                            )
                            for t in range(N_PERIODS)
                        ]
                        rows.append(
                            {"regime": regime, "scenario": scenario_id, "satellite": satellite, "peak_fleet": max(per_period)}
                        )
                logger.info(f"[{regime}] fleet for {start + len(batch)}/{len(ids)} capacity scenarios")
        return pd.DataFrame(rows)

    def seasonal_factors(self) -> dict[str, list[float]]:
        """Per satellite, the demand of its pixels in each period of the expected scenario over their annual mean."""
        regime = SEASONAL_REGIME if SEASONAL_REGIME in self.config.regimes else self.config.regimes[0]
        (expected_id,) = self.layout.scenario_ids(regime, "expected", 1)
        pixels = self.layout.load_pixels(expected_id, regime, "expected")
        factors = {}
        for satellite in self.satellites:
            demand = np.zeros(N_PERIODS)
            for id_pixel, pixel in pixels.items():
                if self.assignment.get(id_pixel) == satellite:
                    demand += np.asarray(pixel.demand_by_period, dtype=float)
            factors[satellite] = [round(float(x), 6) for x in (demand / demand.mean())] if demand.sum() > 0 else [1.0] * N_PERIODS
        return factors

    def run(self, out) -> dict:
        """Write `capacity.json`, `peak_fleet.csv` and `assignment.json` into `out`; return a summary."""
        peaks = self.peak_fleets()
        seasonal = self.seasonal_factors()
        alpha = self.tariffs.alpha_fixed if self.config.alpha_fixed is None else self.config.alpha_fixed
        method = level_method(self.config.levels)
        satellites, summary = {}, {}
        for satellite in self.satellites:
            values = peaks.loc[peaks["satellite"] == satellite, "peak_fleet"].to_numpy()
            levels = [0] + method.levels(values)
            satellites[satellite] = {
                "levels": levels,
                "cost_installation": {str(q): round(self.tariffs.installation_of(q), 2) for q in levels},
                "cost_operation": {
                    str(q): [round(self.tariffs.opex_of(q) * (alpha + (1 - alpha) * s), 2) for s in seasonal[satellite]]
                    for q in levels
                },
                "coverage_pct": {str(q): round(float((values <= q).mean() * 100), 1) for q in levels},
                "peak_fleet": {
                    "p25": float(np.percentile(values, 25)),
                    "p50": float(np.percentile(values, 50)),
                    "p90": float(np.percentile(values, 90)),
                    "p95": float(np.percentile(values, 95)),
                    "max": int(values.max()),
                },
                "n_pixels": sum(1 for sat in self.assignment.values() if sat == satellite),
                "seasonal_factor": seasonal[satellite],
            }
            summary[satellite] = {"levels": levels, "p95": satellites[satellite]["peak_fleet"]["p95"]}
        save_table(out, satellites)
        peaks.to_csv(out / "peak_fleet.csv", index=False)
        write_json(out / "assignment.json", dict(sorted(self.assignment.items())))
        return {"config": asdict(self.config), "alpha_fixed": alpha, "satellites": summary}


def run_analysis(out, layout: ScenarioLayout, version: str, config: CapacityConfig, tariffs_path=PATH_TARIFFS) -> dict:
    return CapacityAnalysis(layout, version, config, Tariffs.load(tariffs_path)).run(out)

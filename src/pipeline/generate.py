"""Spatially correlated demand scenario generator.

Three levels per period, following the fitted structure in `marginals.py` and the
spatial dependence in `spatial.py`:

    level 1  F_t          common shock shared by every pixel in the period
    level 2  dev_j        pixel deviation, spatially correlated via a Gaussian copula
    level 3  nugget       the purely local component, folded into the copula's
                          correlation matrix as `corr(0+) = 1 - nugget`

    stop[j,t]   = max(1, round( E_stop[j,t] * multiplier**0.70 * F_t * dev_j ))
    drop[j,t]   = E_drop[j,t] * multiplier**0.30 * G_t * dev'_j
    demand[j,t] = stop[j,t] * drop[j,t]

All multiplicative shocks are mean-preserving (`exp(xi - sigma^2/2)`), so amplifying
dispersion does not move the level. The previous procedure used `mu * exp(xi)`,
which inflated the mean by up to 79% in December while claiming to preserve it.

Because the marginals are lognormal, the Gaussian copula reduces exactly to a
multivariate lognormal: no NORTA calibration is needed for the continuous part.
Discreteness enters only through the rounding of `stop`, whose effect on the
realized correlogram is measured in the analysis report.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.constants import (
    DEFAULT_SCENARIO_VERSION,
    N_PERIODS,
    REGIME_DROP_EXPONENT,
    REGIME_STOP_EXPONENT,
    SEED_BASE,
    scenario_dir,
)
from src.core.logging import get_logger
from src.pipeline.marginals import expected_matrix

logger = get_logger("Generate")


class ScenarioGenerator:
    """Draws demand scenarios that respect the scenario contract.

    The contract is narrow: each pixel must carry `id_pixel`, `stop`, `drop` and
    `demand` arrays of length `N_PERIODS`, with `stop >= 1` and `drop > 0`, because
    the CA only writes cost keys for `demand > 0` and the uncapacitated model
    indexes them directly.
    """

    def __init__(
        self,
        pixels: list[str],
        expected_stop: np.ndarray,
        expected_drop: np.ndarray,
        sigma_stop: np.ndarray,
        sigma_drop: np.ndarray,
        sigma_common_stop: float,
        sigma_common_drop: float,
        cholesky: np.ndarray,
        regime_stop_exponent: float = REGIME_STOP_EXPONENT,
        regime_drop_exponent: float = REGIME_DROP_EXPONENT,
    ):
        self.pixels = list(pixels)
        self.expected_stop = expected_stop
        self.expected_drop = expected_drop
        self.sigma_stop = sigma_stop
        self.sigma_drop = sigma_drop
        self.sigma_common_stop = sigma_common_stop
        self.sigma_common_drop = sigma_common_drop
        self.cholesky = cholesky
        if not np.isclose(regime_stop_exponent + regime_drop_exponent, 1.0):
            raise ValueError("Regime stop/drop exponents must add to 1.")
        self.regime_stop_exponent = regime_stop_exponent
        self.regime_drop_exponent = regime_drop_exponent
        self.floor_hits = 0
        self.cells_drawn = 0

    @classmethod
    def from_fit(cls, fitted: dict, pixels: list[str], cholesky: np.ndarray) -> "ScenarioGenerator":
        """Build a generator from the fitted marginals, aligned on `pixels`."""
        sigma_stop = fitted["stop"]["sigma"]["sigma"].reindex(pixels).to_numpy(dtype=float)
        sigma_drop = fitted["drop"]["sigma"]["sigma"].reindex(pixels).to_numpy(dtype=float)
        if not np.isfinite(sigma_stop).all() or not np.isfinite(sigma_drop).all():
            missing = [p for p, s in zip(pixels, sigma_stop) if not np.isfinite(s)]
            raise ValueError(f"No dispersion fitted for pixels: {missing[:5]}")

        return cls(
            pixels=pixels,
            expected_stop=expected_matrix(fitted, "stop", pixels),
            expected_drop=expected_matrix(fitted, "drop", pixels),
            sigma_stop=sigma_stop,
            sigma_drop=sigma_drop,
            sigma_common_stop=fitted["stop"]["sigma_common"],
            sigma_common_drop=fitted["drop"]["sigma_common"],
            cholesky=cholesky,
        )

    @classmethod
    def from_params(cls, params: dict) -> "ScenarioGenerator":
        """Rebuild a generator from a persisted `shape_params.json`.

        This is the reproducible path: given the parameter file and the tracked raw
        pixel inputs, the scenarios regenerate bit-for-bit without the raw demand
        file or the panel.
        """
        # Imported here to keep the module importable without the spatial deps
        # when only reading generated scenarios.
        from src.pipeline.spatial import cholesky_factor, correlation_matrix, haversine_matrix, pixel_centroids

        pixels = list(params["pixels"])
        expectation = {}
        for quantity in ("stop", "drop"):
            part = params[quantity]
            matrix = np.zeros((len(pixels), N_PERIODS))
            for i, id_pixel in enumerate(pixels):
                layer = id_pixel.split("-")[0]
                level = part["level"][id_pixel]
                for t in range(N_PERIODS):
                    matrix[i, t] = np.exp(level + part["season"][f"{layer}|{t + 1}"])
            expectation[quantity] = matrix

        centroids = pixel_centroids().set_index("id_pixel").reindex(pixels)
        distances = haversine_matrix(centroids["lon"].to_numpy(), centroids["lat"].to_numpy())
        spatial = params["spatial"]
        sigma = correlation_matrix(distances, spatial["rho_km"], spatial["nugget"], spatial.get("plateau", 0.0))

        scaling = params.get("regime_scaling")
        if scaling is None:
            raise ValueError("shape_params.json has no regime_scaling; run recalibrate_regimes first.")

        return cls(
            pixels=pixels,
            expected_stop=expectation["stop"],
            expected_drop=expectation["drop"],
            sigma_stop=np.array([params["stop"]["sigma"][p] for p in pixels], dtype=float),
            sigma_drop=np.array([params["drop"]["sigma"][p] for p in pixels], dtype=float),
            sigma_common_stop=params["stop"]["sigma_common"],
            sigma_common_drop=params["drop"]["sigma_common"],
            cholesky=cholesky_factor(sigma),
            regime_stop_exponent=float(scaling["stop_exponent"]),
            regime_drop_exponent=float(scaling["drop_exponent"]),
        )

    def _spatial_field(self, rng: np.random.Generator) -> np.ndarray:
        """One draw of the spatially correlated standard normal field."""
        return self.cholesky @ rng.standard_normal(len(self.pixels))

    def draw(self, rng: np.random.Generator, multiplier: float, deterministic: str | None = None):
        """Draw one scenario.

        `deterministic` selects a shock-free variant: `"mean"` sets every
        multiplicative factor to its mean (1.0), `"median"` to its median.
        """
        n_pixels = len(self.pixels)
        stop = np.zeros((n_pixels, N_PERIODS), dtype=int)
        drop = np.zeros((n_pixels, N_PERIODS), dtype=float)

        for t in range(N_PERIODS):
            if deterministic == "mean":
                f_stop = f_drop = 1.0
                dev_stop = np.ones(n_pixels)
                dev_drop = np.ones(n_pixels)
            elif deterministic == "median":
                f_stop = np.exp(-0.5 * self.sigma_common_stop**2)
                f_drop = np.exp(-0.5 * self.sigma_common_drop**2)
                dev_stop = np.exp(-0.5 * self.sigma_stop**2)
                dev_drop = np.exp(-0.5 * self.sigma_drop**2)
            else:
                f_stop = float(np.exp(rng.normal(0.0, self.sigma_common_stop) - 0.5 * self.sigma_common_stop**2))
                f_drop = float(np.exp(rng.normal(0.0, self.sigma_common_drop) - 0.5 * self.sigma_common_drop**2))
                dev_stop = np.exp(self.sigma_stop * self._spatial_field(rng) - 0.5 * self.sigma_stop**2)
                dev_drop = np.exp(self.sigma_drop * self._spatial_field(rng) - 0.5 * self.sigma_drop**2)

            raw_stop = self.expected_stop[:, t] * multiplier**self.regime_stop_exponent * f_stop * dev_stop
            rounded = np.rint(raw_stop).astype(int)
            self.floor_hits += int((rounded < 1).sum())
            self.cells_drawn += n_pixels
            stop[:, t] = np.maximum(1, rounded)
            drop[:, t] = self.expected_drop[:, t] * multiplier**self.regime_drop_exponent * f_drop * dev_drop

        return stop, drop

    def mean_period_total(self, rng: np.random.Generator, multiplier: float) -> float:
        """Mean per-period model demand of one drawn scenario."""
        stop, drop = self.draw(rng, multiplier)
        return float((stop * drop).sum() / N_PERIODS)

    def base_period_total(self) -> float:
        """Per-period model demand with every shock at its mean and no regime scaling."""
        return float((self.expected_stop * self.expected_drop).sum() / N_PERIODS)

    def to_payload(self, stop: np.ndarray, drop: np.ndarray, id_scenario, scenario_type: str) -> dict:
        """Assemble the scenario JSON payload, enforcing the contract invariants."""
        if stop.min() < 1:
            raise ValueError("stop must be >= 1 for every pixel and period.")
        if drop.min() <= 0:
            raise ValueError("drop must be > 0 for every pixel and period.")

        pixels = []
        for i, id_pixel in enumerate(self.pixels):
            stop_row = [int(v) for v in stop[i]]
            drop_row = [round(float(v), 4) for v in drop[i]]
            pixels.append(
                {
                    "id_pixel": id_pixel,
                    "stop": stop_row,
                    "drop": drop_row,
                    "demand": [round(s * d, 4) for s, d in zip(stop_row, drop_row)],
                }
            )
        return {"id_scenario": id_scenario, "type": scenario_type, "pixels": pixels}


def write_scenario(payload: dict, regime: str, version: str, scenario_set: str, id_scenario: str) -> Path:
    """Write one scenario under its immutable version/regime/set directory."""
    directory = scenario_dir(regime, version, scenario_set)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"scenario_{id_scenario}.json"
    with open(path, "w") as file:
        json.dump(payload, file, indent=2)
    return path


def scenario_id(version: str, regime: str, scenario_set: str, index: int | None = None) -> str:
    """Stable, filename-safe identifier; never derived from filesystem ordering."""
    base = f"{version}-{regime}-{scenario_set}"
    return base if index is None else f"{base}-{index:03d}"


def _seeds(seed_base: int, scenario_set: str, n_scenarios: int) -> list[np.random.SeedSequence]:
    """Set-specific but regime-independent streams, so regimes are comparable."""
    set_codes = {"optimization": 30, "validation": 100}
    return np.random.SeedSequence([seed_base, set_codes[scenario_set]]).spawn(n_scenarios)


def _annual_expected_payload(generator: ScenarioGenerator, regime: str, version: str, multiplier: float) -> dict:
    """One-period annual mean of the deterministic expected scenario, outside the LP contract."""
    stop, drop = generator.draw(np.random.default_rng(0), multiplier, deterministic="mean")
    annual_stop = stop.mean(axis=1)
    annual_demand = (stop * drop).mean(axis=1)
    annual_drop = annual_demand / annual_stop
    pixels = [
        {
            "id_pixel": id_pixel,
            "stop": [round(float(annual_stop[i]), 6)],
            "drop": [round(float(annual_drop[i]), 6)],
            "demand": [round(float(annual_demand[i]), 6)],
        }
        for i, id_pixel in enumerate(generator.pixels)
    ]
    return {
        "id_scenario": scenario_id(version, regime, "annual_expected"),
        "type": "annual_expected",
        "periods": 1,
        "source": "mean over the 12 periods of the expected scenario",
        "optimization_compatible": True,
        "pixels": pixels,
    }


def generate_set(
    generator: ScenarioGenerator,
    regime: str,
    multiplier: float,
    scenario_set: str,
    n_scenarios: int,
    version: str = DEFAULT_SCENARIO_VERSION,
    seed_base: int = SEED_BASE,
    params_sha256: str | None = None,
) -> dict:
    """Generate one reproducible scenario set and its self-describing manifest."""
    if scenario_set not in {"optimization", "validation", "expected", "annual_expected"}:
        raise ValueError(f"Unsupported scenario set: {scenario_set}")
    generator.floor_hits = 0
    generator.cells_drawn = 0
    totals = []
    ids = []
    if scenario_set in {"optimization", "validation"}:
        for index, seed in enumerate(_seeds(seed_base, scenario_set, n_scenarios), start=1):
            id_scenario = scenario_id(version, regime, scenario_set, index)
            rng = np.random.Generator(np.random.PCG64(seed))
            stop, drop = generator.draw(rng, multiplier)
            totals.append(float((stop * drop).sum() / N_PERIODS))
            write_scenario(generator.to_payload(stop, drop, id_scenario, "simulated"), regime, version, scenario_set, id_scenario)
            ids.append(id_scenario)
    elif scenario_set == "expected":
        id_scenario = scenario_id(version, regime, scenario_set)
        stop, drop = generator.draw(np.random.default_rng(0), multiplier, deterministic="mean")
        write_scenario(generator.to_payload(stop, drop, id_scenario, "expected"), regime, version, scenario_set, id_scenario)
        ids.append(id_scenario)
        totals.append(float((stop * drop).sum() / N_PERIODS))
    else:
        id_scenario = scenario_id(version, regime, scenario_set)
        payload = _annual_expected_payload(generator, regime, version, multiplier)
        write_scenario(payload, regime, version, scenario_set, id_scenario)
        ids.append(id_scenario)
        totals.append(float(sum(pixel["demand"][0] for pixel in payload["pixels"])))

    summary = {
        "schema_version": 1,
        "version": version,
        "regime": regime,
        "scenario_set": scenario_set,
        "scenario_ids": ids,
        "multiplier": multiplier,
        "stop_factor": multiplier**generator.regime_stop_exponent,
        "drop_factor": multiplier**generator.regime_drop_exponent,
        "regime_scaling": {
            "stop_exponent": generator.regime_stop_exponent,
            "drop_exponent": generator.regime_drop_exponent,
        },
        "n_scenarios": len(ids),
        "seed_base": seed_base,
        "seed_scheme": "SeedSequence([seed_base, set_code]).spawn(index); set_code optimization=30, validation=100",
        "shape_params_sha256": params_sha256,
        "period_total_mean": float(np.mean(totals)),
        "period_total_p10": float(np.quantile(totals, 0.10)),
        "period_total_p90": float(np.quantile(totals, 0.90)),
        "stop_floor_hits": generator.floor_hits,
        "stop_cells_drawn": generator.cells_drawn,
        "stop_floor_share": generator.floor_hits / max(generator.cells_drawn, 1),
        "pixels": len(generator.pixels),
        "periods": 1 if scenario_set == "annual_expected" else N_PERIODS,
        "optimization_compatible": scenario_set in {"optimization", "validation", "expected", "annual_expected"},
    }
    logger.info(
        f"[{version}/{regime}/{scenario_set}] {len(ids)} scenarios | mean total {summary['period_total_mean']:,.0f} "
        f"| stop floor hit {summary['stop_floor_share'] * 100:.3f}% of cells"
    )
    return summary


def write_manifest(regime: str, version: str, scenario_set: str, manifest: dict) -> Path:
    """Write a manifest next to the exact set it describes."""
    directory = scenario_dir(regime, version, scenario_set)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"
    with open(path, "w") as file:
        json.dump(manifest, file, indent=2, default=str)
    logger.info(f"Manifest written to {path}")
    return path


def load_generated(
    regime: str,
    version: str = DEFAULT_SCENARIO_VERSION,
    scenario_set: str = "validation",
) -> pd.DataFrame:
    """Load simulated scenarios from a versioned set into a long DataFrame."""
    directory = scenario_dir(regime, version, scenario_set)
    rows = []
    for path in sorted(directory.glob("scenario_*.json")):
        with open(path) as file:
            data = json.load(file)
        if data["type"] != "simulated":
            continue
        for pixel in data["pixels"]:
            for t in range(N_PERIODS):
                rows.append(
                    {
                        "id_scenario": data["id_scenario"],
                        "id_pixel": pixel["id_pixel"],
                        "period": t,
                        "stop": pixel["stop"][t],
                        "drop": pixel["drop"][t],
                        "demand": pixel["demand"][t],
                    }
                )
    return pd.DataFrame(rows)


def shape_params_digest(params: dict) -> str:
    """Stable digest stored in manifests to identify the exact source parameters."""
    encoded = json.dumps(params, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()

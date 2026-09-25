"""The scenario contract: where scenario sets live on disk and how they are read back.

This is the seam between the two sides of the study. `src.scenarios` writes through a
`ScenarioLayout`, `src.optimization` reads through one, and neither imports the other.

Per pixel a scenario carries four fields, arrays of length `N_PERIODS`: `id_pixel`,
`stop`, `drop`, `demand`, with `stop >= 1` and `drop > 0`. The CA only writes cost keys
for `demand > 0` and the models index those keys directly, so one zero pixel-period is
a `KeyError` at solve time. `id_scenario` and `type` are written but identity comes
from the manifest.
"""

import json
from pathlib import Path

from src.core import constants
from src.core.entities import Pixel
from src.core.inputs import get_pixels
from src.tools.logging import get_logger

logger = get_logger("Contract")

SCENARIO_SETS = ("optimization", "validation", "expected", "annual_expected")
SIMULATED_SETS = ("optimization", "validation")
DEPENDENCE_METHODS = ("independent", "spatial_joint", "historical_bootstrap")


def scenario_id(version: str, regime: str, scenario_set: str, index: int | None = None) -> str:
    """Stable, filename-safe identifier; never derived from filesystem ordering."""
    base = f"{version}-{regime}-{scenario_set}"
    return base if index is None else f"{base}-{index:03d}"


def assert_contract(stop, drop) -> None:
    """Hard invariants of every scenario: `stop >= 1` and `drop > 0` everywhere."""
    if stop.min() < 1:
        raise ValueError("stop must be >= 1 for every pixel and period.")
    if drop.min() <= 0:
        raise ValueError("drop must be > 0 for every pixel and period.")


class ScenarioLayout:
    """Directory structure of one scenario version: `<root>/<regime>[/<method>]/<set>/`.

    A plain version holds the four purpose-specific sets per regime. A comparison
    version adds a dependence-method level and holds simulated sets only.
    """

    def __init__(self, root: Path, comparison: bool = False):
        self.root = Path(root)
        self.comparison = comparison

    @classmethod
    def generated(cls, version: str) -> "ScenarioLayout":
        # Read at call time so the location can be redirected (tests, sandboxes).
        return cls(constants.PATH_GENERATED_SCENARIOS / version)

    @classmethod
    def for_comparison(cls, version: str) -> "ScenarioLayout":
        return cls(constants.PATH_COMPARISON_SCENARIOS / version, comparison=True)

    def set_dir(self, regime: str, scenario_set: str, method: str | None = None) -> Path:
        if self.comparison:
            if scenario_set not in SIMULATED_SETS:
                raise ValueError("Comparison scenarios support validation or optimization sets only.")
            if method not in DEPENDENCE_METHODS:
                raise ValueError(f"Unknown comparison method {method!r}; expected one of {DEPENDENCE_METHODS}.")
            return self.root / regime / method / scenario_set
        if scenario_set not in SCENARIO_SETS:
            raise ValueError(f"Unknown scenario set {scenario_set!r}; expected one of {SCENARIO_SETS}.")
        if method is not None:
            raise ValueError("A plain scenario version has no dependence-method level.")
        return self.root / regime / scenario_set

    def scenario_file(self, regime: str, scenario_set: str, id_scenario: str, method: str | None = None) -> Path:
        return self.set_dir(regime, scenario_set, method) / f"scenario_{id_scenario}.json"

    def set_manifest(self, regime: str, scenario_set: str, method: str | None = None) -> Path:
        return self.set_dir(regime, scenario_set, method) / "manifest.json"

    def scenario_ids(self, regime: str, scenario_set: str, n_scenarios: int) -> list[str]:
        """Canonical scenario ids from the set manifest, never from filename order."""
        directory = self.set_dir(regime, scenario_set)
        with open(directory / "manifest.json") as file:
            manifest = json.load(file)
        if not manifest["optimization_compatible"]:
            raise ValueError(f"Scenario set {directory} is not compatible with the 12-period optimizer.")
        ids = manifest["scenario_ids"]
        if n_scenarios > len(ids):
            raise ValueError(f"Requested {n_scenarios} scenarios but {directory} contains {len(ids)}.")
        return ids[:n_scenarios]

    def load_pixels(self, id_scenario: str, regime: str, scenario_set: str) -> dict[str, Pixel]:
        """Pixels of one scenario, joined with the grid of `input_pixels.xlsx`.

        Only pixels present in both are returned. A pixel in the scenario but missing
        from the grid is dropped, so the mismatch is logged rather than passing silently.
        """
        pixels = get_pixels()
        path = self.scenario_file(regime, scenario_set, id_scenario)
        if not path.exists():
            logger.error(f"Scenario file {path} not found.")
            raise FileNotFoundError(f"Scenario file {path} not found.")

        with open(path, "r") as file:
            data = json.load(file)

        unknown = []
        for pixel_data in data["pixels"]:
            id_pixel = pixel_data["id_pixel"]
            if id_pixel in pixels:
                pixels[id_pixel].set_scenario_data(
                    demand_by_period=pixel_data["demand"],
                    drop_by_period=pixel_data["drop"],
                    stop_by_period=pixel_data["stop"],
                )
            else:
                unknown.append(id_pixel)

        if unknown:
            logger.warning(f"{len(unknown)} pixels in {path.name} are absent from the grid and were dropped: {unknown[:5]}")

        available = {i: p for i, p in pixels.items() if p.is_available}
        missing = len(pixels) - len(available)
        if missing:
            logger.warning(f"{missing} grid pixels have no data in {path.name} and were excluded.")

        logger.info(f"Scenario {id_scenario} ({regime}/{scenario_set}): {len(available)} pixels loaded.")
        return available

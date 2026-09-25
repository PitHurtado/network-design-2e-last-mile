"""The tariff table: OPEX and installation cost per capacity level (`data/raw_facility/tariffs.json`)."""

import json
from dataclasses import dataclass
from pathlib import Path

from src.tools.paths import DATA_DIR

PATH_TARIFFS = DATA_DIR / "raw_facility" / "tariffs.json"


@dataclass(frozen=True)
class Tariffs:
    """Known costs per level, and how to go beyond the largest known level.

    * `opex_extrapolation = "last_step"`: each extra `level_step` vehicles adds the increment
      between the two largest known levels (OLD: +879 per 2 vehicles above 12).
    * `installation_extrapolation = "plateau"`: above the largest known level, installation
      costs what the largest one does.
    """

    opex: dict[int, float]
    installation: dict[int, float]
    level_step: int = 2
    opex_extrapolation: str = "last_step"
    installation_extrapolation: str = "plateau"
    alpha_fixed: float = 0.70

    @classmethod
    def load(cls, path: Path = PATH_TARIFFS) -> "Tariffs":
        raw = json.loads(Path(path).read_text())
        return cls(
            opex={int(k): float(v) for k, v in raw["opex"].items()},
            installation={int(k): float(v) for k, v in raw["installation"].items()},
            level_step=int(raw["level_step"]),
            opex_extrapolation=raw["opex_extrapolation"],
            installation_extrapolation=raw["installation_extrapolation"],
            alpha_fixed=float(raw["alpha_fixed"]),
        )

    def opex_of(self, level: int) -> float:
        if level == 0:
            return 0.0
        if level in self.opex:
            return self.opex[level]
        known = sorted(self.opex)
        if level < known[0] or self.opex_extrapolation != "last_step":
            raise ValueError(f"No OPEX for level {level} and no extrapolation rule covers it.")
        step = self.opex[known[-1]] - self.opex[known[-2]]
        return self.opex[known[-1]] + step * (level - known[-1]) / self.level_step

    def installation_of(self, level: int) -> float:
        if level == 0:
            return 0.0
        if level in self.installation:
            return self.installation[level]
        known = sorted(self.installation)
        if level < known[0] or self.installation_extrapolation != "plateau":
            raise ValueError(f"No installation cost for level {level} and no extrapolation rule covers it.")
        return self.installation[known[-1]]

"""`CapacityTable`: the levels and costs of every satellite, as the models read them.

It is the content of a facilities artifact (`data/facilities/f<N>/capacity.json`), and
`InstanceBuilder` applies it to the facilities loaded from `input_facilities.xlsx`, whose
location and sourcing cost stay; levels and costs come only from here.
"""

from dataclasses import dataclass
from pathlib import Path

from src.core.entities import Facility
from src.tools.io import read_json, write_json

CAPACITY_FILE = "capacity.json"


@dataclass(frozen=True)
class SatelliteCapacity:
    levels: list[int]  # ascending, starting with 0 (not installed)
    installation: dict[int, float]
    operation: dict[int, list[float]]  # one cost per period


@dataclass(frozen=True)
class CapacityTable:
    satellites: dict[str, SatelliteCapacity]

    @classmethod
    def load(cls, path: Path) -> "CapacityTable":
        raw = read_json(Path(path) / CAPACITY_FILE if Path(path).is_dir() else path)
        return cls(
            {
                facility_id: SatelliteCapacity(
                    levels=[int(q) for q in block["levels"]],
                    installation={int(q): float(v) for q, v in block["cost_installation"].items()},
                    operation={int(q): [float(x) for x in v] for q, v in block["cost_operation"].items()},
                )
                for facility_id, block in raw["satellites"].items()
            }
        )

    def apply(self, facilities: dict[str, Facility]) -> dict[str, Facility]:
        """Replace levels and costs of `facilities` in place; every facility must be in the table."""
        missing = set(facilities) - set(self.satellites)
        if missing:
            raise ValueError(f"The capacity table has no entry for facilities {sorted(missing)}.")
        for facility_id, facility in facilities.items():
            block = self.satellites[facility_id]
            facility.capacity = {str(q): q for q in block.levels}
            facility.cost_installation = {str(q): block.installation[q] for q in block.levels}
            facility.cost_operation = {str(q): list(block.operation[q]) for q in block.levels}
        return facilities


def save_table(path: Path, satellites: dict[str, dict]) -> Path:
    return write_json(Path(path) / CAPACITY_FILE, {"schema_version": 1, "satellites": satellites})

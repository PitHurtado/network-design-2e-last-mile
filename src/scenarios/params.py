"""`ShapeParams`: the fitted shape parameters, the single reproducible input of scenario generation.

A read-only mapping over the persisted JSON. It wraps the raw dict instead of parsing it
into fields, so that saving what was loaded reproduces the file byte for byte.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from src.tools.io import read_json, sha256_json, write_json


@dataclass(frozen=True)
class ShapeParams(Mapping):
    """Pixel levels, layer seasonality, dispersion, spatial decay and regime multipliers."""

    raw: dict = field(repr=False)

    @classmethod
    def load(cls, path: Path) -> "ShapeParams":
        return cls(read_json(path))

    def save(self, path: Path) -> Path:
        return write_json(path, self.raw)

    # Mapping protocol: params["spatial"]["rho_km"] keeps working everywhere.
    def __getitem__(self, key):
        return self.raw[key]

    def __iter__(self):
        return iter(self.raw)

    def __len__(self):
        return len(self.raw)

    @property
    def sha256(self) -> str:
        """Digest recorded by every scenario set generated from these parameters."""
        return sha256_json(self.raw)

    @property
    def pixels(self) -> list[str]:
        return list(self.raw["pixels"])

    @property
    def spatial(self) -> dict:
        return self.raw["spatial"]

    @property
    def regime_scaling(self) -> dict:
        return self.raw["regime_scaling"]

    def regime(self, name: str) -> dict:
        return self.raw["regimes"][name]

    def multiplier(self, regime: str) -> float:
        return self.raw["regimes"][regime]["multiplier"]

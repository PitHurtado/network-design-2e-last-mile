"""Writing and reading back versioned scenario sets.

One writer serves both kinds of version the `ScenarioLayout` knows: a plain version
(four purpose-specific sets per regime, the optimizer's input) and a comparison version
(simulated validation sets per dependence method, paired across methods by seed).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.core.constants import N_PERIODS, SEED_BASE
from src.core.contract import SCENARIO_SETS, ScenarioLayout, scenario_id
from src.scenarios.generation.dependence import MeanShocks
from src.scenarios.generation.generator import ScenarioGenerator
from src.scenarios.generation.seeds import SEED_SCHEME, rng_for, set_seeds
from src.tools.io import read_json, write_json
from src.tools.logging import get_logger

logger = get_logger("Generate")

COMPARISON_SEED_SCHEME = "SeedSequence([seed_base, validation_code=100]).spawn(index); paired across methods"


@dataclass(frozen=True)
class SetSpec:
    """One scenario set to generate: which regime, which purpose, how many, at what level."""

    regime: str
    scenario_set: str
    n_scenarios: int
    multiplier: float
    version: str
    seed_base: int = SEED_BASE
    method: str | None = None  # comparison versions only


class ScenarioSetWriter:
    """Generates scenario sets into a `ScenarioLayout` and loads them back."""

    def __init__(self, layout: ScenarioLayout):
        self.layout = layout

    # ── writing ───────────────────────────────────────────────────────────────

    def _write_scenario(self, spec: SetSpec, payload: dict, id_scenario: str) -> None:
        write_json(self.layout.scenario_file(spec.regime, spec.scenario_set, id_scenario, spec.method), payload)

    def generate(self, generator: ScenarioGenerator, spec: SetSpec, params_sha256: str | None = None) -> dict:
        """Write every scenario of `spec` and return the set summary (the manifest body)."""
        if self.layout.comparison:
            return self._generate_comparison(generator, spec, params_sha256)
        if spec.scenario_set not in SCENARIO_SETS:
            raise ValueError(f"Unsupported scenario set: {spec.scenario_set}")
        generator.floor_hits = 0
        generator.cells_drawn = 0
        totals, ids = [], []
        if spec.scenario_set in {"optimization", "validation"}:
            for index, seed in enumerate(set_seeds(spec.seed_base, spec.scenario_set, spec.n_scenarios), start=1):
                id_scenario = scenario_id(spec.version, spec.regime, spec.scenario_set, index)
                stop, drop = generator.draw(rng_for(seed), spec.multiplier)
                totals.append(float((stop * drop).sum() / N_PERIODS))
                self._write_scenario(spec, generator.to_payload(stop, drop, id_scenario, "simulated"), id_scenario)
                ids.append(id_scenario)
        elif spec.scenario_set == "expected":
            id_scenario = scenario_id(spec.version, spec.regime, spec.scenario_set)
            stop, drop = generator.draw(np.random.default_rng(0), spec.multiplier, shocks=MeanShocks())
            self._write_scenario(spec, generator.to_payload(stop, drop, id_scenario, "expected"), id_scenario)
            ids.append(id_scenario)
            totals.append(float((stop * drop).sum() / N_PERIODS))
        else:
            id_scenario = scenario_id(spec.version, spec.regime, spec.scenario_set)
            payload = generator.annual_expected_payload(id_scenario, spec.multiplier)
            self._write_scenario(spec, payload, id_scenario)
            ids.append(id_scenario)
            totals.append(float(sum(pixel["demand"][0] for pixel in payload["pixels"])))

        summary = {
            "schema_version": 1,
            "version": spec.version,
            "regime": spec.regime,
            "scenario_set": spec.scenario_set,
            "scenario_ids": ids,
            "multiplier": spec.multiplier,
            "stop_factor": spec.multiplier**generator.regime_stop_exponent,
            "drop_factor": spec.multiplier**generator.regime_drop_exponent,
            "regime_scaling": {
                "stop_exponent": generator.regime_stop_exponent,
                "drop_exponent": generator.regime_drop_exponent,
            },
            "n_scenarios": len(ids),
            "seed_base": spec.seed_base,
            "seed_scheme": SEED_SCHEME,
            "shape_params_sha256": params_sha256,
            **self._totals(totals, generator),
            "periods": 1 if spec.scenario_set == "annual_expected" else N_PERIODS,
            "optimization_compatible": spec.scenario_set in SCENARIO_SETS,
        }
        logger.info(
            f"[{spec.version}/{spec.regime}/{spec.scenario_set}] {len(ids)} scenarios | mean total "
            f"{summary['period_total_mean']:,.0f} | stop floor hit {summary['stop_floor_share'] * 100:.3f}% of cells"
        )
        return summary

    def _generate_comparison(self, generator: ScenarioGenerator, spec: SetSpec, params_sha256: str | None) -> dict:
        """A validation set for one dependence method.

        Every method consumes exactly the same SeedSequence streams, so scenario `i`
        is a paired comparison across methods.
        """
        if spec.method != generator.dependence.name:
            raise ValueError(f"Generator dependence {generator.dependence.name!r} does not match method={spec.method!r}")
        generator.floor_hits = 0
        generator.cells_drawn = 0
        totals, ids = [], []
        for index, seed in enumerate(set_seeds(spec.seed_base, "validation", spec.n_scenarios), start=1):
            id_scenario = scenario_id(spec.version, spec.regime, "validation", index)
            stop, drop = generator.draw(rng_for(seed), spec.multiplier)
            totals.append(float((stop * drop).sum() / N_PERIODS))
            self._write_scenario(spec, generator.to_payload(stop, drop, id_scenario, "simulated"), id_scenario)
            ids.append(id_scenario)

        return {
            "schema_version": 1,
            "version": spec.version,
            "regime": spec.regime,
            "scenario_set": "validation",
            "method": spec.method,
            "scenario_ids": ids,
            "multiplier": spec.multiplier,
            "n_scenarios": len(ids),
            "seed_base": spec.seed_base,
            "seed_scheme": COMPARISON_SEED_SCHEME,
            "shape_params_sha256": params_sha256,
            **self._totals(totals, generator),
            "periods": N_PERIODS,
        }

    @staticmethod
    def _totals(totals: list[float], generator: ScenarioGenerator) -> dict:
        return {
            "period_total_mean": float(np.mean(totals)),
            "period_total_p10": float(np.quantile(totals, 0.10)),
            "period_total_p90": float(np.quantile(totals, 0.90)),
            "stop_floor_hits": generator.floor_hits,
            "stop_cells_drawn": generator.cells_drawn,
            "stop_floor_share": generator.floor_hits / max(generator.cells_drawn, 1),
            "pixels": len(generator.pixels),
        }

    def write_manifest(self, regime: str, scenario_set: str, manifest: dict, method: str | None = None):
        """Write a manifest next to the exact set it describes."""
        path = write_json(self.layout.set_manifest(regime, scenario_set, method), manifest, default=str)
        logger.info(f"Manifest written to {path}")
        return path

    # ── reading ───────────────────────────────────────────────────────────────

    def load_long(self, regime: str, scenario_set: str = "validation", method: str | None = None) -> pd.DataFrame:
        """One row per (scenario, pixel, period).

        Plain versions return only the simulated scenarios; comparison versions tag each
        row with its method and regime.
        """
        directory = self.layout.set_dir(regime, scenario_set, method)
        rows = []
        for path in sorted(directory.glob("scenario_*.json")):
            data = read_json(path)
            if not self.layout.comparison and data["type"] != "simulated":
                continue
            tag = {"method": method, "regime": regime} if self.layout.comparison else {}
            for pixel in data["pixels"]:
                for t in range(N_PERIODS):
                    rows.append(
                        {
                            **tag,
                            "id_scenario": data["id_scenario"],
                            "id_pixel": pixel["id_pixel"],
                            "period": t,
                            "stop": pixel["stop"][t],
                            "drop": pixel["drop"][t],
                            "demand": pixel["demand"][t],
                        }
                    )
        return pd.DataFrame(rows)

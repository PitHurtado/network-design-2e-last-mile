"""Main driver for the Capacitated SAA Model (Proposal-A levels + opening costs)."""

import json
import logging
from pathlib import Path
from typing import Optional

from src.constants import TypeOfFlexibility
from src.models.capacitated_saa_model import CapacitatedSAAModel
from src.utils.custom_logger import get_logger
from src.utils.instance import Instance

logger = get_logger("MainCapacitatedSAA")

# Proposal-A capacity levels per satellite (from fleet analysis).
# Level 0 (closed) is always added automatically by the model.
PROPOSAL_A_LEVELS: dict[str, list[int]] = {
    "ABAROA":           [6, 8, 10],
    "ACHACHICALA":      [4, 6, 8, 10],
    "COTA_COTA":        [8, 10, 12, 14],
    "LLOJETA":          [4, 6, 8],
    "LOS_PINOS":        [4, 6, 8, 10],
    "MALLASA":          [2, 4],
    "PERIFERICA":       [12, 14, 16, 18],
    "SOPOCACHI":        [14, 16, 18, 20],
    "ZONA_CEMENTERIO":  [8, 10, 12, 14],
}


class MainCapacitated:
    """Solve the Capacitated SAA Model for a given satellite subset."""

    def __init__(
        self,
        id_instance: str,
        folder_path: Path,
        configuration: tuple,
        max_run_time: int,
        is_evaluation: bool = False,
        id_sampling: Optional[int] = None,
        use_euclidean_distance: bool = False,
        facilities_subset: Optional[list] = None,
    ):
        self.folder_path = folder_path
        self.configuration = configuration
        self.max_run_time = max_run_time

        N, is_continuous_var_x = configuration

        self.instance = Instance(
            id_instance=id_instance,
            is_continuous_var_x=is_continuous_var_x,
            type_of_flexibility=TypeOfFlexibility.FIXED_CAPACITY.value,
            periods=12,
            N=N,
            is_evaluation=is_evaluation,
            id_sampling=id_sampling,
            use_euclidean_distance=use_euclidean_distance,
            facilities_subset=facilities_subset,
        )
        logger.info(f"Instance loaded: {id_instance}")

    def solve(self) -> Path:
        """Solve and save results to JSON."""
        logger.info(f"Solving capacitated SAA for: {self.instance.id_instance}")

        solver = CapacitatedSAAModel(
            instance=self.instance,
            capacity_levels=PROPOSAL_A_LEVELS,
        )
        solver.build()
        solver.set_params({"TimeLimit": self.max_run_time})

        logger.disabled = True
        logging.disable(logging.CRITICAL)
        solver_metrics = solver.solve()
        logger.disabled = False
        logging.disable(logging.NOTSET)
        logger.info("Solving ended")

        results = self._build_results(solver, solver_metrics)

        out = self.folder_path / f"capacitated_{self.instance.id_instance}_{self.instance.id_sampling}.json"
        with open(out, "w") as f:
            f.write(json.dumps(results, indent=4))

        logger.info(f"Results saved → {out.name}")
        return out

    # ------------------------------------------------------------------
    # Result builders
    # ------------------------------------------------------------------
    def _build_results(self, solver: CapacitatedSAAModel, metrics: dict) -> dict:
        N, is_continuous_var_x = self.configuration
        return {
            # Objective decomposition
            "objective":                     solver.obj.cost_total.getValue(),
            "cost_installation":             solver.obj.cost_installation.getValue(),
            "cost_served_from_facilities":   solver.obj.cost_served_from_facilities.getValue(),
            "cost_served_from_dc":           solver.obj.cost_served_from_dc.getValue(),
            # Scenario info
            "scenarios":     self.instance.scenarios_ids,
            # Solver metadata
            "solver_info":   metrics,
            # Decision variables
            "Y": {
                str(key): int(round(var.X))
                for key, var in solver.model._Y.items()
            },
            "X": {
                str(key): round(var.X, 6)
                for key, var in solver.model._X.items()
                if var.X > 1e-6
            },
            "W": {
                str(key): round(var.X, 6)
                for key, var in solver.model._W.items()
                if var.X > 1e-6
            },
            # Chosen capacity per facility
            "chosen_capacity": self._chosen_capacity(solver),
            # Enrichment data
            "pixel_info":            self._build_pixel_info(),
            "facility_info":         self._build_facility_info(),
            "cost_serving":          self._build_cost_serving(),
            "fleet_size_serving":    self._build_fleet_size_serving(),
            # Run config
            "id_sampling":           self.instance.id_sampling,
            "configuration":         self.instance.id_instance,
            "N":                     N,
            "is_continuous_x":       is_continuous_var_x,
            "periods":               12,
            "max_run_time":          self.max_run_time,
            "capacity_proposal":     "A",
        }

    def _chosen_capacity(self, solver: CapacitatedSAAModel) -> dict:
        """Extract which capacity level was chosen per facility."""
        chosen = {}
        for i in self.instance.facilities:
            for q, (vehicles, _) in solver._cap[i].items():
                if round(solver.vars.Y[(i, q)].X) == 1:
                    chosen[i] = {"level": q, "vehicles": vehicles}
                    break
        return chosen

    def _build_pixel_info(self) -> dict:
        first_sc = next(iter(self.instance.scenarios.values()))
        return {
            k: {
                "lon":              px.geo_point.lon,
                "lat":              px.geo_point.lat,
                "layer":            k.split("-", 1)[0],
                "pixel_id":         k.split("-", 1)[1],
                "demand_by_period": px.demand_by_period,
            }
            for k, px in first_sc.pixels.items()
        }

    def _build_facility_info(self) -> dict:
        return {
            i: {"lon": f.geo_point.lon, "lat": f.geo_point.lat}
            for i, f in self.instance.facilities.items()
        }

    def _build_cost_serving(self) -> dict:
        cf, cd = {}, {}
        for n, sc in self.instance.scenarios.items():
            for k, v in sc.get_cost_serving("facility").items():
                cf[str(k)] = v
            for k, v in sc.get_cost_serving("dc").items():
                cd[str(k)] = v
        return {"facility": cf, "dc": cd}

    def _build_fleet_size_serving(self) -> dict:
        ff, fd = {}, {}
        for n, sc in self.instance.scenarios.items():
            for k, v in sc.get_fleet_size("facility").items():
                ff[str(k)] = v
            for k, v in sc.get_fleet_size("dc").items():
                fd[str(k)] = v
        return {"facility": ff, "dc": fd}

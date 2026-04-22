"""Module of class Main Uncapacitated SAA Model."""

import json
import logging
from typing import Optional

from src.constants import RESULTS_DIR, TypeOfFlexibility
from src.models.uncapacitated_saa_model import UncapacitatedSAAModel
from src.utils.custom_logger import get_logger
from src.utils.instance import Instance

logger = get_logger("Main Uncapacitated SAA")


class Main:
    """Main class to solve Uncapacitated SAA Model."""

    def __init__(
        self,
        id_instance: str,
        folder_path,
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

        logger.info(f"Loading instance: {configuration}")
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
        logger.info(f"Instance Loaded: {configuration}")

    def solve(self):
        """Solve Uncapacitated SAA Model and save results to JSON."""
        logger.info(f"Starting solving Uncapacitated SAA Model for instance {self.instance.id_instance}")

        # (1) Build and solve model
        solver = UncapacitatedSAAModel(self.instance)
        solver.build()
        solver.set_params({"TimeLimit": self.max_run_time})

        logger.disabled = True
        logging.disable(logging.CRITICAL)
        solver_metrics = solver.solve()
        logger.disabled = False
        logging.disable(logging.NOTSET)
        logger.info("Solving ended")

        # (2) Extract solution variables
        results = {
            "objective": solver.obj.cost_total.getValue(),
            "cost_served_from_facilities": solver.obj.cost_served_from_facilities.getValue(),
            "cost_served_from_dc": solver.obj.cost_served_from_dc.getValue(),
            "scenarios": self.instance.scenarios_ids,
            "Solver information": solver_metrics,
            "X": {str(keys): value.X for keys, value in solver.model._X.items()},
            "W": {str(keys): value.X for keys, value in solver.model._W.items()},
        }

        # (3) Enrich with pixel and facility data for visualization
        results["pixel_info"] = self._build_pixel_info()
        results["facility_info"] = self._build_facility_info()
        results["cost_serving"] = self._build_cost_serving()
        results["fleet_size_serving"] = self._build_fleet_size_serving()

        results.update(self._get_configuration_info())

        # (4) Save to JSON
        path_file_output = (
            self.folder_path / f"uncapacitated_{self.instance.id_instance}_{self.instance.id_sampling}.json"
        )
        with open(path_file_output, "w") as file:
            file.write(json.dumps(results, indent=4))

        logger.info(f"Results saved in {path_file_output}")
        return path_file_output

    def _build_pixel_info(self) -> dict:
        """Build pixel metadata dict for visualization."""
        first_scenario = next(iter(self.instance.scenarios.values()))
        pixel_info = {}
        for k, pixel in first_scenario.pixels.items():
            layer, pixel_id = k.split("-", 1)
            pixel_info[k] = {
                "lon": pixel.geo_point.lon,
                "lat": pixel.geo_point.lat,
                "layer": layer,
                "pixel_id": pixel_id,
                "demand_by_period": pixel.demand_by_period,
            }
        return pixel_info

    def _build_facility_info(self) -> dict:
        """Build facility metadata dict for visualization."""
        return {
            i: {"lon": facility.geo_point.lon, "lat": facility.geo_point.lat}
            for i, facility in self.instance.facilities.items()
        }

    def _build_cost_serving(self) -> dict:
        """Build cost serving dicts (facility and dc) across all scenarios."""
        cost_facility = {}
        cost_dc = {}
        for n, scenario in self.instance.scenarios.items():
            for key, val in scenario.get_cost_serving("facility").items():
                cost_facility[str(key)] = val
            for key, val in scenario.get_cost_serving("dc").items():
                cost_dc[str(key)] = val
        return {"facility": cost_facility, "dc": cost_dc}

    def _build_fleet_size_serving(self) -> dict:
        """Build fleet size serving dicts (facility and dc) across all scenarios."""
        fleet_facility = {}
        fleet_dc = {}
        for n, scenario in self.instance.scenarios.items():
            for key, val in scenario.get_fleet_size("facility").items():
                fleet_facility[str(key)] = val
            for key, val in scenario.get_fleet_size("dc").items():
                fleet_dc[str(key)] = val
        return {"facility": fleet_facility, "dc": fleet_dc}

    def _get_configuration_info(self) -> dict:
        N, is_continuous_var_x = self.configuration
        return {
            "id_sampling": self.instance.id_sampling,
            "configuration": self.instance.id_instance,
            "N": N,
            "is_continuous_x": is_continuous_var_x,
            "periods": 12,
            "max_run_time": self.max_run_time,
        }

"""Module of class Instance."""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.constants import PATH_SAMPLING_SCENARIO
from src.data.etl import get_facilities, get_scenario, get_vehicles
from src.routing_tools.continuous_approximation import ContinuousApproximation
from src.utils.classes import Facility, Pixel, Vehicle
from src.utils.custom_logger import get_logger
from src.utils.scenario import Scenario

logger = get_logger("Instance")


@dataclass
class ConfigurationInstance:
    """Configuration for the instance."""

    is_continuous_var_x: bool
    type_of_flexibility: str
    N: int
    is_evaluation: bool
    sampling_id: Optional[int]


class Instance:
    """Class to define Instance"""

    def __init__(
        self,
        id_instance: str,
        is_continuous_var_x: bool,
        type_of_flexibility: str,
        periods: int,
        N: int,
        is_evaluation: bool,
        sampling_id: Optional[int] = None,
    ):  # pylint: disable=too-many-arguments

        self.id_instance = id_instance
        self.config = ConfigurationInstance(
            is_continuous_var_x=is_continuous_var_x,
            type_of_flexibility=type_of_flexibility,
            N=N,
            is_evaluation=is_evaluation,
            sampling_id=sampling_id,
        )
        self.periods = periods

        # Inputs
        self.matrix_facilities_pixels = dict()
        self.matrix_dc_pixels = dict()

        # Read vehicles and facilities
        self.vehicles: Dict[str, Vehicle] = self.__read_vehicles()
        self.facilities: Dict[str, Facility] = self.__read_facilities()

        # Read the demand (pixels) of each scenario
        self.scenarios: Dict[str, Scenario] = self.__read_scenarios()
        self.scenarios_ids = list(key for key in self.scenarios)

        # compute continuous approximation parameters
        self.__compute_continuous_approximation()

    def __str__(self):
        return (
            f"---- Instance ----\n"
            f"ID of the instance: {self.id_instance}\n"
            f"ID scenario sample: {self.scenarios_ids}\n"
            f"Is continuous X: {self.config.is_continuous_var_x}\n"
            f"Type of flexibility: {self.config.type_of_flexibility}\n"
            f"Periods: {self.periods}\n"
            f"N: {self.config.N}\n"
            f"Is evaluation: {self.config.is_evaluation}\n"
            f"Quantity of facilities: {len(self.facilities)}\n"
            f"Quantity of vehicles: {len(self.vehicles)}\n"
            f"Quantity of scenarios: {len(self.scenarios)}\n"
            f"-----------------"
        )

    def __get_scenarios_sample(self) -> List[str]:
        """Get the scenarios for sample."""
        id_scenarios_sample = []
        if self.config.is_evaluation:
            path_json = PATH_SAMPLING_SCENARIO / "evaluation.json"
            if os.path.exists(path_json):
                with open(path_json, "r") as file:
                    data = json.load(file)
                    id_scenarios_sample = data["id_scenarios_sample"]
        elif self.config.sampling_id is not None:
            path_json = PATH_SAMPLING_SCENARIO / f"sampling_{self.config.sampling_id}.json"
            if os.path.exists(path_json):
                with open(path_json, "r") as file:
                    data = json.load(file)
                    id_scenarios_sample = data["id_scenarios_sample"]
        else:
            id_scenarios_sample = list(map(str, range(1, self.config.N + 1)))  # random.sample(range(500), self.N)
        return id_scenarios_sample

    def __read_vehicles(self) -> Dict[str, Vehicle]:
        """Reads the vehicles from the file."""
        try:
            vehicles = get_vehicles()
        except FileNotFoundError as error:
            logger.error(f"Not Found Vehicles: {error}")
            raise error
        return vehicles

    def __read_facilities(self) -> Dict[str, Facility]:
        """Reads the facilities from the file."""
        try:
            facilities = get_facilities()
        except FileNotFoundError as error:
            logger.error(f"Not Found Facilities: {error}")
            raise error
        return facilities

    def __read_pixels(self, id_scenario: str) -> Dict[str, Pixel]:
        """Reads the pixels from the file."""
        try:
            pixels = get_scenario(id_scenario=id_scenario)
        except FileNotFoundError as error:
            logger.error(f"Not Found Scenario: {error}")
            raise error
        return pixels

    def __read_scenarios(self) -> Dict[str, Scenario]:
        """Reads the scenarios."""
        id_scenarios_sample = self.__get_scenarios_sample()

        logger.info(f"Scenarios sample: {id_scenarios_sample}")
        scenarios = {}
        for id_scenario in id_scenarios_sample:
            logger.info(f"------- Reading scenario: {id_scenario}")
            pixels = self.__read_pixels(id_scenario)
            scenario = Scenario(
                id_scenario=id_scenario,
                pixels=pixels,
                periods=self.periods,
            )
            scenarios[str(id_scenario)] = scenario
        return scenarios

    def __compute_continuous_approximation(self) -> None:
        """Compute the continuous approximation parameters."""
        ca = ContinuousApproximation(
            facilities=self.facilities,
            vehicles=self.vehicles,
            scenarios=self.scenarios,
        )
        scenarios_with_parameters = ca.run_continuous_approximation()
        self.scenarios = scenarios_with_parameters

    def get_info(self) -> Dict:
        """Get the information of the instance."""
        return {
            "id_instance": self.id_instance,
            "is_continuous_var_x": self.config.is_continuous_var_x,
            "type_of_flexibility": self.config.type_of_flexibility,
            "periods": self.periods,
            "N": self.config.N,
            "is_evaluation": self.config.is_evaluation,
            "quantity_facilities": len(self.facilities),
            "quantity_vehicles": len(self.vehicles),
            "quantity_scenarios": len(self.scenarios),
        }


if __name__ == "__main__":
    logger.info("Testing Instance Class")
    instance = Instance(
        id_instance="1",
        is_continuous_var_x=True,
        type_of_flexibility="Flex-Capacity",
        periods=12,
        N=1,
        is_evaluation=True,
    )
    logger.info(instance)
    for scenario_id, scenario in instance.scenarios.items():
        logger.info(scenario.get_info())

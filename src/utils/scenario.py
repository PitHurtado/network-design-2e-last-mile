"""Module for the Scenario class."""

from typing import Any, Dict, Optional

from src.utils.classes import Pixel


class Scenario:
    """Class for a scenario."""

    def __init__(
        self,
        id_scenario: int,
        pixels: Dict[str, Pixel],
        periods: int,
        costs: Optional[Dict[str, Dict]] = None,
        fleet_size: Optional[Dict[str, Dict]] = None,
    ):  # pylint: disable=too-many-arguments
        self.id_scenario = id_scenario
        self.pixels = pixels
        self.costs = costs
        self.fleet_size = fleet_size
        self.periods = periods
        self.parameters = None

    def __str__(self):
        """Return a string representation of the scenario."""
        return f"Scenario {self.id_scenario} with {len(self.pixels)} pixels and {self.periods} periods"  # pylint: disable=line-too-long # noqa:E501

    def get_info(self) -> Dict[str, Any]:
        """Get the information of the scenario."""
        return {
            "costs": self.costs,
            "fleet_size": self.fleet_size,
            "periods": self.periods,
        }

    def set_fleet_size(self, fleet_size: Dict[str, Dict]) -> None:
        """Set the fleet size for the scenario."""
        self.fleet_size = fleet_size

    def set_costs(self, costs: Dict[str, Dict]) -> None:
        """Set the costs for the scenario."""
        self.costs = costs

    def set_parameters(self, parameters: Dict[str, Dict]) -> None:
        """Set the continuous approximation parameters for the scenario."""
        self.parameters = parameters

    def get_fleet_size(self, echelon: str = "facility") -> Dict[Any, float]:
        """Return the fleet size required for a given echelon and period
        ----
        Params:
        - echelon: str
            could be "dc" or "facility"
        ----
        Returns:
        - Dict[Any, float]
            the fleet size required for a given echelon
        """
        return self.fleet_size[echelon]

    def get_cost_serving(self, echelon: str = "facility") -> Dict[Any, float]:
        """Return the costs of serving for a given echelon and period
        ----
        Params:
        - echelon: str
            could be "dc" or "facility"
        ----
        Returns:
        - Dict[Any, float]
            the total costs for a given echelon
        """
        return self.costs[echelon]

    def get_periods(self) -> int:
        """Return the number of periods"""
        return self.periods

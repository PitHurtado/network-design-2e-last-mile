"""Module for the Scenario class."""

from typing import Any, Dict

from src.utils.classes import Pixel


class Scenario:
    """Class for a scenario."""

    def __init__(
        self,
        id_scenario: int,
        pixels: Dict[str, Pixel],
        costs: Dict[str, Dict],
        fleet_size: Dict[str, Dict],
        periods: int,
    ):  # pylint: disable=too-many-arguments
        self.id_scenario = id_scenario
        self.pixels = pixels
        self.costs = costs
        self.fleet_size = fleet_size
        self.periods = periods

    def __str__(self):
        """Return a string representation of the scenario."""
        return f"Scenario {self.id_scenario} with {len(self.pixels)} pixels and {self.periods} periods"  # pylint: disable=line-too-long # noqa:E501

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

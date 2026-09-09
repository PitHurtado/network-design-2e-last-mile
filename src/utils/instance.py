"""Module of class Instance."""

from dataclasses import dataclass
from typing import Dict, List, Optional

from src.constants import N_PERIODS
from src.data.etl import get_facilities, get_scenario, get_vehicles
from src.routing_tools.continuous_approximation import ContinuousApproximation
from src.utils.classes import Facility, Vehicle
from src.utils.custom_logger import get_logger
from src.utils.scenario import Scenario

logger = get_logger("Instance")


@dataclass
class ConfigurationInstance:
    """Configuration for the instance."""

    is_continuous_var_x: bool
    type_of_flexibility: str
    N: int
    regime: str


class Instance:
    """Class to define an instance of the two-echelon problem.

    Scenario selection is explicit: `N` scenarios of the given `regime`, or the ids
    passed in `scenario_ids`. The previous version had three selection branches in
    which a missing sampling file left the scenario list empty, and the objective
    then divided by zero; here a missing scenario file raises.
    """

    def __init__(
        self,
        id_instance: str,
        is_continuous_var_x: bool,
        type_of_flexibility: str,
        N: int,
        regime: str = "normal",
        periods: int = N_PERIODS,
        scenario_ids: Optional[List[str]] = None,
        use_euclidean_distance: bool = False,
        facilities_subset: Optional[List[str]] = None,
    ):  # pylint: disable=too-many-arguments
        self.id_instance = id_instance
        self.regime = regime
        self.use_euclidean_distance = use_euclidean_distance
        self.config = ConfigurationInstance(
            is_continuous_var_x=is_continuous_var_x,
            type_of_flexibility=type_of_flexibility,
            N=N,
            regime=regime,
        )

        if periods != N_PERIODS:
            raise ValueError(
                f"periods={periods} but the Continuous Approximation and the scenario contract are fixed at "
                f"{N_PERIODS}. Changing it silently truncates or overruns the per-period arrays."
            )
        self.periods = periods

        self.vehicles: Dict[str, Vehicle] = get_vehicles()
        self.facilities: Dict[str, Facility] = get_facilities()
        if facilities_subset is not None:
            self.facilities = {k: v for k, v in self.facilities.items() if k in facilities_subset}

        self.scenario_ids_requested = scenario_ids or [str(i) for i in range(1, N + 1)]
        self.scenarios: Dict[str, Scenario] = self.__read_scenarios()
        self.scenarios_ids = list(self.scenarios)
        if not self.scenarios:
            raise ValueError("No scenarios were loaded; the objective would divide by zero.")

        self.__compute_continuous_approximation()

    def __str__(self):
        return (
            f"---- Instance ----\n"
            f"ID of the instance: {self.id_instance}\n"
            f"Regime: {self.regime}\n"
            f"ID scenario sample: {self.scenarios_ids}\n"
            f"Is continuous X: {self.config.is_continuous_var_x}\n"
            f"Type of flexibility: {self.config.type_of_flexibility}\n"
            f"Periods: {self.periods}\n"
            f"N: {self.config.N}\n"
            f"Quantity of facilities: {len(self.facilities)}\n"
            f"Quantity of vehicles: {len(self.vehicles)}\n"
            f"Quantity of scenarios: {len(self.scenarios)}\n"
            f"-----------------"
        )

    def __read_scenarios(self) -> Dict[str, Scenario]:
        """Load each requested scenario. Keys are always the string form of the id."""
        scenarios = {}
        for id_scenario in self.scenario_ids_requested:
            pixels = get_scenario(str(id_scenario), regime=self.regime)
            scenarios[str(id_scenario)] = Scenario(id_scenario=str(id_scenario), pixels=pixels, periods=self.periods)
        logger.info(f"Loaded {len(scenarios)} scenarios of regime '{self.regime}'.")
        return scenarios

    def __compute_continuous_approximation(self) -> None:
        """Compute the routing cost and fleet size parameters for every scenario."""
        approximation = ContinuousApproximation(
            scenarios=self.scenarios,
            facilities=self.facilities,
            vehicles=self.vehicles,
            use_euclidean_distance=self.use_euclidean_distance,
        )
        self.scenarios = approximation.run_continuous_approximation()

"""Module of class Instance."""

from dataclasses import dataclass
from typing import Dict, List, Optional

from src.core.constants import N_PERIODS
from src.core.entities import Facility, Vehicle
from src.core.inputs import generated_scenario_ids, get_facilities, get_scenario, get_vehicles
from src.core.logging import get_logger
from src.optimization.routing.continuous_approximation import ContinuousApproximation
from src.optimization.scenario import Scenario

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
        scenario_version: Optional[str] = None,
        scenario_set: str = "optimization",
        use_euclidean_distance: bool = False,
        facilities_subset: Optional[List[str]] = None,
    ):  # pylint: disable=too-many-arguments
        self.id_instance = id_instance
        self.regime = regime
        self.scenario_version = scenario_version
        self.scenario_set = scenario_set
        self.use_euclidean_distance = use_euclidean_distance
        self.config = ConfigurationInstance(
            is_continuous_var_x=is_continuous_var_x,
            type_of_flexibility=type_of_flexibility,
            N=N,
            regime=regime,
        )

        if periods not in (N_PERIODS, 1):
            raise ValueError(
                f"periods={periods}; supported planning horizons are {N_PERIODS} and the one-period annual aggregate."
            )
        if periods == 1 and scenario_set != "annual_expected":
            raise ValueError("The one-period horizon is reserved for the annual_expected scenario set.")
        self.periods = periods
        # annual_expected stores one average period; scale its variable costs back to
        # the annual 12-period horizon while keeping installation one-time.
        self.horizon_weight = N_PERIODS if scenario_set == "annual_expected" else 1

        self.vehicles: Dict[str, Vehicle] = get_vehicles()
        self.facilities: Dict[str, Facility] = get_facilities()
        if facilities_subset is not None:
            self.facilities = {k: v for k, v in self.facilities.items() if k in facilities_subset}

        self.scenario_ids_requested = scenario_ids or generated_scenario_ids(regime, N, scenario_version, scenario_set)
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
            pixels = get_scenario(
                str(id_scenario), regime=self.regime, version=self.scenario_version, scenario_set=self.scenario_set
            )
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
            periods=self.periods,
        )
        self.scenarios = approximation.run_continuous_approximation()

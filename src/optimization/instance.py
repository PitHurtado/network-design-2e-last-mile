"""An `Instance` of the two-echelon problem, and the builder that assembles one.

`InstanceSpec` says which instance: regime, scenario set, how many scenarios, horizon.
`InstanceBuilder` reads those scenarios through a `ScenarioLayout`, loads facilities and
vehicles, and runs the routing-cost model (the Continuous Approximation by default).
`Instance` is the plain result the models read.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from src.core.constants import N_PERIODS
from src.core.contract import ScenarioLayout
from src.core.entities import Facility, Vehicle
from src.core.inputs import get_facilities, get_vehicles
from src.optimization.routing.continuous_approximation import ContinuousApproximation
from src.optimization.scenario import Scenario
from src.tools.artifacts import ArtifactKind, ArtifactStore
from src.tools.logging import get_logger

logger = get_logger("Instance")


@dataclass
class ConfigurationInstance:
    """Configuration for the instance."""

    is_continuous_var_x: bool
    type_of_flexibility: str
    N: int
    regime: str


@dataclass(frozen=True)
class InstanceSpec:
    """Which instance to build.

    Scenario selection is explicit: the first `n_scenarios` ids of the set manifest, or
    exactly `scenario_ids`. A missing scenario file raises rather than leaving the
    scenario list empty (the objective divides by its length).
    """

    n_scenarios: int
    regime: str = "normal"
    scenario_set: str = "optimization"
    periods: int = N_PERIODS
    is_continuous_var_x: bool = False
    type_of_flexibility: str = "fixed_operation"
    use_euclidean_distance: bool = False
    scenario_ids: Optional[tuple[str, ...]] = None
    facilities_subset: Optional[tuple[str, ...]] = None
    id_instance: str = "instance"

    def __post_init__(self):
        if self.periods not in (N_PERIODS, 1):
            raise ValueError(
                f"periods={self.periods}; supported planning horizons are {N_PERIODS} and the one-period annual aggregate."
            )
        if self.periods == 1 and self.scenario_set != "annual_expected":
            raise ValueError("The one-period horizon is reserved for the annual_expected scenario set.")

    @property
    def horizon_weight(self) -> int:
        # annual_expected stores one average period; its variable costs are scaled back
        # to the annual 12-period horizon while installation stays one-time.
        return N_PERIODS if self.scenario_set == "annual_expected" else 1


class Instance:  # pylint: disable=too-many-instance-attributes
    """One instance of the two-echelon problem, routing costs included."""

    def __init__(
        self,
        spec: InstanceSpec,
        vehicles: Dict[str, Vehicle],
        facilities: Dict[str, Facility],
        scenarios: Dict[str, Scenario],
        scenario_version: str | None = None,
    ):
        if not scenarios:
            raise ValueError("No scenarios were loaded; the objective would divide by zero.")
        self.spec = spec
        self.id_instance = spec.id_instance
        self.regime = spec.regime
        self.scenario_version = scenario_version
        self.scenario_set = spec.scenario_set
        self.use_euclidean_distance = spec.use_euclidean_distance
        self.config = ConfigurationInstance(
            is_continuous_var_x=spec.is_continuous_var_x,
            type_of_flexibility=spec.type_of_flexibility,
            N=spec.n_scenarios,
            regime=spec.regime,
        )
        self.periods = spec.periods
        self.horizon_weight = spec.horizon_weight
        self.vehicles = vehicles
        self.facilities = facilities
        self.scenarios = scenarios
        self.scenarios_ids: List[str] = list(scenarios)

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


class InstanceBuilder:
    """Builds instances from the scenario sets of one version.

    `routing` is the routing-cost model: a class taking `(scenarios, facilities, vehicles,
    use_euclidean_distance, periods)` whose `run_continuous_approximation()` fills each
    scenario's costs and fleet sizes.
    """

    def __init__(self, layout: ScenarioLayout, version: str | None = None, routing=ContinuousApproximation):
        self.layout = layout
        self.version = version
        self.routing = routing

    @classmethod
    def for_version(cls, version: str, store: ArtifactStore | None = None) -> "InstanceBuilder":
        """A builder over scenario version `version` (`v1`, a `cv-` candidate, or `latest`)."""
        artifact = (store or ArtifactStore()).resolve(version, ArtifactKind.SCENARIOS)
        return cls(ScenarioLayout(artifact.path), version=artifact.id)

    def build(self, spec: InstanceSpec) -> Instance:
        vehicles = get_vehicles()
        facilities = get_facilities()
        if spec.facilities_subset is not None:
            facilities = {k: v for k, v in facilities.items() if k in spec.facilities_subset}

        ids = (
            list(spec.scenario_ids)
            if spec.scenario_ids
            else self.layout.scenario_ids(spec.regime, spec.scenario_set, spec.n_scenarios)
        )
        scenarios = {}
        for id_scenario in ids:
            pixels = self.layout.load_pixels(str(id_scenario), spec.regime, spec.scenario_set)
            scenarios[str(id_scenario)] = Scenario(id_scenario=str(id_scenario), pixels=pixels, periods=spec.periods)
        logger.info(f"Loaded {len(scenarios)} scenarios of regime '{spec.regime}'.")
        if not scenarios:
            raise ValueError("No scenarios were loaded; the objective would divide by zero.")

        approximation = self.routing(
            scenarios=scenarios,
            facilities=facilities,
            vehicles=vehicles,
            use_euclidean_distance=spec.use_euclidean_distance,
            periods=spec.periods,
        )
        scenarios = approximation.run_continuous_approximation()
        return Instance(spec, vehicles, facilities, scenarios, scenario_version=self.version)

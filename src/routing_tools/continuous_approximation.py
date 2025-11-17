"""Module for continuous approximation methods in routing problems."""

import math
from dataclasses import dataclass, field

from src.data.etl import get_distance_facilities, get_distance_facility_delivery_zone
from src.utils.classes import Facility, Pixel, Vehicle
from src.utils.custom_logger import get_logger
from src.utils.scenario import Scenario

logger = get_logger("ContinuousApproximation")


@dataclass
class ApproximationMetrics:
    """Class to store approximation metrics including costs, fleet sizes, and parameters."""

    costs: dict = field(default_factory=dict)  # Total costs
    fleet_sizes: dict = field(default_factory=dict)  # Average number of vehicles
    parameters: dict = field(default_factory=dict)  # Dictionary with all relevant CA parameters


@dataclass
class ApproximationConfiguration:
    """Class to store Approximation configuration including:
    - scenarios
    - facilities
    - delivery zones
    - depot keys
    - vehicles."""

    scenarios: dict[str, Scenario]
    facilities: dict[str, Facility]
    delivery_zones: dict[str, Pixel]
    vehicles: dict[str, Vehicle]
    periods: int = 12


@dataclass
class ApproximationDistances:
    """Class to store distances between facilities and delivery zones."""

    facility_delivery_zone: dict[tuple[str, str], float] = field(
        default_factory=dict
    )  # Distances between facilities and delivery zones
    facilities: dict[str, float] = field(default_factory=dict)  # Distances between facilities


class ContinuousApproximation:
    """
    Class for continuous approximation methods between facilities and delivery zones.
    1. Computes cost and fleet size parameters based on continuous approximation formulas.
    2. Loads distance data from external files.
    3. Stores computed parameters for further analysis.
    Attributes:
        scenarios (dict): Dictionary of scenario objects containing demand data.
        facilities (dict): Dictionary of facility objects.
        pixels (dict): Dictionary of pixel/delivery zone objects.
        vehicles (dict): Dictionary of vehicle objects.
    Outputs:
        c (dict): Total costs for each facility-delivery zone-vehicle combination.
        n (dict): Average number of vehicles required for each combination.
        parameters (dict): Detailed CA parameters for each combination.
    """

    def __init__(
        self,
        scenarios: dict[str, Scenario],
        facilities: dict[str, Facility],
        pixels: dict[str, Pixel],
        vehicles: dict[str, Vehicle],
        show_logs=False,
    ):
        self.__show_logs = show_logs
        self.config = ApproximationConfiguration(
            scenarios=scenarios,
            facilities=facilities,
            delivery_zones=pixels,
            vehicles=vehicles,
            periods=12,
        )
        self.metrics = ApproximationMetrics()
        self.distances = ApproximationDistances()

    def run_continuous_approximation(self) -> tuple[dict, dict, dict]:
        """Compute continuous approximation parameters for all facility-delivery zone-vehicle combinations.
        Returns:
            tuple: A tuple containing three dictionaries:
                - costs (dict): Total costs for each combination.
                - fleet_sizes (dict): Average number of vehicles required for each combination.
                - parameters (dict): Detailed CA parameters for each combination.
        """

        self.__initialize_parameters()
        self.__compute_distances()

        for w, scenario in self.config.scenarios.items():
            for key_delivery_zone, pixel in scenario.pixels.items():
                j = key_delivery_zone
                area = pixel.geo_point.area_surface
                for t in range(self.config.periods):
                    density = pixel.stop_by_period[t] / area    # customers per km^2 # TODO Check if stop or demand
                    drop = pixel.drop_by_period[t]              # items per customer
                    demand = pixel.demand_by_period[t]          # items

                    if demand > 0:
                        for i in self.config.facilities.keys():
                            for v in self.config.vehicles.keys():
                                self.compute_approximation_parameters(
                                    area=area,
                                    density=density,
                                    drop=drop,
                                    i=i,
                                    j=j,
                                    w=w,
                                    v=v,
                                    t=t,
                                )

        self._add_first_echelon_costs()  # Snoeck and Winkenbach (2020) use a per-parcel cost

        return self.metrics.costs, self.metrics.fleet_sizes, self.metrics.parameters

    def __initialize_parameters(self):
        """Initialize cost and fleet size dictionaries for all combinations."""
        for w in self.config.scenarios.keys():
            for i in self.config.facilities.keys():
                for j in self.config.delivery_zones.keys():
                    for v in self.config.vehicles.keys():
                        self.metrics.costs[(i, j, v, w)] = 0
                        self.metrics.fleet_sizes[(i, j, v, w)] = 0

        keys = [
            "T_max",
            "effective_capacity",
            "intra_tour_time_per_customer",
            "tour_time_per_customer",
            "average_tour_time",
            "average_number_fully_loaded_tours",
            "average_number_customers_per_tour",
            "average_number_tours",
            "average_fleet_size",
            "cost_tour_preparation",
            "cost_line_haul",
            "cost_intra_stop",
            "cost_total",
        ]

        self.metrics.parameters = {
            (i, j, v, w): {key: 0 for key in keys}
            for w in self.config.scenarios
            for i in self.config.facilities
            for j in self.config.delivery_zones
            for v in self.config.vehicles
        }

    def __compute_distances(self):
        """Load distance data from external files."""
        self.distances.facility_delivery_zone = get_distance_facility_delivery_zone()
        self.distances.facilities = get_distance_facilities()

    def compute_approximation_parameters(
        self,
        area,
        density,
        drop,
        i,
        j,
        w,
        v,
        t,
    ):
        """Compute continuous approximation parameters for a specific facility-delivery zone-vehicle combination."""
        area = area
        density = density
        drop = drop
        i = i
        j = j
        w = w
        v = v
        t = t
        vehicle = self.config.vehicles[v]
        delivery_zone_circuit_factor = self.config.delivery_zones[j].k

        if v == "first_echelon_truck":
            distance = self.distances.facilities[i]
        else:
            distance = self.distances.facility_delivery_zone[(i, j)]

        # (1) Calculation of auxiliar parameters:

        T_max = vehicle.t_max                   # [hours]

        effective_capacity = (                  # [customer]
            vehicle.capacity / drop             # [item] / [item/customer]
        )

        intra_tour_time_per_customer = (        # [hour/customer]
            vehicle.k                           # [-]
            * delivery_zone_circuit_factor      # [-]
            / (
                math.sqrt(density)              # [sqrt(customer)/km]
                * vehicle.speed_inter_stop      # [km/hour]
            )
        )

        tour_time_per_customer = (              # [hour/customer]
            vehicle.time_set_up                 # [hour/customer]
            + (vehicle.time_service * drop)     # [hours/item] * [item/customer]
            + intra_tour_time_per_customer      # [hour/sqrt(customer)]
        )

        average_tour_time = (                   # [hour]
            effective_capacity                  # [customer]
            * tour_time_per_customer            # [hour/customer]
        )


        average_number_fully_loaded_tours = (       # [-]
            T_max / (                               # [hour]
                (average_tour_time                  # [hour]
                if v!= "first_echelon_truck" else 0)
                + vehicle.time_prep                 # [hour]
                + (                                 # [hour]
                    vehicle.time_loading_per_item   # [hour/item]
                    * effective_capacity            # [customer]
                    * drop                          # [item/customers]
                )
                + (                                 # [hour]
                    2
                    * distance                      # [km]
                    * vehicle.k                     # [-]
                    / vehicle.speed_line_haul       # [km/hour]
                )
            )
        )

        average_number_customers_per_tour = (       # [customer]
            effective_capacity                      # [customer]
            * min(
                1,
                average_number_fully_loaded_tours   # [-]
            )
        )

        average_number_tours = max(             # [-]
            1,
            average_number_fully_loaded_tours   # [-]
        )

        # (2) Compute average fleet size:

        average_fleet_size = (                      # [-]
            area                                    # [km^2]
            * density                               # [customer/km^2]
            / (
                average_number_fully_loaded_tours   # [-]
                * effective_capacity                # [customer]
            )
        )

        # (3) Calculation of costs:

        # (3.1) Preparation costs:
        cost_tour_preparation = (                       # [$]
            vehicle.cost_hour * (                       # [$/hour]
                vehicle.time_prep                       # [hour]
                + vehicle.time_loading_per_item         # [hour/item]
                * average_number_customers_per_tour     # [customer]
                * drop                                  # [item/customer]
            )
        )

        # (3.2) Line-haul transportation costs:
        cost_line_haul = (                  # [$]
            vehicle.cost_hour * (           # [$/hour]
                2
                * distance                  # [km]
                * vehicle.k                 # [-]
                / vehicle.speed_line_haul   # [km/hour]
            )
            + vehicle.cost_km * (           # [$/km]
                2
                * distance                  # [km]
                * vehicle.k                 # [-]
            )
        )

        # (3.3) Intra-stop transportation costs:
        if v == "first_echelon_truck":
            cost_intra_stop = 0
        else:
            cost_intra_stop = (
                vehicle.cost_hour * (                       # [$/hour]
                    tour_time_per_customer                  # [hour/customer]
                    * average_number_customers_per_tour     # [customers]
                )
                + vehicle.cost_km * (                       # [$/km]
                    vehicle.k                               # [-]
                    * delivery_zone_circuit_factor          # [-]
                    * average_number_customers_per_tour     # [customer]
                    / math.sqrt(density)                    # [sqrt(customer)/km]
                )
            )

        # (3.4) Fixed costs per vehicle needed:
        cost_fixed = average_fleet_size * vehicle.cost_fixed

        cost_variable = (
            average_fleet_size * average_number_tours * (cost_tour_preparation + cost_line_haul + cost_intra_stop)
        )  # noqa: E501

        # (3.5) Total cost:
        cost_total = cost_fixed + cost_variable

        self.metrics.costs[(i, j, v, t, w)] = round(cost_total, 2)
        self.metrics.fleet_sizes[(i, j, v, t, w)] = round(average_fleet_size, 2)
        self.metrics.parameters[(i, j, v, t, w)] = {
            "T_max": T_max,
            "effective_capacity": effective_capacity,
            "intra_tour_time_per_customer": intra_tour_time_per_customer,
            "tour_time_per_customer": tour_time_per_customer,
            "average_tour_time": average_tour_time,
            "average_number_fully_loaded_tours": average_number_fully_loaded_tours,
            "average_number_customers_per_tour": average_number_customers_per_tour,
            "average_number_tours": average_number_tours,
            "average_fleet_size": average_fleet_size,
            "cost_fixed": cost_fixed,
            "cost_variable": cost_variable,
            "cost_tour_preparation": cost_tour_preparation,
            "cost_line_haul": cost_line_haul,
            "cost_intra_stop": cost_intra_stop,
            "cost_total": cost_total,
            "distance_to_centroid": distance,
            "line_haul_distance": 0,
            "first_echelon_vehicles": 0,
        }

    def _add_first_echelon_costs(self):
        """Add first echelon costs to other vehicle costs as per Snoeck and Winkenbach (2020)."""
        for w, scenario in self.config.scenarios.items():
            for key_delivery_zone in scenario.demands.keys():
                j = key_delivery_zone
                for i, facility in self.config.facilities.items():
                    if not facility.is_depot:
                        for t in range(self.config.periods):
                            first_echelon_cost = self.metrics.costs[(i, j, "first_echelon_truck", t, w)]
                            line_haul_distance = self.metrics.parameters[(i, j, "first_echelon_truck", t, w)][
                                "distance_to_centroid"
                            ]  # noqa: E501
                            first_echelon_vehicles = self.metrics.parameters[(i, j, "first_echelon_truck", t, w)][
                                "average_fleet_size"
                            ]  # noqa: E501

                            for v in self.config.vehicles.keys():
                                if v not in ("first_echelon_truck"):
                                    self.metrics.costs[(i, j, v, t, w)] = self.metrics.costs[(i, j, v, t, w)] + first_echelon_cost
                                    self.metrics.parameters[(i, j, v, t, w)]["line_haul_distance"] = line_haul_distance
                                    self.metrics.parameters[(i, j, v, t, w)]["first_echelon_vehicles"] = first_echelon_vehicles


if __name__ == "__main__":
    logger.info("This module is intended to be imported and used within other modules.")
    CA = ContinuousApproximation
    logger.info(f"Class '{CA.__name__}' is ready for use.")

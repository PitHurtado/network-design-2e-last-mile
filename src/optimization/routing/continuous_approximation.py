"""Module for continuous approximation methods in routing problems."""

import math
from dataclasses import dataclass, field

from src.core.constants import N_PERIODS
from src.core.entities import Facility, Vehicle
from src.core.inputs import get_distance_facilities, get_distance_facility_delivery_zone
from src.core.logging import get_logger
from src.optimization.scenario import Scenario

logger = get_logger("ContinuousApproximation")


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
    vehicles: dict[str, Vehicle]
    periods: int = N_PERIODS


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
        vehicles: dict[str, Vehicle],
        use_euclidean_distance: bool = False,
        show_logs=False,
        periods: int = N_PERIODS,
    ):
        self.__show_logs = show_logs
        self.use_euclidean_distance = use_euclidean_distance
        self.config = ApproximationConfiguration(
            scenarios=scenarios,
            facilities=facilities,
            vehicles=vehicles,
            periods=periods,
        )
        self.distances = ApproximationDistances()

    def run_continuous_approximation(self) -> dict[str, Scenario]:
        """Compute continuous approximation parameters for all facility-delivery zone-vehicle combinations.
        Returns:
            dict[str, Scenario]: Updated scenarios with continuous approximation parameters.
        """

        self.__compute_distances()

        for w, scenario in self.config.scenarios.items():
            costs = {"facility": dict(), "dc": dict()}
            fleet_sizes = {"facility": dict(), "dc": dict()}
            parameters = {"facility": dict(), "dc": dict()}

            for key_delivery_zone, pixel in scenario.pixels.items():
                j = key_delivery_zone
                area = pixel.geo_point.area_surface
                for t in range(self.config.periods):
                    density = pixel.stop_by_period[t] / area  # customers per km^2 # TODO Check if stop or demand
                    drop = pixel.drop_by_period[t]  # items per customer
                    demand = pixel.demand_by_period[t]  # items

                    if demand > 0:
                        for i in self.config.facilities.keys():
                            for v in self.config.vehicles.keys():
                                cost, fleet_size, params = self.compute_approximation_parameters(
                                    area=area,
                                    density=density,
                                    drop=drop,
                                    i=i,
                                    j=j,
                                    w=w,
                                    v=v,
                                    t=t,
                                )
                                costs["facility"][(i, j, v, t, w)] = round(cost, 5)
                                fleet_sizes["facility"][(i, j, v, t, w)] = round(fleet_size, 5)
                                parameters["facility"][(i, j, v, t, w)] = params
                        cost, fleet_size, params = self.compute_approximation_parameters(
                            area=area,
                            density=density,
                            drop=drop,
                            i="DC",
                            j=j,
                            w=w,
                            v="large",
                            t=t,
                        )
                        costs["dc"][(j, "large", t, w)] = round(cost, 5)
                        fleet_sizes["dc"][(j, "large", t, w)] = round(fleet_size, 5)
                        parameters["dc"][(j, "large", t, w)] = params
            scenario.set_costs(costs)
            scenario.set_fleet_size(fleet_sizes)
            scenario.set_parameters(parameters)

        self._add_first_echelon_costs()

        return self.config.scenarios

    def __compute_distances(self):
        """Load or compute distances between facilities and pixels."""
        if self.use_euclidean_distance:
            self.distances.facility_delivery_zone = self.__compute_euclidean_distances()
        else:
            self.distances.facility_delivery_zone = get_distance_facility_delivery_zone()
        self.distances.facilities = get_distance_facilities()

    def __compute_euclidean_distances(self) -> dict:
        """Override facility-to-pixel distances with haversine; keep DC distances from Excel."""

        def _haversine(lon1, lat1, lon2, lat2):
            phi1, phi2 = math.radians(lat1), math.radians(lat2)
            a = (
                math.sin((math.radians(lat2 - lat1)) / 2) ** 2
                + math.cos(phi1) * math.cos(phi2) * math.sin((math.radians(lon2 - lon1)) / 2) ** 2
            )
            return 6371.0 * 2 * math.asin(math.sqrt(a))

        # Start with Excel distances (covers DC-to-pixel entries)
        distances = get_distance_facility_delivery_zone()

        all_pixels = {}
        for scenario in self.config.scenarios.values():
            all_pixels.update(scenario.pixels)

        # Override only facility-to-pixel pairs with haversine
        for i, facility in self.config.facilities.items():
            for j, pixel in all_pixels.items():
                distances[(i, j)] = _haversine(
                    facility.geo_point.lon,
                    facility.geo_point.lat,
                    pixel.geo_point.lon,
                    pixel.geo_point.lat,
                )
        return distances

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
        delivery_zone_circuit_factor = self.config.scenarios[w].pixels[j].k

        if v == "first_echelon_truck":
            distance = self.distances.facilities[i]
        else:
            distance = self.distances.facility_delivery_zone[(i, j)]

        # (1) Calculation of auxiliar parameters:

        T_max = vehicle.t_max  # [hours]

        effective_capacity = vehicle.capacity / drop  # [customer]  # [item] / [item/customer]

        intra_tour_time_per_customer = (  # [hour/customer]
            vehicle.k  # [-]
            * delivery_zone_circuit_factor  # [-]
            / (math.sqrt(density) * vehicle.speed_inter_stop)  # [sqrt(customer)/km]  # [km/hour]
        )

        tour_time_per_customer = (  # [hour/customer]
            vehicle.time_set_up  # [hour/customer]
            + (vehicle.time_service * drop)  # [hours/item] * [item/customer]
            + intra_tour_time_per_customer  # [hour/sqrt(customer)]
        )

        average_tour_time = effective_capacity * tour_time_per_customer  # [hour]  # [customer]  # [hour/customer]

        average_number_fully_loaded_tours = T_max / (  # [-]  # [hour]
            (average_tour_time if v != "first_echelon_truck" else 0)  # [hour]
            + vehicle.time_prep  # [hour]
            + (  # [hour]
                vehicle.time_loading_per_item * effective_capacity * drop  # [hour/item]  # [customer]  # [item/customers]
            )
            + (2 * distance * vehicle.k / vehicle.speed_line_haul)  # [hour]  # [km]  # [-]  # [km/hour]
        )

        average_number_customers_per_tour = effective_capacity * min(  # [customer]  # [customer]
            1, average_number_fully_loaded_tours  # [-]
        )

        average_number_tours = max(1, average_number_fully_loaded_tours)  # [-]  # [-]

        # (2) Compute average fleet size:

        average_fleet_size = (  # [-]
            area  # [km^2]
            * density  # [customer/km^2]
            / (average_number_fully_loaded_tours * effective_capacity)  # [-]  # [customer]
        )

        # (3) Calculation of costs:

        # (3.1) Preparation costs:
        cost_tour_preparation = vehicle.cost_hour * (  # [$]  # [$/hour]
            vehicle.time_prep  # [hour]
            + vehicle.time_loading_per_item  # [hour/item]
            * average_number_customers_per_tour  # [customer]
            * drop  # [item/customer]
        )

        # (3.2) Line-haul transportation costs:
        cost_line_haul = vehicle.cost_hour * (  # [$]  # [$/hour]
            2 * distance * vehicle.k / vehicle.speed_line_haul  # [km]  # [-]  # [km/hour]
        ) + vehicle.cost_km * (  # [$/km]
            2 * distance * vehicle.k  # [km]  # [-]
        )

        # (3.3) Intra-stop transportation costs:
        if v == "first_echelon_truck":
            cost_intra_stop = 0
        else:
            cost_intra_stop = vehicle.cost_hour * (  # [$/hour]
                tour_time_per_customer * average_number_customers_per_tour  # [hour/customer]  # [customers]
            ) + vehicle.cost_km * (  # [$/km]
                vehicle.k  # [-]
                * delivery_zone_circuit_factor  # [-]
                * average_number_customers_per_tour  # [customer]
                / math.sqrt(density)  # [sqrt(customer)/km]
            )

        # (3.4) Fixed costs per vehicle needed:
        cost_fixed = average_fleet_size * vehicle.cost_fixed

        cost_variable = (
            average_fleet_size * average_number_tours * (cost_tour_preparation + cost_line_haul + cost_intra_stop)
        )  # noqa: E501

        # (3.5) Total cost:
        cost_total = cost_fixed + cost_variable

        return (
            round(cost_total, 5),
            round(average_fleet_size, 5),
            {
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
            },
        )

    def _add_first_echelon_costs(self) -> None:
        """Add DC-to-satellite costs to satellite-to-pixel costs per Snoeck and Winkenbach (2020).

        Mutates scenario.costs["facility"][(i, j, v, t, w)] in-place for all
        non-first-echelon vehicles. Also backfills "line_haul_distance" and
        "first_echelon_vehicles" in parameters for downstream analysis.
        Must be called after run_continuous_approximation() completes all scenarios.
        """
        for w, scenario in self.config.scenarios.items():
            for key_delivery_zone, pixel in scenario.pixels.items():
                j = key_delivery_zone
                for t in range(self.config.periods):
                    if pixel.demand_by_period[t] <= 0:
                        continue

                    for i, facility in self.config.facilities.items():
                        if facility.is_depot:
                            continue

                        fe_key = (i, j, "first_echelon_truck", t, w)
                        first_echelon_cost = scenario.costs["facility"][fe_key]
                        line_haul_distance = scenario.parameters["facility"][fe_key]["distance_to_centroid"]
                        first_echelon_vehicles = scenario.parameters["facility"][fe_key]["average_fleet_size"]

                        for v in self.config.vehicles.keys():
                            if v == "first_echelon_truck":
                                continue

                            v_key = (i, j, v, t, w)
                            scenario.costs["facility"][v_key] = round(scenario.costs["facility"][v_key] + first_echelon_cost, 5)
                            scenario.parameters["facility"][v_key]["line_haul_distance"] = line_haul_distance
                            scenario.parameters["facility"][v_key]["first_echelon_vehicles"] = first_echelon_vehicles


if __name__ == "__main__":
    logger.info("This module is intended to be imported and used within other modules.")
    CA = ContinuousApproximation
    logger.info(f"Class '{CA.__name__}' is ready for use.")

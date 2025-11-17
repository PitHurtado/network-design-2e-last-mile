"""Module for classes used in the application model."""

from dataclasses import dataclass, field


@dataclass
class GeoPoint:
    """Class for geographical points."""

    lon: float
    lat: float
    area_surface: float = field(default=0.0)
    speed_intra_stop: dict[str, float] = field(default_factory=dict)


class Pixel:
    """Class for pixels in the map."""

    def __init__(
        self,
        id_pixel: str,
        lon: float,
        lat: float,
        area_surface: float,
        speed_intra_stop: dict[str, float],
        k: float = 0.57,
        is_available: bool = False,
    ):  # pylint: disable=too-many-arguments
        self.id_pixel = id_pixel
        self.geo_point = GeoPoint(lon, lat, area_surface, speed_intra_stop)
        self.demand_by_period = []
        self.drop_by_period = []
        self.stop_by_period = []
        self.k = k
        self.is_available = is_available

    def __str__(self):
        return f"Pixel(ID: {self.id_pixel})"

    def set_scenario_data(
        self, demand_by_period: list[float], drop_by_period: list[float], stop_by_period: list[float]
    ):  # noqa: E501
        """Sets the scenario data for the pixel."""
        self.demand_by_period = demand_by_period
        self.drop_by_period = drop_by_period
        self.stop_by_period = stop_by_period
        self.is_available = True


class Facility:
    """Class for facility in the map."""

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        id_facility: str,
        lon: float,
        lat: float,
        capacity: dict[str, float],
        cost_installation: dict[str, float],
        cost_operation: dict[str, list[float]],
        cost_sourcing: float = 0.335,
        is_depot: bool = False,
    ):  # pylint: disable=too-many-arguments
        self.geo_point = GeoPoint(lon, lat)
        self.id_facility = id_facility
        self.cost_installation = cost_installation
        self.cost_operation = cost_operation
        self.cost_sourcing = cost_sourcing
        self.capacity = capacity
        self.is_depot = is_depot

    def __str__(self):
        return f"Facility(ID: {self.id_facility})"


class Vehicle:
    """Class for vehicles"""

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        id_vehicle: str,
        type_vehicle: str,  # Type of Vehicle (e.g. Truck, Van, Bike) # pylint: disable=line-too-long
        capacity: float,  # Capacity of the vehicle (e.g. 1000 items) # pylint: disable=line-too-long
        cost_fixed: float,  # Fix Costs - only applicable for own # pylint: disable=line-too-long
        time_prep: float,  # Time Set-up: Fixed Set-Up Time at the facility for vehicle type v. Time to dispatch (~time prep called) # pylint: disable=line-too-long # noqa: E501
        time_loading_per_item: float,  # Time Set-Up: Time to Load the Vehicle (Time/ package) # pylint: disable=line-too-long # noqa: E501
        time_set_up: float,  # Intra-Stop Time: Set-up Time per vehicle (h/customer; ie parkting time) # pylint: disable=line-too-long # noqa: E501
        time_service: float,  # Intra-Stop Time: Incremental Service Time for delivery option (h/item) # pylint: disable=line-too-long # noqa: E501
        speed_line_haul: float,
        speed_inter_stop: float,
        t_max: float,
        cost_hour: float,
        cost_km: float,
        cost_item: float,
        k: float = 0.57,
    ):
        self.id_vehicle = str(id_vehicle)
        self.type_vehicle = type_vehicle
        self.capacity = capacity
        self.cost_fixed = cost_fixed
        self.cost_hour = cost_hour
        self.cost_km = cost_km
        self.cost_item = cost_item

        self.time_set_up = time_set_up
        self.time_service = time_service
        self.time_prep = time_prep
        self.time_loading_per_item = time_loading_per_item
        self.t_max = t_max

        self.speed_line_haul = speed_line_haul
        self.speed_inter_stop = speed_inter_stop
        self.k = k

    def __str__(self):
        return f"Vehicle(ID: {self.id_vehicle}, Type: {self.type_vehicle}, Capacity: {self.capacity})"

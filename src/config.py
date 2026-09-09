"""Module to get configuration from json file"""

# Configurations of the vehicles
SMALL_CONFIG = {
    "id_vehicle": "van",
    "type_vehicle": "small",
    "capacity": 115,
    "cost_fixed": 67,
    "time_prep": 5 / 60,
    "time_loading_per_item": 0.067 / 60,
    "time_set_up": 2 / 60,
    "time_service": 1 / 60,
    "speed_line_haul": 50,
    "speed_inter_stop": 35,
    "t_max": 12,
    "cost_hour": 53.9,
    "cost_km": 0.37,
    "cost_item": 0.5,
}
LARGE_CONFIG = {
    "id_vehicle": "first_echelon_truck",
    "type_vehicle": "large",
    "capacity": 460,
    "cost_fixed": 268,
    "time_prep": 10 / 60,
    "time_loading_per_item": 0.05 / 60,
    "time_set_up": 2 / 60,
    "time_service": 2 / 60,
    "speed_line_haul": 35,
    "speed_inter_stop": 20,
    "t_max": 12,
    "cost_hour": 50,
    "cost_km": 8.7,
    "cost_item": 0.5,
    "k": 1,
}

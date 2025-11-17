"""Constants for the application."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

# Paths
PATH_DATA_SATELLITE = DATA_DIR / "input_satellites.xlsx"
PATH_DATA_PIXEL = DATA_DIR / "input_pixels.xlsx"

# Path root from scenarios
PATH_ROOT_SCENARIO = DATA_DIR / "scenarios/"

PATH_SAMPLING_SCENARIO = PATH_ROOT_SCENARIO / "sampling/"
PATH_BEST_SOLUTION_SAA = PATH_ROOT_SCENARIO / "best_solution_saa/"

# distances matrix
PATH_DATA_DISTANCES_FACILITY_DELIVERY_ZONE = DATA_DIR / "distances/input_matrix_distance_facilities_pixels.xlsx"

PATH_DATA_DISTANCES_FACILITIES = DATA_DIR / "distances/input_matrix_distance_dc_facilities.xlsx"

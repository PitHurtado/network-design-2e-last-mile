"""Readers for the raw inputs of the study: pixel grid, facilities, distance matrices
and vehicle configs. Generated scenarios are read through `src.core.contract`.

Shared on purpose: `src.scenarios` needs the pixel grid to fit shape parameters, and
`src.optimization` needs all of it to build an `Instance`.
"""

import ast

import pandas as pd

from src.core.config import LARGE_CONFIG, SMALL_CONFIG
from src.core.constants import (
    PATH_DATA_DISTANCES_FACILITIES,
    PATH_DATA_DISTANCES_FACILITY_DELIVERY_ZONE,
    PATH_DATA_FACILITY,
    PATH_DATA_PIXEL,
)
from src.core.entities import Facility, Pixel, Vehicle
from src.tools.logging import get_logger

logger = get_logger("Inputs")


def get_distance_facility_delivery_zone() -> dict:
    """Get distances between facilities and delivery zones from an external file."""
    try:
        distance_facility_delivery_zone = {}

        df = pd.read_excel(PATH_DATA_DISTANCES_FACILITY_DELIVERY_ZONE, engine="openpyxl")
        for _, row in df.iterrows():
            i = row["id_facility"].upper()
            j = f'{row["layer"].upper()}-{int(row["pixel"])}'
            distance_facility_delivery_zone[(i, j)] = row["distance"]

        return distance_facility_delivery_zone
    except FileNotFoundError as error:
        logger.error(f"File {PATH_DATA_DISTANCES_FACILITY_DELIVERY_ZONE} not found")
        raise error


def get_distance_facilities() -> dict:
    """Get distances between the DC and each facility from an external file."""
    try:
        distance_facilities = {}

        df = pd.read_excel(PATH_DATA_DISTANCES_FACILITIES, engine="openpyxl")
        for _, row in df.iterrows():
            distance_facilities[row["id_facility"].upper()] = row["distance"]

        return distance_facilities
    except FileNotFoundError as error:
        logger.error(f"File {PATH_DATA_DISTANCES_FACILITIES} not found")
        raise error


def get_vehicles() -> dict[str, Vehicle]:
    """Get the vehicle fleet configuration."""
    return {
        "small": Vehicle(**SMALL_CONFIG),
        "large": Vehicle(**LARGE_CONFIG),
        "first_echelon_truck": Vehicle(**LARGE_CONFIG),
    }


def get_facilities() -> dict[str, Facility]:
    """Get facilities from an external file."""
    try:
        facilities = {}

        df = pd.read_excel(PATH_DATA_FACILITY, engine="openpyxl")
        for _, row in df.iterrows():
            id_facility = row["id_facility"].upper()
            facilities[id_facility] = Facility(
                id_facility=id_facility,
                lon=row["lon"],
                lat=row["lat"],
                capacity=ast.literal_eval(row["capacity"]),
                cost_installation=ast.literal_eval(row["cost_installation"]),
                cost_operation=ast.literal_eval(row["cost_operation"]),
                cost_sourcing=row["cost_sourcing"],
            )

        return facilities
    except FileNotFoundError as error:
        logger.error(f"File {PATH_DATA_FACILITY} not found")
        raise error


def get_pixels() -> dict[str, Pixel]:
    """Get the pixel grid (geometry and service speeds) from an external file.

    Note that `lon`/`lat` here are demand-weighted representative service points,
    not the geometric cell centroids, and `area_surface` is the number of ~1 km^2
    grid cells merged into the pixel (so it is an area in km^2).
    """
    try:
        pixels = {}
        df = pd.read_excel(PATH_DATA_PIXEL, engine="openpyxl")
        for _, row in df.iterrows():
            id_pixel = f'{row["layer"].upper()}-{int(row["pixel"])}'
            pixels[id_pixel] = Pixel(
                id_pixel=id_pixel,
                lon=row["lon"],
                lat=row["lat"],
                area_surface=row["area_surface"],
                speed_intra_stop=ast.literal_eval(row["speed_intra_stop"]),
            )

        return pixels
    except FileNotFoundError as error:
        logger.error(f"File {PATH_DATA_PIXEL} not found")
        raise error

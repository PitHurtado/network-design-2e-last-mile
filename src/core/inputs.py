"""Readers for every file the study takes as input: pixel grid, facilities,
distance matrices, vehicle configs and the generated scenario JSONs.

Shared on purpose: `src.pipeline` needs the pixel grid to fit shape parameters, and
`src.optimization` needs all of it to build an `Instance`.
"""

import ast
import json
from pathlib import Path

import pandas as pd

from src.core.config import LARGE_CONFIG, SMALL_CONFIG
from src.core.constants import (
    PATH_DATA_DISTANCES_FACILITIES,
    PATH_DATA_DISTANCES_FACILITY_DELIVERY_ZONE,
    PATH_DATA_FACILITY,
    PATH_DATA_PIXEL,
    scenario_dir,
)
from src.core.entities import Facility, Pixel, Vehicle
from src.core.logging import get_logger

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


def scenario_path(id_scenario: str, regime: str = "normal") -> Path:
    """Path of a generated scenario file for a given demand regime."""
    return scenario_dir(regime) / f"scenario_{id_scenario}.json"


def get_scenario(id_scenario: str, regime: str = "normal") -> dict[str, Pixel]:
    """Get scenario pixels from an external file.

    Only pixels present in both the scenario file and `input_pixels.xlsx` are
    returned. A pixel in the scenario but missing from the grid is dropped, so the
    mismatch is logged as a warning and counted rather than passing silently.
    """
    pixels = get_pixels()
    path = scenario_path(id_scenario, regime)
    if not path.exists():
        logger.error(f"Scenario file {path} not found.")
        raise FileNotFoundError(f"Scenario file {path} not found.")

    with open(path, "r") as file:
        data = json.load(file)

    unknown = []
    for pixel_data in data["pixels"]:
        id_pixel = pixel_data["id_pixel"]
        if id_pixel in pixels:
            pixels[id_pixel].set_scenario_data(
                demand_by_period=pixel_data["demand"],
                drop_by_period=pixel_data["drop"],
                stop_by_period=pixel_data["stop"],
            )
        else:
            unknown.append(id_pixel)

    if unknown:
        logger.warning(f"{len(unknown)} pixels in {path.name} are absent from the grid and were dropped: {unknown[:5]}")

    available = {i: p for i, p in pixels.items() if p.is_available}
    missing = len(pixels) - len(available)
    if missing:
        logger.warning(f"{missing} grid pixels have no data in {path.name} and were excluded.")

    logger.info(f"Scenario {id_scenario} ({regime}): {len(available)} pixels loaded.")
    return available

"""Module for ETL processes related to distance data."""

import ast
import json
import os

import pandas as pd

from src.config import LARGE_CONFIG, SMALL_CONFIG
from src.constants import (
    PATH_DATA_DISTANCES_FACILITIES,
    PATH_DATA_DISTANCES_FACILITY_DELIVERY_ZONE,
    PATH_DATA_FACILITY,
    PATH_DATA_PIXEL,
    PATH_ROOT_SCENARIO,
)
from src.utils.classes import Facility, Pixel, Vehicle
from src.utils.custom_logger import get_logger

logger = get_logger("ETL")


def get_distance_facility_delivery_zone() -> dict:
    """Get distances between facilities and delivery zones from an external file."""
    try:
        distance_facility_delivery_zone = {}

        df = pd.read_excel(PATH_DATA_DISTANCES_FACILITY_DELIVERY_ZONE)
        for _, row in df.iterrows():
            i = row["id_facility"].upper()
            j = f'{row["layer"].upper()}-{row["pixel"]}'
            distance = row["distance"]
            distance_facility_delivery_zone[(i, j)] = distance

        return distance_facility_delivery_zone
    except FileNotFoundError as error:
        logger.error(f"File {PATH_DATA_DISTANCES_FACILITY_DELIVERY_ZONE} not found")
        raise error


def get_distance_facilities() -> dict:
    """Get distances between facilities from an external file."""
    try:
        distance_facilities = {}

        df = pd.read_excel(PATH_DATA_DISTANCES_FACILITIES)
        for _, row in df.iterrows():
            i = row["id_facility"].upper()
            distance = row["distance"]
            distance_facilities[i] = distance

        return distance_facilities

    except FileNotFoundError as error:
        logger.error(f"File {PATH_DATA_DISTANCES_FACILITIES} not found")
        raise error


def get_vehicles() -> dict[str, Vehicle]:
    """Get vehicles from an external file."""
    return {
        "small": Vehicle(**SMALL_CONFIG),
        "large": Vehicle(**LARGE_CONFIG),
    }


def get_facilities() -> dict[str, Facility]:
    """Get facilities from an external file."""
    try:
        facilities = {}

        df = pd.read_excel(PATH_DATA_FACILITY)
        for _, row in df.iterrows():
            id_facility = row["id_facility"].upper()
            lon = row["lon"]
            lat = row["lat"]
            capacity = ast.literal_eval(row["capacity"])
            cost_installation = ast.literal_eval(row["cost_installation"])
            cost_operation = ast.literal_eval(row["cost_operation"])
            cost_sourcing = row["cost_sourcing"]
            facilities[id_facility] = Facility(
                id_facility=id_facility,
                lon=lon,
                lat=lat,
                capacity=capacity,
                cost_installation=cost_installation,
                cost_operation=cost_operation,
                cost_sourcing=cost_sourcing,
            )

        return facilities
    except FileNotFoundError as error:
        logger.error(f"File {PATH_DATA_DISTANCES_FACILITIES} not found")
        raise error


def get_pixels() -> dict[str, Pixel]:
    """Get pixels from an external file."""
    try:
        pixels = {}
        df = pd.read_excel(PATH_DATA_PIXEL)
        for _, row in df.iterrows():
            id_pixel = f'{row["layer"].upper()}-{row["pixel"]}'
            lon = row["lon"]
            lat = row["lat"]
            area_surface = row["area_surface"]
            speed_intra_stop = ast.literal_eval(row["speed_intra_stop"])
            pixels[id_pixel] = Pixel(
                id_pixel=id_pixel,
                lon=lon,
                lat=lat,
                area_surface=area_surface,
                speed_intra_stop=speed_intra_stop,
            )

        return pixels
    except FileNotFoundError as error:
        logger.error(f"File {PATH_DATA_DISTANCES_FACILITIES} not found")
        raise error


def get_scenario(id_scenario: str) -> dict[str, Pixel]:
    """Get scenario pixels from an external file."""
    pixels = get_pixels()
    try:

        path_scenario = PATH_ROOT_SCENARIO / f"scenario_{id_scenario}.json"
        # logger.info(f"Loading scenario from {path_scenario}")
        if os.path.exists(path_scenario):
            with open(path_scenario, "r") as file:
                data = json.load(file)
                logger.info(f"Scenario data loaded: {data}")
                for pixel_data in data["pixels"]:
                    id_pixel = pixel_data["id_pixel"]
                    demand_by_period = pixel_data["demand"]
                    drop_by_period = pixel_data["drop"]
                    stop_by_period = pixel_data["stop"]
                    if id_pixel in pixels:
                        pixels[id_pixel].set_scenario_data(
                            demand_by_period=demand_by_period,
                            drop_by_period=drop_by_period,
                            stop_by_period=stop_by_period,
                        )
                    # elif id_pixel.startswith("0-"):
                    #     continue
                    else:
                        logger.warning(f"Pixel {id_pixel} not found in pixels data.")
        else:
            logger.error(f"Scenario file {path_scenario} not found.")
            raise FileNotFoundError(f"Scenario file {path_scenario} not found.")
    except FileNotFoundError as error:
        logger.error(f"Not Found Scenario: {error}")
        raise error
    return {id: pixels[id] for id in pixels if pixels[id].is_available}


if __name__ == "__main__":
    logger.info("ETL Distance Data Module")
    distances_facilities_delivery_zone = get_distance_facility_delivery_zone()
    distances_facilities = get_distance_facilities()
    vehicles = get_vehicles()
    facilities = get_facilities()
    pixels = get_pixels()
    logger.info(f"Distances Facility-Delivery Zone: {distances_facilities_delivery_zone}")
    logger.info(f"Distances Facilities: {distances_facilities}")
    logger.info(f"Vehicles: {[str(vehicles[v]) for v in vehicles]}")
    logger.info(f"Facilities: {[str(facilities[f]) for f in facilities]}")
    logger.info(f"Pixels: {[str(pixels[p]) for p in pixels]}")
    logger.info("ETL Distance Data Module Finished")

    # id_scenario = "1"
    # scenario_pixels = get_scenario(id_scenario)
    # logger.info(f"Scenario {id_scenario} Pixels: {[str(scenario_pixels[p]) for p in scenario_pixels]}")

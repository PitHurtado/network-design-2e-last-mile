"""Module for ETL processes related to distance data."""

import pandas as pd

from src.constants import PATH_DATA_DISTANCES_FACILITIES, PATH_DATA_DISTANCES_FACILITY_DELIVERY_ZONE
from src.utils.custom_logger import get_logger

logger = get_logger("ETL")


def get_distance_facility_delivery_zone() -> dict:
    """Get distances between facilities and delivery zones from an external file."""
    try:
        distance_facility_delivery_zone = {}

        df = pd.read_excel(PATH_DATA_DISTANCES_FACILITY_DELIVERY_ZONE)
        for _, row in df.iterrows():
            i = row["id_facility"]
            j = row["id_pixel"]
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
            i = row["id_facility"]
            distance = row["distance"]
            distance_facilities[i] = distance

        return distance_facilities

    except FileNotFoundError as error:
        logger.error(f"File {PATH_DATA_DISTANCES_FACILITIES} not found")
        raise error


if __name__ == "__main__":
    logger.info("ETL Distance Data Module")
    distances_facilities_delivery_zone = get_distance_facility_delivery_zone()
    distances_facilities = get_distance_facilities()
    logger.info(f"Distances Facility-Delivery Zone: {distances_facilities_delivery_zone}")
    logger.info(f"Distances Facilities: {distances_facilities}")
    logger.info("ETL Distance Data Module Finished")

"""Geometry of the regular lon/lat grid the pixels are built from.

Cell ids follow `cell = row * GRID_N_COLS + col`, with `col` increasing eastward and
`row` increasing northward from the SW corner (see `constants.py`).
"""

import math

from src.core.constants import GRID_DLAT, GRID_DLON, GRID_LAT0, GRID_LON0, GRID_N_COLS, GRID_N_ROWS

# Cell-centre coordinates of every grid column / row, the axes of cell-level heatmaps.
GRID_X = [GRID_LON0 + (col + 0.5) * GRID_DLON for col in range(GRID_N_COLS)]
GRID_Y = [GRID_LAT0 + (row + 0.5) * GRID_DLAT for row in range(GRID_N_ROWS)]


def cell_center(cell: int) -> tuple[float, float]:
    """Geometric centre (lon, lat) of a grid cell."""
    row, col = divmod(int(cell), GRID_N_COLS)
    return GRID_LON0 + (col + 0.5) * GRID_DLON, GRID_LAT0 + (row + 0.5) * GRID_DLAT


def layer_of(id_pixel: str) -> str:
    """Layer prefix of ids such as ``A-104`` or ``A_104``."""
    return str(id_pixel).replace("_", "-").split("-", 1)[0]


EARTH_RADIUS_KM = 6371.0


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance between two points, in km."""
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return EARTH_RADIUS_KM * 2 * math.asin(math.sqrt(a))

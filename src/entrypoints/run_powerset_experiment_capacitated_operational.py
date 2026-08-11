"""Powerset experiment — Capacitated SAA with flex operational decisions + operational costs.

Same 2^9 = 512 satellite subsets as the other experiments.
Results: results/powerset_experiment_capacitated_operational/
"""

import logging
from itertools import combinations

from src.constants import RESULTS_DIR
from src.data.etl import get_facilities
from src.entrypoints.main_capacitated_flex_saa import MainCapacitatedFlex
from src.utils.custom_logger import get_logger

logger = get_logger("PowersetCapacitatedOperational")


def _label(subset: tuple) -> str:
    return "DC_only" if len(subset) == 0 else "+".join(sorted(subset))


if __name__ == "__main__":
    FOLDER_PATH = RESULTS_DIR / "powerset_experiment_capacitated_operational"
    FOLDER_PATH.mkdir(parents=True, exist_ok=True)

    all_facs = sorted(get_facilities().keys())
    subsets  = [c for r in range(0, len(all_facs) + 1)
                  for c in combinations(all_facs, r)]
    total    = len(subsets)
    logger.info(f"Total configs: {total}  (2^{len(all_facs)} = {2**len(all_facs)})")

    for idx, subset in enumerate(subsets, start=1):
        label    = _label(subset)
        out_json = FOLDER_PATH / f"capflex_{label}_N1.json"

        if out_json.exists():
            logger.info(f"[{idx}/{total}] Skip: {label}")
            continue

        logger.info(f"[{idx}/{total}] Running: {label}")
        try:
            main = MainCapacitatedFlex(
                id_instance=label,
                folder_path=FOLDER_PATH,
                configuration=(1, True),   # N=1, continuous X (fast screening)
                max_run_time=60 * 5,
                use_euclidean_distance=True,
                facilities_subset=list(subset),   # empty list [] = DC_only (no satellites)
            )
            logging.disable(logging.CRITICAL)
            path = main.solve()
            logging.disable(logging.NOTSET)
            logger.info(f"[{idx}/{total}] Done → {path.name}")
        except Exception as e:
            logging.disable(logging.NOTSET)
            logger.error(f"[{idx}/{total}] Failed: {label} — {e}")

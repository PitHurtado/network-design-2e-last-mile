"""Run the uncapacitated SAA model for all powerset combinations of satellites."""

import logging
from itertools import combinations

from src.constants import RESULTS_DIR
from src.data.etl import get_facilities
from src.entrypoints.main_uncapacitated_saa import Main
from src.utils.custom_logger import get_logger

logger = get_logger("PowersetExperiment")


def _label(subset: tuple) -> str:
    return "DC_only" if len(subset) == 0 else "+".join(sorted(subset))


if __name__ == "__main__":
    FOLDER_PATH = RESULTS_DIR / "powerset_experiment"
    FOLDER_PATH.mkdir(parents=True, exist_ok=True)

    all_facilities = sorted(get_facilities().keys())
    n_fac = len(all_facilities)

    subsets = []
    for r in range(0, n_fac + 1):
        subsets.extend(combinations(all_facilities, r))

    total = len(subsets)
    logger.info(f"Total configurations: {total}  (2^{n_fac} = {2**n_fac})")

    for idx, subset in enumerate(subsets, start=1):
        label = _label(subset)
        output_json = FOLDER_PATH / f"uncapacitated_{label}_None.json"
        if output_json.exists():
            logger.info(f"[{idx}/{total}] Skipping (already exists): {label}")
            continue

        logger.info(f"[{idx}/{total}] Running: {label}")
        try:
            main = Main(
                id_instance=label,
                folder_path=FOLDER_PATH,
                configuration=(1, True),
                id_sampling=None,
                max_run_time=60 * 5,
                use_euclidean_distance=True,
                facilities_subset=list(subset),
            )
            # Suppress Gurobi output for each run
            logging.disable(logging.CRITICAL)
            path = main.solve()
            logging.disable(logging.NOTSET)
            logger.info(f"[{idx}/{total}] Done: {label} → {path.name}")
        except Exception as e:
            logging.disable(logging.NOTSET)
            logger.error(f"[{idx}/{total}] Failed: {label} — {e}")

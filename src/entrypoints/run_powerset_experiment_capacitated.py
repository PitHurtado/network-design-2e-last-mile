"""Powerset experiment for the Capacitated SAA Model.

Runs all 2^9 = 512 satellite subset combinations, saving results to
results/powerset_experiment_capacitated/.

Skips subsets whose output JSON already exists (safe to re-run).
"""

import logging
from itertools import combinations

from src.constants import RESULTS_DIR
from src.data.etl import get_facilities
from src.entrypoints.main_capacitated_saa import MainCapacitated
from src.utils.custom_logger import get_logger

logger = get_logger("PowersetExperimentCapacitated")


def _label(subset: tuple) -> str:
    return "DC_only" if len(subset) == 0 else "+".join(sorted(subset))


if __name__ == "__main__":
    FOLDER_PATH = RESULTS_DIR / "powerset_experiment_capacitated"
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
        out_json = FOLDER_PATH / f"capacitated_{label}_None.json"

        if out_json.exists():
            logger.info(f"[{idx}/{total}] Skipping (exists): {label}")
            continue

        logger.info(f"[{idx}/{total}] Running: {label}")
        try:
            main = MainCapacitated(
                id_instance=label,
                folder_path=FOLDER_PATH,
                configuration=(1, True),   # N=1 scenario, continuous X
                id_sampling=None,
                max_run_time=60 * 5,
                use_euclidean_distance=True,
                facilities_subset=list(subset),
            )
            logging.disable(logging.CRITICAL)
            path = main.solve()
            logging.disable(logging.NOTSET)
            logger.info(f"[{idx}/{total}] Done: {label} → {path.name}")
        except Exception as e:
            logging.disable(logging.NOTSET)
            logger.error(f"[{idx}/{total}] Failed: {label} — {e}")

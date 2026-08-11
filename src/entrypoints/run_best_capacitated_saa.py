"""Run the best capacitated configurations with N=30 scenarios and integer X variables.

Best configurations are extracted from the powerset experiment results
(best objective per number of active satellites).

Results saved to: results/best_capacitated_saa/
"""

import json
import logging
from pathlib import Path

from src.constants import RESULTS_DIR
from src.entrypoints.main_capacitated_saa import MainCapacitated
from src.utils.custom_logger import get_logger

logger = get_logger("BestCapacitatedSAA")

POWERSET_DIR = RESULTS_DIR / "powerset_experiment_capacitated"
OUTPUT_DIR   = RESULTS_DIR / "best_capacitated_saa"

N_SCENARIOS      = 30
IS_CONTINUOUS_X  = False   # integer (binary) X variables
MAX_RUN_TIME     = 60 * 15  # 15 min per config


def _best_per_n_active() -> dict[int, dict]:
    """Return {n_active: {label, obj, facilities_subset}} from powerset results."""
    jsons = sorted(POWERSET_DIR.glob("capacitated_*_None.json"))
    if not jsons:
        raise FileNotFoundError(f"No powerset results in {POWERSET_DIR}")

    best: dict[int, dict] = {}
    for p in jsons:
        d = json.loads(p.read_text())
        label = d["configuration"]
        cap   = d.get("chosen_capacity", {})
        n_active = sum(1 for v in cap.values() if v.get("vehicles", 0) > 0)

        if n_active not in best or d["objective"] < best[n_active]["obj"]:
            subset = [] if label == "DC_only" else label.split("+")
            best[n_active] = {
                "label":   label,
                "obj":     d["objective"],
                "subset":  subset,
                "n_active": n_active,
            }
    return best


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    best_configs = _best_per_n_active()
    total = len(best_configs)
    logger.info(f"Running {total} best configurations  N={N_SCENARIOS}  binary_X=True")

    for idx, (n_active, cfg) in enumerate(sorted(best_configs.items()), start=1):
        label  = cfg["label"]
        subset = cfg["subset"]

        out_json = OUTPUT_DIR / f"best_{n_active}active_{label}_N{N_SCENARIOS}.json"
        if out_json.exists():
            logger.info(f"[{idx}/{total}] Skip (exists): {label}")
            continue

        logger.info(
            f"[{idx}/{total}] n_active={n_active}  subset={label}  "
            f"N={N_SCENARIOS}  binary_X={not IS_CONTINUOUS_X}"
        )
        try:
            main = MainCapacitated(
                id_instance=label,
                folder_path=OUTPUT_DIR,
                configuration=(N_SCENARIOS, IS_CONTINUOUS_X),
                max_run_time=MAX_RUN_TIME,
                use_euclidean_distance=True,
                facilities_subset=subset if subset else None,
            )

            logging.disable(logging.CRITICAL)
            path = main.solve()
            logging.disable(logging.NOTSET)

            # Rename to include n_active in filename for clarity
            path.rename(out_json)
            logger.info(f"[{idx}/{total}] Done → {out_json.name}")

        except Exception as e:
            logging.disable(logging.NOTSET)
            logger.error(f"[{idx}/{total}] Failed: {label} — {e}")

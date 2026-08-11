"""Run best operational configurations with N=30 scenarios and integer X variables."""

import json
import logging
from pathlib import Path

from src.constants import RESULTS_DIR
from src.entrypoints.main_capacitated_flex_saa import MainCapacitatedFlex
from src.utils.custom_logger import get_logger

logger = get_logger("BestCapacitatedFlexSAA")

POWERSET_DIR = RESULTS_DIR / "powerset_experiment_capacitated_operational"
OUTPUT_DIR   = RESULTS_DIR / "best_capacitated_operational"

N_SCENARIOS     = 30
IS_CONTINUOUS_X = False
MAX_RUN_TIME    = 60 * 20  # 20 min per config


def _best_per_n_active() -> dict[int, dict]:
    jsons = sorted(POWERSET_DIR.glob("capflex_*_N1.json"))
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
            best[n_active] = {"label": label, "obj": d["objective"], "subset": subset}
    return best


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_configs = _best_per_n_active()
    total = len(best_configs)
    logger.info(f"Running {total} best configs  N={N_SCENARIOS}  binary_X=True")

    for idx, (n_active, cfg) in enumerate(sorted(best_configs.items()), start=1):
        label  = cfg["label"]
        subset = cfg["subset"]
        out_json = OUTPUT_DIR / f"best_{n_active}active_{label}_N{N_SCENARIOS}.json"
        if out_json.exists():
            logger.info(f"[{idx}/{total}] Skip: {label}")
            continue

        logger.info(f"[{idx}/{total}] n_active={n_active}  N={N_SCENARIOS}  binary_X=True  {label}")
        try:
            main = MainCapacitatedFlex(
                id_instance=label,
                folder_path=OUTPUT_DIR,
                configuration=(N_SCENARIOS, IS_CONTINUOUS_X),
                max_run_time=MAX_RUN_TIME,
                use_euclidean_distance=True,
                facilities_subset=subset,
            )
            logging.disable(logging.CRITICAL)
            path = main.solve()
            logging.disable(logging.NOTSET)
            path.rename(out_json)
            logger.info(f"[{idx}/{total}] Done -> {out_json.name}")
        except Exception as e:
            logging.disable(logging.NOTSET)
            logger.error(f"[{idx}/{total}] Failed: {label} - {e}")

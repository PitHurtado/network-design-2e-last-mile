"""Entry point for the Uncapacitated SAA Model."""

from src.constants import RESULTS_DIR
from src.entrypoints.main_uncapacitated_saa import Main
from src.utils.custom_logger import get_logger

logger = get_logger("Run Uncapacitated SAA")

if __name__ == "__main__":
    logger.info("Generating instances")

    # (N, is_continuous_var_x)
    configuration = [
        (1, True),
    ]

    FOLDER_PATH = RESULTS_DIR / "uncapacitated_saa"
    FOLDER_PATH.mkdir(parents=True, exist_ok=True)
    logger.info(f"Total configurations to solve: {len(configuration)}")

    for config in configuration:
        try:
            logger.info(f"Solving instance with configuration: {config}")
            main = Main(
                id_instance=f"1_{config[1]}",
                folder_path=FOLDER_PATH,
                configuration=config,
                id_sampling=None,
                max_run_time=60 * 5,
                use_euclidean_distance=True,
            )
            main.solve()
        except Exception as e:
            logger.info(f"Exception occurred: {e}")

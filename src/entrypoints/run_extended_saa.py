"""Main module for the SAA application."""

from src.constants import RESULTS_DIR, TypeOfFlexibility
from src.entrypoints.main_extended_saa import Main
from src.utils.custom_logger import get_logger

logger = get_logger("Run Extended SAA")

if __name__ == "__main__":
    # (1) Generate instance:
    logger.info("Generating instances")
    configuration = [
        (1, True, TypeOfFlexibility.FLEX_CAPACITY.value),
        (1, False, TypeOfFlexibility.FLEX_CAPACITY.value),
        (1, True, TypeOfFlexibility.FIXED_CAPACITY.value),
        (1, False, TypeOfFlexibility.FIXED_CAPACITY.value),
    ]
    FOLDER_PATH = RESULTS_DIR / "extended_saa"
    logger.info(f"Total configuration to be solved: {len(configuration)}")

    # (3) Solve the instances:
    for config in configuration:
        try:
            logger.info(f"Solving instance with configuration: {config}")
            # main = Main(
            #     id_instance="expected",
            #     folder_path=FOLDER_PATH,
            #     configuration=config,
            #     is_evaluation=True,
            #     # id_sampling=1,
            #     max_run_time=60,
            # )
            main = Main(
                id_instance=f"1_{config[1]}_{config[2]}",
                folder_path=FOLDER_PATH,
                configuration=config,
                is_evaluation=False,
                id_sampling=1,
                max_run_time=60,
            )
            main.solve()
        except Exception as e:
            logger.info(f"Exception occurred: {e}")

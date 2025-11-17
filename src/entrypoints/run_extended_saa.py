"""Main module for the SAA application."""

from src.constants import RESULTS_DIR, TypeOfFlexibility
from src.entrypoints.main_extended_saa import Main
from src.utils.custom_logger import get_logger

logger = get_logger("Run Extended SAA")

if __name__ == "__main__":
    # (1) Generate instance:
    logger.info("Generating instances")
    configuration = [(1, True, TypeOfFlexibility.FLEX_CAPACITY.value)]
    FOLDER_PATH = RESULTS_DIR / "extended_saa"
    logger.info(f"Total configuration to be solved: {len(configuration)}")

    # (3) Solve the instances:
    for config in configuration:
        try:
            logger.info(f"Solving instance with configuration: {config}")
            # main = Main(
            #     id_instance="extended_saa_1", folder_path=FOLDER_PATH, configuration=config, is_evaluation=True, max_run_time=10 #noqa: E501
            # )
            main = Main(
                id_instance="extended_saa_2",
                folder_path=FOLDER_PATH,
                configuration=config,
                is_evaluation=False,
                id_sampling=1,
                max_run_time=60,
            )
            main.solve()
        except Exception as e:
            logger.info(f"Exception occurred: {e}")

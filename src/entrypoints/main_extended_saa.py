"""Module of class Main Extended SAA Model."""

import json
import logging
from typing import Optional

from src.models.extended_saa_model import ExtendedSAAModel
from src.utils.custom_logger import get_logger
from src.utils.instance import Instance

logger = get_logger("Main Extended SAA")


class Main:
    """Main class to solve Extended SAA Model."""

    def __init__(
        self,
        id_instance: str,
        folder_path: str,
        configuration: tuple,
        max_run_time: int,
        is_evaluation: bool = False,
        id_sampling: Optional[int] = None,
    ):
        self.folder_path = folder_path
        self.configuration = configuration
        self.max_run_time = max_run_time

        logger.info(f"Loading instance: {configuration}")
        (
            N,
            is_continuous_var_x,
            type_of_flexibility,
        ) = configuration

        self.instance = Instance(
            id_instance=id_instance,
            is_continuous_var_x=is_continuous_var_x,
            type_of_flexibility=type_of_flexibility,
            periods=12,
            N=N,
            is_evaluation=is_evaluation,
            id_sampling=id_sampling,
        )
        logger.info(f"Instance Loaded: {configuration}")

    def solve(self):
        """Solve Extended SAA Model."""
        logger.info(f"Starting solving Extended SAA Model for instance {self.instance.id_instance}")

        # (1) Create model:
        solver = ExtendedSAAModel(self.instance)
        solver.build()
        params_config = {
            "TimeLimit": self.max_run_time,
            # "MIPGap": 0.0005
        }
        solver.set_params(params_config)

        # (2) Solve model:
        logger.disabled = True
        logging.disable(logging.CRITICAL)
        solver_metrics = solver.solve()
        logger.disabled = False
        logging.disable(logging.NOTSET)
        logger.info("Solving ended")

        # (3) Save results:
        results = {
            "objective": solver.obj.cost_total.getValue(),
            "cost_installation_facilities": solver.obj.cost_installation_facilities.getValue(),
            "cost_operating_facilities": solver.obj.cost_operating_facilities.getValue(),
            "cost_served_from_facilities": solver.obj.cost_served_from_facilities.getValue(),
            "cost_served_from_dc": solver.obj.cost_served_from_dc.getValue(),
            "scenarios": self.instance.scenarios_ids,
            # "Instance information": self.get_information(run_time), # MUST BE UNPACKED
            "Solver information": solver_metrics,
            "Y": {str(keys): value.X for keys, value in solver.model._Y.items()},
            "X": {str(keys): value.X for keys, value in solver.model._X.items()},
            "Z": {str(keys): value.X for keys, value in solver.model._Z.items()},
            "W": {str(keys): value.X for keys, value in solver.model._W.items()},
        }

        results.update(self.get_information())

        path_file_output = (
            self.folder_path / f"extended_saa_{self.instance.id_instance}_{self.instance.id_sampling}.json"
        )  # noqa E501
        with open(path_file_output, "w") as file:
            file.write(json.dumps(results, indent=4))

        logger.info(f"Results saved in {path_file_output}")

    def get_information(self):
        """Return a dictionary with info combination."""
        (
            N,
            is_continuous_x,
            type_of_flexibility,
        ) = self.configuration

        info_combination = {
            "id_sampling": self.instance.id_sampling,
            "configuration": self.instance.id_instance,
            "N": N,
            "is_continuous_x": is_continuous_x,
            "type_of_flexibility": type_of_flexibility,
            "periods": 12,
            "max_run_time": self.max_run_time,
        }
        return info_combination

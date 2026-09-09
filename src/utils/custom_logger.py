"""Module for creating a customizable logger with consistent formatting across the project."""

import logging
import sys
from typing import Optional


class CustomLogger:
    """
    Crea un logger configurable con formato consistente para todo el proyecto.
    Ejemplo:
        from utils.custom_logger import get_logger
        logger = get_logger(__name__)
        logger.info("Mensaje de prueba")
    """

    def __init__(
        self,
        name: str,
        level: int = logging.INFO,
        log_format: Optional[str] = None,
        date_format: str = "%Y-%m-%d %H:%M:%S",
        use_colors: bool = True,
    ):
        self.name = name
        self.level = level
        self.log_format = log_format or "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        self.date_format = date_format
        self.use_colors = use_colors
        self.logger = self._create_logger()

    def _create_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.name)
        logger.setLevel(self.level)

        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(self.level)
            formatter = self._get_formatter()
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        logger.propagate = False
        return logger

    def _get_formatter(self):
        if self.use_colors:
            try:
                from colorama import Fore, Style

                RESET = Style.RESET_ALL
                COLORS = {
                    "DEBUG": Fore.BLUE,
                    "INFO": Fore.GREEN,
                    "WARNING": Fore.YELLOW,
                    "ERROR": Fore.RED,
                    "CRITICAL": Fore.MAGENTA,
                }

                class ColorFormatter(logging.Formatter):
                    def format(self, record):
                        color = COLORS.get(record.levelname, "")
                        record.levelname = f"{color}{record.levelname}{RESET}"
                        return super().format(record)

                return ColorFormatter(self.log_format, self.date_format)
            except ImportError:
                pass  # Si colorama no está instalado, usa formato normal

        return logging.Formatter(self.log_format, self.date_format)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Función de acceso rápido para obtener un logger consistente.
    Ejemplo:
        logger = get_logger(__name__)
    """
    return CustomLogger(name=name, level=level).logger

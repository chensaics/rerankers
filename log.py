import sys
import os
import logging
from loguru import logger as Logger

# from colorama import Fore
# from __future__ import annotations
# from pathlib import PurePath

# COLORS = [
#     Fore.BLACK,
#     Fore.RED,
#     Fore.GREEN,
#     Fore.YELLOW,
#     Fore.BLUE,
#     Fore.MAGENTA,
#     Fore.CYAN,
#     Fore.WHITE,
# ]

# #######################################################
# loguru
# #######################################################
# loguru format
fmt = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

_SET_UP_LOGGERS = set()


def get_logger(name: str = "agent") -> logging.Logger:
    """Get logger. Use this instead of `logging.getLogger` to ensure
    that the logger is set up with the correct handlers.

    https://rich.readthedocs.io/en/stable/reference/logging.html#logging
    """
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    # INFO ERROR DEBUG WARNING
    Logger.add(f"./logs/{name}.log", format=fmt, level="INFO", rotation="500 MB")
    Logger.add(sys.stderr, format=fmt, level="WARNING")
    Logger.propagate = False

    if name in _SET_UP_LOGGERS:  # Already set up
        return Logger

    _SET_UP_LOGGERS.add(name)

    return Logger

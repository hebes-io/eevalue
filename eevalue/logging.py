# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import logging
import sys

from colorlog import ColoredFormatter

formatter = ColoredFormatter(
    "%(log_color)s%(asctime)s | %(name)s | %(levelname)s | %(message_log_color)s%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    reset=True,
    log_colors={
        "DEBUG": "light_cyan",
        "INFO": "green",
        "WARNING": "light_yellow",
        "ERROR": "light_red",
        "CRITICAL": "red,bg_white",
    },
    secondary_log_colors={
        "message": {
            "DEBUG": "white",
            "INFO": "white",
            "WARNING": "white",
            "ERROR": "white",
            "CRITICAL": "white",
        }
    },
    style="%",
)

handler = logging.StreamHandler(stream=sys.stdout)
handler.setFormatter(formatter)
logging.basicConfig(level=logging.WARNING, handlers=[handler])


def getLogger(name: str):
    return logging.getLogger(name)

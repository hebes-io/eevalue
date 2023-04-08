# -*- coding: utf-8 -*-

import importlib.resources

import eevalue

with importlib.resources.path(eevalue, "__main__.py") as main_path:
    SOURCE_PATH = main_path.resolve().parent

DATA_DIR = SOURCE_PATH / "artifacts"

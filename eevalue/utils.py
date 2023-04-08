# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import calendar
import glob
import os
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.utils.validation import column_or_1d

from eevalue.exceptions import NotEnoughData
from eevalue.settings import SOURCE_PATH


def _convert_paths_to_absolute_posix(parent_path, conf_dictionary):
    if not parent_path.is_absolute():
        raise ValueError(
            f"parent_path must be an absolute path. Received: {parent_path}"
        )

    # only check a few conf keys that are known to specify a path string as value
    conf_keys_with_filepath = ("FileName", "FilePath", "Path")

    for conf_key, conf_value in conf_dictionary.items():
        # if the conf_value is another dictionary, absolutify its paths first.
        if isinstance(conf_value, dict):
            conf_dictionary[conf_key] = _convert_paths_to_absolute_posix(
                parent_path, conf_value
            )
            continue

        # if the conf_value is not a dictionary nor a string, skip
        if not isinstance(conf_value, str):
            continue

        # if the conf_value is a string but the conf_key isn't one associated with filepath, skip
        if conf_key not in conf_keys_with_filepath:
            continue

        conf_value_absolute_path = (parent_path / conf_value).as_posix()
        conf_dictionary[conf_key] = conf_value_absolute_path

    return conf_dictionary


def as_list(val: Any):
    """Cast input as list.

    Helper function, always returns a list of the input value.
    """
    if isinstance(val, str):
        return [val]
    if hasattr(val, "__iter__"):
        return list(val)
    if val is None:
        return []
    return [val]


def as_series(x: Union[np.ndarray, pd.Series, pd.DataFrame]):
    """Cast an iterable to a Pandas Series object."""
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0]
    else:
        return pd.Series(column_or_1d(x))


def create_period(start_date: str, end_date: str, date_format: str = "%d-%m-%Y"):
    return pd.date_range(
        start=pd.to_datetime(start_date, format=date_format),
        end=pd.to_datetime(end_date, format=date_format) + timedelta(days=1),
        freq="H",
        inclusive="left",
    )


def expand_dates(data: pd.DataFrame, fillna=True):
    _, last_day = calendar.monthrange(data.index.max().year, data.index.max().month)
    full_index = pd.date_range(
        start=datetime.combine(
            date(data.index.min().year, data.index.min().month, 1), time(0, 0)
        ),
        end=datetime.combine(
            date(data.index.max().year, data.index.max().month, last_day),
            time(23, 0),
        ),
        freq="H",
        inclusive="both",
    )

    if fillna:
        return data.reindex(full_index).fillna(method="ffill").fillna(method="bfill")
    else:
        return data.reindex(full_index)


def match_period(data: pd.DataFrame, period: pd.DatetimeIndex):
    diff = period.difference(data.index)

    if not diff.empty:
        raise NotEnoughData(
            "The index of the available data does not include the whole `period`. "
            "Update the start and end dates so that they do not include the missing data. "
            f"Missing data index: {diff}. Missing data length: {len(diff)}"
        )

    data = data.reindex(period)
    return data


def cvrmse(y_true: np.ndarray, y_pred: np.ndarray):
    """Compute the coefficient of variation of the Root Mean Squared Error.

    Args:
        y_true (np.ndarray): Ground truth (correct) target values.
        y_pred (np.ndarray): Estimated target values.

    Returns:
        float: The metric's value.
    """
    resid = column_or_1d(y_true) - column_or_1d(y_pred)
    return float(np.sqrt((resid**2).sum() / len(resid)) / np.mean(y_true))


def load_config(config: Path = None, country: str = None):
    """Load a  YAML configuration file.

    Args:
        config (Path, optional): The path to the YAML configuration file. Defaults
            to None.
        country (str, optional): The name of the country (ISO 3166 alpha-2 codes,
            except for Greece, for which the abbreviation EL is used). If `config`
            is None, `country` must be provided. Defaults to None.
    """
    if config is None:
        if country is None:
            raise ValueError("If `config` is None then `country` must be provided.")
        types = ("*.yaml", "*.yml")
        path = os.path.join(SOURCE_PATH, "config", country)
        config_files = []
        for file_type in types:
            config_files.extend(glob.glob(path + f"/{file_type}"))
        config = OmegaConf.to_container(
            OmegaConf.merge(*[OmegaConf.load(file) for file in config_files])
        )
        config = _convert_paths_to_absolute_posix(Path(path), config)
    else:
        path = config.parent
        config = OmegaConf.to_container(OmegaConf.load(config))
        config = _convert_paths_to_absolute_posix(path, config)

    return config

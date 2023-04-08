# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import numpy as np
import pandas as pd


def create_markup_state(
    *,
    load: pd.Series,
    res: pd.Series,
    net_imports: pd.Series,
    capacity: pd.DataFrame,
    water_value: pd.Series = None,
):
    """Transform market data into states for learning a markup function.

    Args:
        load (pandas.Series): The electricity demand time series.
        res (pandas.Series): The renewable power generation time series.
        net_imports (pandas.Series): The net imports time series.
        capacity (pandas.DataFrame): The available capacity per cluster.
        water_value (pandas.Series, optional): The value of water time series
            (deviation between reservoir levels and their long-term average values).
            Defaults to None.
    Returns:
        pandas.DataFrame: The dataframe of the generated states.
    """
    state = []

    state.append(
        load.sub(res)
        .add(net_imports)
        .divide(capacity.sum(axis=1), axis=0)
        .to_frame("Margin")
    )

    if water_value is not None:
        state.append(water_value.to_frame("WaterValue"))

    state = pd.concat(state, axis=1)
    return state


def create_hydro_state(*, availability_factor: pd.DataFrame, water_value: pd.Series):
    """Transform market data into states for learning a markup function.

    Args:
        availability_factor (pandas.DataFrame): The available factor for hydropower
            generation.
        water_value (pandas.Series): The value of water time series (deviation
            between reservoir levels and their long-term average values).
    Returns:
        pandas.DataFrame: The dataframe of the generated states.
    """
    state = pd.concat((availability_factor, water_value.to_frame("WaterValue")), axis=1)
    return state


def create_markup_trials(
    *,
    period: pd.DatetimeIndex,
    clusters: List[str],
    active_clusters: List[str] = None,
    min_value: int = 0,
    max_value: int = 100,
):
    """Create trial combinations for the markup.

    Args:
        period (pandas.DatetimeIndex with freq='H'): The period over which to generate
            the data.
        clusters (list of str): The names of all the generation clusters.
        active_clusters (list of str, optional): The names of the generation clusters
            for which markup should be considered. If None, markup will be generated
            for all clusters. Defaults to None.
        min_value (int, optional): The minimum value for the markup. Defaults to 0.
        max_value (int, optional): The maximum value for the markup. Defaults to 100.

    Returns:
        pandas.DataFrame: The dataframe of the generated markup combinations.
    """
    if active_clusters is None:
        markup = pd.DataFrame(0, index=period, columns=clusters)
        markup[:] = np.random.randint(min_value, high=max_value, size=markup.shape)
    else:
        markup = pd.DataFrame(0, index=period, columns=clusters)
        for col in active_clusters:
            markup[col] = np.random.randint(
                min_value, high=max_value, size=markup.shape[0]
            )
    return markup


def create_hydro_trials(*, period: pd.DatetimeIndex):
    """Create trial combinations for the effective availability factor for hydropower.

    Args:
        period (pandas.DatetimeIndex with freq='H'): The period over which to generate
            the data.

    Returns:
        pandas.Series: The series of the effective availability factor for hydropower.
    """

    return pd.Series(
        data=np.random.uniform(0, high=0.99, size=len(period)),
        index=period,
    )

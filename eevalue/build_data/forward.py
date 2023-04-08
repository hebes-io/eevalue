# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import logging
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd

from eevalue.utils import expand_dates, match_period

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eevalue")


def sample_historical(
    *,
    period: pd.DatetimeIndex,
    data: pd.Series,
    group: Literal["season", "month", "week"],
) -> Callable:
    """Generate scenarios by sampling from historical data.

    Args:
        period (pandas.DatetimeIndex with freq='H'): The period over which to generate
            the data.
        data (pandas.Series): The historical data to sample from.
        group ({"season", "month", "week"}): Parameter to define how to group historical data
            before sampling.

    Returns:
        Callable: Function that returns scenarios.
    """

    data = data.to_frame("values")
    data["date"] = data.index.date
    data["hour"] = data.index.hour

    result = pd.DataFrame(0, index=np.unique(period.date), columns=range(24))

    if group == "season":
        data["group"] = data.index.month % 12 // 3 + 1
        grouped = result.groupby(lambda x: x.month % 12 // 3 + 1)
    elif group == "month":
        data["group"] = data.index.month
        grouped = result.groupby(lambda x: x.month)
    elif group == "week":
        data["group"] = data.index.isocalendar().week
        grouped = result.groupby(lambda x: x.isocalendar().week)
    else:
        raise ValueError(f"Group {group} not supported")

    def scen_gen():
        for i, group in grouped:
            selected = data.loc[data["group"] == i]
            selected = selected.pivot(
                index="date", columns="hour", values="values"
            ).sample(n=len(group), replace=True)
            result.loc[result.index.isin(group.index)] = selected.values

        sample = result.stack()
        sample.index = sample.index.map(
            lambda x: datetime.datetime.combine(x[0], datetime.time(int(x[1])))
        )
        return sample.sort_index()

    return scen_gen


def sample_yearly(
    *, period: pd.DatetimeIndex, filepath: str, multiplier: float = None
) -> Callable:
    """Generate scenarios from yearly scenario data.

    Args:
        period (pandas.DatetimeIndex with freq='H'): The period over which to generate
            the data.
        filepath (str): The path where the scenario data csv file can be found.
        multiplier (float, optional): If provided, each year's data will be multiplied
            by a random number in [1, `multiplier`). Defaults to None.

    Returns:
        Callable: Function that returns scenarios.
    """
    result = (
        pd.read_csv(filepath, index_col="ds", parse_dates=["ds"])
        .resample("H")
        .asfreq()
        .pipe(expand_dates, fillna=False)
    )

    na_mapping = (
        result.loc[(result.index.dayofyear == 1) & (result.index.hour == 0)]
        .interpolate(method="slinear")
        .pipe(expand_dates, fillna=True)
    )

    result = (
        result.mask(result.isna(), na_mapping)
        .pipe(match_period, period=period)
        .iloc[:, 0]
    )

    def scen_gen():
        if multiplier is not None:
            return result.groupby(lambda x: x.year).apply(
                lambda x: x * np.random.uniform(1, multiplier)
            )
        else:
            return result

    return scen_gen


def sample_components(
    *, period: pd.DatetimeIndex, country: str, data_dir: Path
) -> Callable:
    """Sample the principal components for forward simulation of hourly time series.

    Args:
        period (pandas.DatetimeIndex with freq='H'): The period over which to generate
            the data.
        country (str): The country to load the data for (in ISO 3166-1 alpha-2 format).
        data_dir (pathlib.Path): Path to data generated by the `preprocess` stage.

    Returns:
        Callable: Function that generates component samples.
    """

    path = data_dir / "components" / country
    U = pd.read_csv(path / "components.csv", index_col="ds", parse_dates=["ds"])
    grouped = U.groupby([lambda x: x.month % 12 // 3 + 1, lambda x: x.dayofweek])

    samples = pd.DataFrame(
        0, index=period.to_frame().resample("1D").first().index, columns=U.columns
    )
    samples["season"] = samples.index.month % 12 // 3 + 1
    samples["dow"] = samples.index.dayofweek

    def scen_gen():
        for (season, dow), group in grouped:
            selected = samples.loc[
                (samples["season"] == season) & (samples["dow"] == dow)
            ]
            samples.loc[selected.index, group.columns] = group.sample(
                n=selected.shape[0], replace=True
            ).values
        return samples.drop(["season", "dow"], axis=1)

    return scen_gen

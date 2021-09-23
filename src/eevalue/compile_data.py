# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import math
import os
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Union

import dateparser
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy.signal import savgol_filter

from eevalue.settings import CONF_PATH, DATA_PATH
from eevalue.utils import as_list, as_series

###################################################################################
# Utilities
###################################################################################


def years_to_hours(data: pd.DataFrame):
    full_index = pd.date_range(
        start=datetime.combine(date(data.index.min().year, 1, 1), time(0, 0)),
        end=datetime.combine(
            date(data.index.max().year, 12, 31) + timedelta(days=1), time(0, 0)
        ),
        freq="H",
    )[:-1]
    return data.reindex(full_index).fillna(method="ffill")


def months_to_hours(data: pd.DataFrame):
    full_index = pd.date_range(
        start=datetime.combine(
            date(data.index.min().year, data.index.min().month, 1), time(0, 0)
        ),
        end=datetime.combine(
            date(data.index.max().year, data.index.max().month + 1, 1),
            time(0, 0),
        ),
        freq="H",
    )[:-1]
    return data.reindex(full_index).fillna(method="ffill")


def days_to_hours(data: pd.DataFrame):
    full_index = pd.date_range(
        start=datetime.combine(data.index.min().date(), time(0, 0)),
        end=datetime.combine(data.index.max().date() + timedelta(days=1), time(0, 0)),
        freq="H",
    )[:-1]
    return data.reindex(full_index).fillna(method="ffill")


def match_period(data: pd.DataFrame, period: pd.DatetimeIndex):
    time_col = data.index.to_series()
    time_step = time_col.diff().min()

    if time_step <= pd.Timedelta("1H"):
        pass
    elif time_step == pd.Timedelta("1D"):
        data = data.pipe(days_to_hours)
    elif time_step <= pd.Timedelta("31D"):
        data = data.pipe(months_to_hours)
    elif time_step <= pd.Timedelta("366D"):
        data = data.pipe(years_to_hours)
    else:
        raise ValueError("Frequency of input data not understood.")

    data = data.reindex(period).interpolate("nearest")
    if data.isnull().values.any():
        raise ValueError(
            "NaN values in data. Probably, the data index does not cover the whole `period`."
        )
    return data


###################################################################################
# Get individual datasets
###################################################################################


def get_availability_factors(
    period: pd.DatetimeIndex, filepath: str, filename: Union[str, List[str]] = None
):
    avail_factors = None
    for name in as_list(filename):
        x = pd.read_csv(
            os.path.join(filepath, name + ".csv"), index_col=0, parse_dates=[0]
        )
        x = x["Available capacity [MW]"] / x["Nominal capacity [MW]"]
        x = x.to_frame(name)
        avail_factors = pd.concat(
            [
                avail_factors,
                x.pipe(match_period, period=period),
            ],
            axis=1,
        )
    return avail_factors


def get_committed_capacity(
    period: pd.DatetimeIndex, filepath: str, filename: Union[str, List[str]] = None
):
    committed_capacity = None
    for name in as_list(filename):
        x = pd.read_csv(
            os.path.join(filepath, name + ".csv"), index_col=0, parse_dates=[0]
        )
        x = x.rename(columns={"Capacity committed [MW]": name})
        committed_capacity = pd.concat(
            [
                committed_capacity,
                x.pipe(match_period, period=period),
            ],
            axis=1,
        )
    return committed_capacity


def get_res_generation(
    period: pd.DatetimeIndex, filepath: str, filename: Union[str, List[str]] = None
):
    res_generation = None
    filenames = as_list(filename)
    if len(filenames) > 1:
        raise ValueError("More than 1 files found for RES generation data")
    elif len(filenames) == 1:
        res_generation = pd.read_csv(
            os.path.join(filepath, filename + ".csv"), index_col=0, parse_dates=[0]
        )
        res_generation = res_generation.pipe(match_period, period=period)
        res_generation = as_series(res_generation).to_frame("res_generation")
    return res_generation


def get_available_capacity(
    period: pd.DatetimeIndex, filepath: str, filename: Union[str, List[str]] = None
):
    avail_capacity = None
    for name in as_list(filename):
        x = pd.read_csv(
            os.path.join(filepath, name + ".csv"), index_col=0, parse_dates=[0]
        )
        x = x.drop("Nominal capacity [MW]", axis=1).rename(
            columns={"Available capacity [MW]": name}
        )
        avail_capacity = pd.concat(
            [
                avail_capacity,
                x.pipe(match_period, period=period),
            ],
            axis=1,
        )
    return avail_capacity


def get_demand(
    period: pd.DatetimeIndex, filepath: str, load: str, reserves: str = None
):
    load = pd.read_csv(
        os.path.join(filepath, load + ".csv"), index_col=0, parse_dates=[0]
    )

    if isinstance(load, pd.DataFrame) and load.columns[0] != "DA":
        load = load.rename(columns={load.columns[0]: "DA"})
    elif isinstance(load, pd.Series):
        load = load.to_frame("DA")

    load = load.pipe(match_period, period=period)

    if reserves is not None:
        resv = pd.read_csv(
            os.path.join(filepath, reserves + ".csv"), index_col=0, parse_dates=[0]
        )
        resv_2U = resv["2U"].pipe(match_period, period=period)
        resv_2D = resv["2D"].pipe(match_period, period=period)
        demand = pd.concat([load, resv_2U, resv_2D], axis=1)
    else:
        resv = (
            load[["DA"]]
            .groupby(lambda x: x.date)
            .apply(lambda x: np.sqrt(10 * x.max() + 22500) - 150)
        )
        resv.index = resv.index.map(pd.to_datetime)
        resv = resv.pipe(match_period, period=period)
        demand = pd.concat(
            [
                load,
                resv.rename(columns={resv.columns[0]: "2U"}),
                resv.multiply(0.5).rename(columns={resv.columns[0]: "2D"}),
            ],
            axis=1,
        )
    return demand


def get_net_imports(
    period: pd.DatetimeIndex, filepath: str, filename: Union[str, List[str]] = None
):
    imports = None
    filenames = as_list(filename)

    if len(filenames) > 1:
        raise ValueError("More than 1 files found for net imports data")
    elif len(filenames) == 1:
        imports = pd.read_csv(
            os.path.join(filepath, filename + ".csv"), index_col=0, parse_dates=[0]
        )
        imports = imports.pipe(match_period, period=period)
        imports = imports.rename(columns={imports.columns[0]: "net_imports"})

    return imports


def get_water_value(
    period: pd.DatetimeIndex, filepath: str, filename: Union[str, List[str]] = None
):
    water_value = None
    filenames = as_list(filename)

    if len(filenames) > 1:
        raise ValueError("More than 1 files found for water data")
    elif len(filenames) == 1:
        water_value = pd.read_csv(
            os.path.join(filepath, filename + ".csv"),
            index_col=0,
            parse_dates=[0],
            na_values=0,
        )
        water_value = water_value["Average"] / water_value["Filling Rate"]
        water_value = water_value.pipe(days_to_hours).pipe(match_period, period=period)
        window_length = math.ceil(0.05 * len(water_value)) // 2 * 2 + 1
        water_value = pd.DataFrame(
            data=savgol_filter(water_value.values.squeeze(), window_length, 3),
            index=period,
            columns=["water_value"],
        )
    return water_value


def get_fuel_prices(
    config: Dict,
    period: pd.DatetimeIndex,
    filepath: str,
    filename: Union[str, List[str]] = None,
):
    fuel_prices = None

    for name in as_list(filename):
        x = pd.read_csv(
            os.path.join(filepath, name + ".csv"), index_col=0, parse_dates=[0]
        )
        x = x.rename(columns={x.columns[0]: name})
        fuel_prices = pd.concat(
            [
                fuel_prices,
                x.pipe(match_period, period=period),
            ],
            axis=1,
        )

    if "FuelPrice" in config:
        fuel_prices_other = pd.DataFrame.from_dict(config["FuelPrice"], orient="index")
        fuel_prices_other.index = fuel_prices_other.index.astype("str")

        fuel_prices_other.index = fuel_prices_other.index.map(
            lambda x: dateparser.parse(
                x,
                settings={
                    "PREFER_DAY_OF_MONTH": "first",
                    "RELATIVE_BASE": datetime(2000, 1, 1),
                },
            )
        )
        fuel_prices_other = fuel_prices_other.pipe(match_period, period=period)
        if np.intersect1d(fuel_prices.columns, fuel_prices_other.columns):
            raise ValueError("Found common fuels between different data sources")
        fuel_prices = pd.concat([fuel_prices, fuel_prices_other], axis=1)

    return fuel_prices


def get_permit_prices(
    config: Dict,
    period: pd.DatetimeIndex,
    filepath: str,
    filename: Union[str, List[str]] = None,
):
    permit_prices = None
    filenames = as_list(filename)

    if len(filenames) > 1:
        raise ValueError("More than 1 files found for permit price data data")
    elif len(filenames) == 1:
        permit_prices = pd.read_csv(
            os.path.join(filepath, filename + ".csv"), index_col=0, parse_dates=[0]
        )
        permit_prices = permit_prices.rename(
            columns={permit_prices.columns[0]: "permit_price"}
        ).pipe(match_period, period=period)
    elif "PermitPrice" in config:
        permit_prices = pd.DataFrame.from_dict(config["PermitPrice"], orient="index")
        permit_prices.index = permit_prices.index.astype("str")
        permit_prices.index = permit_prices.index.map(
            lambda x: dateparser.parse(
                x,
                settings={
                    "PREFER_DAY_OF_MONTH": "first",
                    "RELATIVE_BASE": datetime(2000, 1, 1),
                },
            )
        )
        permit_prices = permit_prices.rename(
            columns={permit_prices.columns[0]: "permit_price"}
        ).pipe(match_period, period=period)

    return permit_prices


###################################################################################
# Create the dataset for the simulation
###################################################################################


def create_dataset(
    *,
    country,
    period,
    include=None,
    exclude=None,
    calibration=False,
    has_hydro=True,
    return_flat=False,
) -> Dict:

    # Load data catalog
    catalog_files = glob.glob(os.path.join(CONF_PATH, country) + "/catalog*")

    if not catalog_files:
        raise ValueError(f"Data catalog for country {country} not found")

    catalog = OmegaConf.merge(*[OmegaConf.load(file) for file in catalog_files])

    if OmegaConf.is_missing(catalog, "DATA_DIR"):
        catalog.DATA_DIR = str(DATA_PATH)
    if OmegaConf.is_missing(catalog, "country"):
        catalog.country = country

    catalog = OmegaConf.to_container(catalog, resolve=True)["sources"]

    # Load additional configuration data
    if calibration:
        config_files = glob.glob(
            os.path.join(CONF_PATH, country) + "/calibration/config*"
        )
    else:
        config_files = glob.glob(
            os.path.join(CONF_PATH, country) + "/scenarios/config*"
        )

    if not config_files:
        raise ValueError(f"Configuration data for country {country} not found")

    config = OmegaConf.merge(*[OmegaConf.load(file) for file in config_files])
    config = OmegaConf.to_container(config)

    if (include is not None) and not isinstance(include, list):
        include = [include]
    elif include is None:
        include = [
            "availability_factors",
            "capacity_margin",
            "committed_capacity",
            "res_generation",
            "available_capacity",
            "demand",
            "net_load",
            "net_imports",
            "plants",
            "ramping_events",
            "fuel_price",
            "permit_price",
            "no_load_cost",
            "water_value",
        ]

    if (exclude is not None) and not isinstance(exclude, list):
        exclude = [exclude]
    elif exclude is None:
        exclude = []

    include = [x for x in include if x not in exclude]

    cache = set()
    if ("net_load" in include) and ("demand" not in include):
        cache.add("demand")
    if ("net_load" in include) and ("res_generation" not in include):
        cache.add("res_generation")
    if ("capacity_margin" in include) and ("available_capacity" not in include):
        cache.add("available_capacity")
    if ("capacity_margin" in include) and ("net_load" not in include):
        cache.update(["demand", "res_generation"])
    if ("ramping_events" in include) and ("net_load" not in include):
        cache.update(["demand", "res_generation"])

    time_series = {}

    # Create all time series
    time_series["simulation_period"] = period

    # Availability factors
    if "availability_factors" in include:
        filepath = catalog["capacity"].get("filepath")
        filename = catalog["capacity"].get("filename")
        time_series["availability_factors"] = get_availability_factors(
            period, filepath, filename
        )

    # Historical data for power generation
    if "committed_capacity" in include:
        filepath = catalog["generation"].get("filepath")
        filename = catalog["generation"].get("filename")
        time_series["committed_capacity"] = get_committed_capacity(
            period, filepath, filename
        )

    # Historical data for generation from renewables
    if ("res_generation" in include) or ("res_generation" in cache):
        filepath = catalog["res_generation"].get("filepath")
        filename = catalog["res_generation"].get("filename")
        time_series["res_generation"] = get_res_generation(period, filepath, filename)

    # Historical data for available capacity
    if ("available_capacity" in include) or ("available_capacity" in cache):
        filepath = catalog["capacity"].get("filepath")
        filename = catalog["capacity"].get("filename")
        time_series["available_capacity"] = get_available_capacity(
            period, filepath, filename
        )

    # Demand data
    if ("demand" in include) or ("demand" in cache):
        filepath = catalog["load"].get("filepath")
        load_file = catalog["load"].get("filename")
        reserves_file = catalog["reserves"].get("filename")
        time_series["demand"] = get_demand(period, filepath, load_file, reserves_file)

    if "net_load" in include:
        load = time_series["demand"]["DA"]
        res_generation = as_series(time_series["res_generation"])
        net_load = load - res_generation
        time_series["net_load"] = net_load.to_frame("net_load")

    if "capacity_margin" in include:
        available_capacity = time_series["available_capacity"]
        try:
            net_load = as_series(time_series["net_load"])
        except KeyError:
            load = time_series["demand"]["DA"]
            res_generation = as_series(time_series["res_generation"])
            net_load = load - res_generation

        total_available = available_capacity.sum(axis=1)
        assert net_load.shape == total_available.shape
        time_series["capacity_margin"] = net_load.divide(
            total_available, axis=0
        ).to_frame("capacity_margin")

    if "ramping_events" in include:
        try:
            net_load = as_series(time_series["net_load"])
        except KeyError:
            load = time_series["demand"]["DA"]
            res_generation = as_series(time_series["res_generation"])
            net_load = load - res_generation

        ramps = (
            net_load - (net_load.shift(1) + net_load.shift(2) + net_load.shift(3)) / 3
        )
        ramp_up = ramps[ramps >= ramps[ramps > 0].quantile(0.90)]
        ramp_down = ramps[ramps <= ramps[ramps < 0].quantile(0.10)]

        ramps = ramps.to_frame("ramping")
        ramps["RU"] = 0
        ramps["RD"] = 0
        ramps.loc[ramps.index.isin(ramp_up.index), "RU"] = 1
        ramps.loc[ramps.index.isin(ramp_down.index), "RD"] = 1
        time_series["ramping_events"] = ramps.drop("ramping", axis=1)

    # Net imports
    if "net_imports" in include:
        filepath = catalog["imports"].get("filepath")
        filename = catalog["imports"].get("filename")
        time_series["net_imports"] = get_net_imports(period, filepath, filename)

    # Water value
    if "water_value" in include:
        filepath = catalog["waters"].get("filepath")
        filename = catalog["waters"].get("filename")
        time_series["water_value"] = (
            get_water_value(period, filepath, filename) if has_hydro else None
        )

    if "fuel_price" in include:
        filepath = (
            catalog["fuel_prices"].get("filepath") if "fuel_prices" in catalog else None
        )
        filename = (
            catalog["fuel_prices"].get("filename") if "fuel_prices" in catalog else None
        )
        time_series["fuel_price"] = get_fuel_prices(config, period, filepath, filename)

    if "permit_price" in include:
        filepath = (
            catalog["permit_prices"].get("filepath")
            if "permit_prices" in catalog
            else None
        )
        filename = (
            catalog["permit_prices"].get("filename")
            if "permit_prices" in catalog
            else None
        )
        time_series["permit_price"] = get_permit_prices(
            config, period, filepath, filename
        )

    for key in cache - set(include):
        del time_series[key]

    non_time_series = {}
    # Plant clusters
    if "plants" in include:
        path = os.path.join(DATA_PATH, f"PowerPlants/{country}")
        clustered_plants = pd.read_csv(os.path.join(path, "clustered.csv"))
        non_time_series["clustered_plants"] = clustered_plants.drop("Units", axis=1)

        non_time_series["reserve_technologies"] = {}
        if "ReserveParticipation" in config:
            for tech in clustered_plants["Technology"].unique().tolist():
                non_time_series["reserve_technologies"].update(
                    {tech: 1 if tech in config["ReserveParticipation"] else 0}
                )
        else:
            for tech in clustered_plants["Technology"].unique().tolist():
                non_time_series["reserve_technologies"].update({tech: 1})

    if ("no_load_cost" in include) and ("NoLoadCost" in config):
        non_time_series["no_load_cost"] = config["NoLoadCost"]

    if return_flat:
        dataset = {**time_series, **non_time_series}
    else:
        dataset = {"time_series": time_series, "non_time_series": non_time_series}
    return dataset


def subsample_dataset(dataset, period):
    output = {}
    for name, df in dataset["time_series"].items():
        if name == "simulation_period":
            output[name] = period
        else:
            output[name] = df.pipe(match_period, period=period)
    output.update(dataset["non_time_series"])
    return output

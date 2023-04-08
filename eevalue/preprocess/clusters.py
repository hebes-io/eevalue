# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import datetime
from typing import List, Union

import intake
import numpy as np
import pandas as pd

###########################################################################################
# Create clusters
###########################################################################################


def group_plants(country: str, years: List[int], plan: pd.DataFrame = None):
    """Create and store power plant clusters.

    Args:
        country (str): The country to load the plant data for (in ISO 3166-1 alpha-2 format).
        years (list of int): The years to consider.
        plan (pd.DataFrame, optional): Plan of future entries and exits of generation plans.
            Defaults to None.

    Returns:
        dict: A dictionary that associates each year in `years` with clustered plant data.
    """
    plants = intake.cat["markets"].plants(country=country).read()

    if plan is not None:
        plants["Entry"] = 0
        for i, (year, capacity, fuel) in plan.iterrows():
            updated = {}
            updated["Unit"] = f"New_{i}"
            updated["Entry"] = year
            updated["Exit"] = np.nan
            updated["Fuel"] = fuel
            updated["PowerCapacity"] = capacity
            for item in [
                "Efficiency",
                "MinUpTime",
                "MinDownTime",
                "RampUpRate",
                "RampDownRate",
                "StartUpCost",
                "RampingCost",
                "PartLoadMin",
                "StartUpTime",
                "CO2Intensity",
            ]:
                updated[item] = plants[
                    (plants["Fuel"] == fuel) & (plants["Entry"] == 0)
                ][item].mean()

            for item in ["Zone", "Technology"]:
                updated[item] = (
                    plants[(plants["Fuel"] == fuel) & (plants["Entry"] == 0)][item]
                    .mode()
                    .item()
                )

            plants = pd.concat(
                (plants, pd.Series(updated).to_frame().T), ignore_index=True
            )

    results = {}

    for year in years:
        plants = plants.loc[plants["Exit"] != year]

        if not "Nunits" in plants:
            plants["Nunits"] = 1

        plants["Cluster"] = plants[["Technology", "Fuel"]].agg("_".join, axis=1)
        clustered_plants = plants.groupby("Cluster")["Nunits"].sum().to_frame("Nunits")
        clustered_plants["Fuel"] = (
            plants.groupby("Cluster")["Fuel"].unique().map(lambda x: x[0])
        )
        clustered_plants["Technology"] = (
            plants.groupby("Cluster")["Technology"].unique().map(lambda x: x[0])
        )

        clustered_plants["Units"] = (
            plants.groupby("Cluster")["Unit"].unique().map(lambda x: ",".join(x))
        )

        clustered_plants["TotalCapacity"] = plants.groupby("Cluster")[
            "PowerCapacity"
        ].sum()
        clustered_plants["PowerCapacity"] = (
            clustered_plants["TotalCapacity"] / clustered_plants["Nunits"]
        )

        clustered_plants["PowerMinStable"] = plants.groupby("Cluster").apply(
            lambda x: (x["PartLoadMin"] * x["PowerCapacity"]).min()
        )

        clustered_plants["RampStartUpRate"] = clustered_plants[
            "PowerCapacity"
        ] / plants.groupby("Cluster").apply(
            lambda x: (x["PowerCapacity"] * x["StartUpTime"]).sum()
            / x["PowerCapacity"].sum()
        )

        for key in [
            "Efficiency",
            "MinUpTime",
            "MinDownTime",
            "RampUpRate",
            "RampDownRate",
            "CO2Intensity",
        ]:
            clustered_plants[key] = plants.groupby("Cluster").apply(
                lambda x: (x["PowerCapacity"] * x[key]).sum() / x["PowerCapacity"].sum()
            )

        for key in ["RampUpRate", "RampDownRate"]:
            clustered_plants[key] = (
                60 * clustered_plants[key] * clustered_plants["PowerCapacity"]
            )

        ramping_cost = plants.groupby("Cluster").apply(
            lambda x: (x["PowerCapacity"] * x["RampingCost"]).sum()
            / x["PowerCapacity"].sum()
        )

        clustered_plants["CostRampUp"] = ramping_cost + (
            plants.groupby("Cluster").apply(
                lambda x: x["StartUpCost"].sum() / x["PowerCapacity"].sum()
            )
        )

        if "ShutDownCost" in plants.columns:
            clustered_plants["CostRampDown"] = ramping_cost + (
                plants.groupby("Cluster").apply(
                    lambda x: x["ShutDownCost"].sum() / x["PowerCapacity"].sum()
                )
            )
        results[year] = clustered_plants

    return results


###########################################################################################
# Data aggregation into clusters
###########################################################################################


def aggregate_committed_capacity(
    country: str,
    clustered_plants: pd.DataFrame,
    period: Union[List[datetime.date], pd.DatetimeIndex] = None,
):
    """Aggregate committed capacity data into clusters.

    Args:
        country (str): The country to aggregate the data for (in ISO 3166-1 alpha-2 format).
        clustered_plants (pandas.DataFrame): The clustered plant data.
        period (list of datetime.date or pandas.DatetimeIndex with freq='D', optional): The
            period for the data to aggregate. Defaults to None.

    Returns:
        pandas.DataFrame: The aggregated committed capacity data.
    """
    generation = intake.cat["markets"].generation(country=country).read()
    if period is not None:
        generation = generation.loc[generation["ds"].isin(period)]

    committed_clustered = {}
    plant_mapping = {}

    for cluster, units in clustered_plants["Units"].to_dict().items():
        units = units.split(",")
        if (country == "EL") and ("THESAVROS" in units):
            units.remove("THESAVROS")
            units.extend(["THESAVROS1", "THESAVROS2", "THESAVROS3"])
        plant_mapping[cluster] = units

    for cluster, units in plant_mapping.items():
        values = (
            generation.loc[generation["unit_name"].isin(units)]
            .drop("unit_name", axis=1)
            .groupby("ds")
            .sum()
        )
        values = values.stack()
        values.index = values.index.map(
            lambda x: datetime.datetime.combine(x[0], datetime.time(int(x[1])))
        )
        committed_clustered[cluster] = values

    return pd.DataFrame.from_dict(committed_clustered)


def aggregate_available_capacity(
    country: str,
    clustered_plants: pd.DataFrame,
    period: Union[List[datetime.date], pd.DatetimeIndex] = None,
):
    """Aggregate available capacity data into clusters.

    Args:
        country (str): The country to aggregate the data for (in ISO 3166-1 alpha-2 format).
        clustered_plants (pandas.DataFrame): The clustered plant data.
        period (list of datetime.date or pandas.DatetimeIndex with freq='D', optional): The
            period for the data to aggregate. Defaults to None.

    Returns:
        pandas.DataFrame: The aggregated available capacity data.
    """

    capacity = intake.cat["markets"].capacity(country=country).read()
    if period is not None:
        capacity = capacity.loc[capacity["ds"].isin(period)]
    capacity = capacity.set_index("ds").sort_index()

    available_clustered = {}
    for _, row in clustered_plants.iterrows():
        cluster_name = row.name
        units = row["Units"].split(",")
        result = None
        for unit_name in units:
            data = capacity[unit_name]
            result = data if result is None else result.add(data, fill_value=0)
        available_clustered[f"{cluster_name}"] = result

    return pd.DataFrame.from_dict(available_clustered)

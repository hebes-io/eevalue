# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import pickle

import click
import intake
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import eevalue.logging as logging
from eevalue.build_data.historical import (
    build_exports,
    build_imports,
    build_water_value,
)
from eevalue.exceptions import NotEnoughData
from eevalue.preprocess.clusters import (
    aggregate_available_capacity,
    aggregate_committed_capacity,
    group_plants,
)
from eevalue.settings import DATA_DIR
from eevalue.utils import create_period, expand_dates, load_config, match_period

from .common import (
    CONFIG_HELP,
    COUNTRY_HELP,
    DATA_DIR_HELP,
    END_DATE_HELP,
    START_DATE_HELP,
)

CLUSTERS_HELP = """Create generation plant clusters and their aggregated data (committed
and available capacity)."""
PCA_HELP = """Carry out Principal Component Analysis on all hourly time series in the historical
data and save the estimated components for scenario generation."""


@click.group(name="eevalue")
def prep_cli():
    pass


@prep_cli.group()
def preprocess():
    """Commands for data preprocessing."""


@preprocess.command("clusters", help=CLUSTERS_HELP)
@click.option("--country", "-c", type=str, required=True, help=COUNTRY_HELP)
@click.option(
    "--start-date",
    "-sd",
    type=click.DateTime(formats=["%d-%m-%Y"]),
    required=True,
    help=START_DATE_HELP,
)
@click.option(
    "--end-date",
    "-ed",
    type=click.DateTime(formats=["%d-%m-%Y"]),
    required=True,
    help=END_DATE_HELP,
)
@click.option(
    "--data-dir",
    "-dd",
    type=click.Path(exists=True),
    default=None,
    envvar="EEVALUE_DATA_DIR",
    help=DATA_DIR_HELP,
)
def create_clusters(country, start_date, end_date, data_dir):
    period = create_period(start_date, end_date)
    data_dir = data_dir or DATA_DIR
    years = np.unique(period.year)

    logger = logging.getLogger("eevalue:preprocess:clusters")
    logger.setLevel(20)

    # aggregate plants
    clustered_plants = group_plants(country, years)
    for year, plants in clustered_plants.items():
        path = data_dir / "plants" / country / str(year)
        path.mkdir(parents=True, exist_ok=True)
        plants.to_csv(path / "clustered.csv")
        logger.info(f"Aggregated plant data saved in: {path}")

        # aggregate generation
        committed_clustered = aggregate_committed_capacity(
            country, plants, period=period[period.year == year]
        )

        if committed_clustered.empty:
            logger.error(
                "Aggregated committed capacity data is empty. No data was found "
                f"for the selected period {period.min()} - {period.max()}"
            )
        else:
            path = data_dir / "generation" / country / str(year)
            path.mkdir(parents=True, exist_ok=True)
            committed_clustered.to_csv(path / "clustered.csv", index_label="ds")
            logger.info(f"Aggregated committed capacity data saved in: {path}")

        # aggregate capacity
        available_clustered = aggregate_available_capacity(
            country, plants, period=period[period.year == year]
        )

        if available_clustered.empty:
            logger.error(
                "Aggregated available capacity data is empty. No data was found "
                f"for the selected period {period.min()} - {period.max()}"
            )
        else:
            path = data_dir / "capacity" / country / str(year)
            path.mkdir(parents=True, exist_ok=True)
            available_clustered.to_csv(path / "clustered.csv", index_label="ds")
            logger.info(f"Aggregated available capacity data saved in: {path}")


@preprocess.command("components", help=PCA_HELP)
@click.option("--country", "-c", type=str, required=True, help=COUNTRY_HELP)
@click.option(
    "--start-date",
    "-sd",
    type=click.DateTime(formats=["%d-%m-%Y"]),
    required=True,
    help=START_DATE_HELP,
)
@click.option(
    "--end-date",
    "-ed",
    type=click.DateTime(formats=["%d-%m-%Y"]),
    required=True,
    help=END_DATE_HELP,
)
@click.option(
    "--config",
    "-cfg",
    type=click.Path(exists=True),
    default=None,
    help=CONFIG_HELP,
)
@click.option(
    "--data-dir",
    "-dd",
    type=click.Path(exists=True),
    default=None,
    envvar="EEVALUE_DATA_DIR",
    help=DATA_DIR_HELP,
)
def find_components(country, start_date, end_date, config, data_dir):
    period = create_period(start_date, end_date)
    config = load_config(config=config, country=country)
    data_dir = data_dir or DATA_DIR

    logger = logging.getLogger("eevalue:preprocess:components")
    logger.setLevel(20)

    features = []
    feature_mapping = {"names": []}

    try:
        load = (
            intake.cat["markets"]
            .load(country=country)
            .read()
            .set_index("ds")
            .pipe(expand_dates)
            .pipe(match_period, period=period)
        )
        load_nt = load.div(load.groupby(lambda x: x.year).transform("mean"))
        load_nt["hour"] = load_nt.index.hour
        load_nt["date"] = load_nt.index.date
        features.append(
            load_nt.pivot(index="date", columns="hour", values="Total load [MW]")
        )
        feature_mapping["names"].append("demand")
        feature_mapping["demand"] = "demand"

        avail_factors = (
            intake.cat["markets"]
            .availability_factors(country=country)
            .read()
            .set_index("ds")
            .pipe(expand_dates)
        )[["PHOT", "WTON"]]

        try:
            avail_factors = avail_factors.pipe(match_period, period=period)
            avail_factors["hour"] = avail_factors.index.hour
            avail_factors["date"] = avail_factors.index.date
        except NotEnoughData:
            load["hour"] = load.index.hour
            load["day"] = load.index.dayofyear
            avail_factors["hour"] = avail_factors.index.hour
            avail_factors["day"] = avail_factors.index.dayofyear
            avail_factors = (
                pd.merge(
                    load.reset_index(names="index"),
                    avail_factors,
                    how="left",
                    left_on=["day", "hour"],
                    right_on=["day", "hour"],
                )
                .drop(["Total load [MW]", "day"], axis=1)
                .set_index("index")
            )
            avail_factors["date"] = avail_factors.index.date
        finally:
            for col in avail_factors.columns:
                if col in ("hour", "date"):
                    continue
                features.append(
                    avail_factors.pivot(index="date", columns="hour", values=col)
                )
                feature_mapping["names"].append(col)
                if col == "PHOT":
                    feature_mapping["avail_factor_pv"] = col
                if col == "WTON":
                    feature_mapping["avail_factor_wind"] = col

        exports = build_exports(period, country).to_frame("values")
        exports["hour"] = exports.index.hour
        exports["date"] = exports.index.date
        features.append(exports.pivot(index="date", columns="hour", values="values"))
        feature_mapping["names"].append("max_exports")
        feature_mapping["max_exports"] = "max_exports"

        imports = build_imports(period, country).to_frame("values").mul(-1)
        imports["hour"] = imports.index.hour
        imports["date"] = imports.index.date
        features.append(imports.pivot(index="date", columns="hour", values="values"))
        feature_mapping["names"].append("max_imports")
        feature_mapping["max_imports"] = "max_imports"

        if config["Market"].get("HasHydro", False):
            water_value = build_water_value(period, country).to_frame("values")
            water_value["hour"] = water_value.index.hour
            water_value["date"] = water_value.index.date
            features.append(
                water_value.pivot(index="date", columns="hour", values="values")
            )
            feature_mapping["names"].append("water_value")
            feature_mapping["water_value"] = "water_value"

        features = pd.concat(features, axis=1)

        if ("ExVar" in config["Preprocessing"]) and (
            config["Preprocessing"]["ExVar"] is not None
        ):
            ex_var = config["Preprocessing"]["ExVar"]
            pca = make_pipeline(StandardScaler(), PCA()).fit(features.values)
            n_components = (
                np.argmax(np.cumsum(pca["pca"].explained_variance_ratio_) > ex_var) + 1
            )
        else:
            n_components = config["Preprocessing"].get("NumComponents", 10)

    except NotEnoughData as ex:
        logger.error(str(ex))
        raise click.Abort()

    pca = make_pipeline(StandardScaler(), PCA(n_components=n_components)).fit(
        features.values
    )
    U = pd.DataFrame(data=pca.transform(features.values), index=features.index)

    path = data_dir / "components" / country
    path.mkdir(parents=True, exist_ok=True)

    U.to_csv(path / "components.csv", index_label="ds")
    with open(path / "pca.pickle", "wb") as f:
        pickle.dump(pca, f)
    with open(path / "features.pickle", "wb") as f:
        pickle.dump(feature_mapping, f)

    logger.info(f"Principal components saved in: {path}")

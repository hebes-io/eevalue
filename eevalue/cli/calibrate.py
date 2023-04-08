# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import datetime
import functools
from typing import Literal

import click
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from mlflow.tracking.context.registry import resolve_tags
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer, OrdinalEncoder
from tqdm import tqdm

import eevalue.logging as logging
from eevalue.build_data.historical import create_historical_data
from eevalue.calibrate.inputs import (
    create_hydro_state,
    create_hydro_trials,
    create_markup_state,
    create_markup_trials,
)
from eevalue.settings import DATA_DIR
from eevalue.simulate.inputs import SimData
from eevalue.simulate.model import create_model, run_solver
from eevalue.simulate.postprocess import pyomo_to_pandas
from eevalue.utils import create_period, load_config

from .common import (
    CONFIG_HELP,
    COUNTRY_HELP,
    DATA_DIR_HELP,
    END_DATE_HELP,
    HYDRO_FLAG_HELP,
    LOCAL_FILE_URI_PREFIX,
    MARKUP_FLAG_HELP,
    START_DATE_HELP,
    TAG_ARG_HELP,
    TR_URI_HELP,
)

CALIB_HELP = """Calibrate the model by learning the effective availability of hydropower
(if the flag `--no-hydro` is not passed) and a cost markup function (if the flag `--no-markup`
is not passed)."""
EXP_NAME_HELP = """The name of the experiment for experiment tracking."""


########################################################################################
# Utility functions
########################################################################################


def sample_dataset(dataset: dict, day: datetime.date, warmup: int = 1, length: int = 1):
    """Create a sample from the provided dataset.

    Args:
        dataset (dict): The input dataset.
        day (datetime.date): The date to sample.
        warmup (int, optional): The number of previous days to include in the
            sample. Defaults to 1.
        length (int, optional): The number of days (including `day`) to add in the
            sample. Defaults to 1.

    Returns:
        dict: The sample from the provided dataset.
    """
    sample = copy.deepcopy(dataset)

    period = pd.date_range(
        start=day - datetime.timedelta(warmup),
        end=day + datetime.timedelta(days=length),
        freq="H",
        inclusive="left",
    )
    sample["simulation_period"] = period

    for name in [
        "avail_factors",
        "demand",
        "available_capacity",
        "committed_capacity",
        "net_imports",
        "res_generation",
        "fuel_price",
        "permit_price",
        "water_value",
    ]:
        if name in dataset:
            sample[name] = (
                dataset[name].loc[np.isin(dataset[name].index, period)].copy()
            )

    return sample


def run_sample(
    *,
    dataset: dict,
    day: datetime.date,
    parameter: Literal["markup", "hydro"],
    config: dict,
):
    """Sample the provided dataset, create random trials for markup or effective hydro
    availability, and run simulation.

    Args:
        dataset (dict): The input dataset.
        day (datetime.date): The date to sample.
        parameter ({"markup", "hydro"}): The parameter for which to create random trials.
        config (dict): The configuration dictionary.
    """
    sample_data = sample_dataset(
        dataset,
        day,
        warmup=config["Simulation"].get("Warmup", 2),
        length=config["Calibration"].get("SimLength", 5),
    )

    if parameter == "markup":
        markup = create_markup_trials(
            period=sample_data["simulation_period"],
            clusters=sample_data["available_capacity"].columns,
            active_clusters=config["Calibration"].get("HasMarkup"),
            min_value=config["Calibration"].get("MarkupMin"),
            max_value=config["Calibration"].get("MarkupMax"),
        )
        sample_data["markup"] = markup
    else:
        hydro_avail_factor = create_hydro_trials(
            period=sample_data["simulation_period"]
        )
        sample_data["avail_factors"][
            sample_data["avail_factors"].filter(regex="_WAT$").columns[0]
        ] = hydro_avail_factor

    _ = sample_data.pop("water_value", None)
    available_capacity = sample_data.pop("available_capacity")
    committed_capacity = sample_data.pop("committed_capacity")

    sim_data = SimData(**sample_data, action="calibrate")
    sets, parameters = sim_data.build()

    model = create_model(sets, parameters, action="calibrate")
    model = run_solver(model, solver=config["Simulation"]["Solver"])

    predicted_generation = pyomo_to_pandas(
        model, "Power", period=sample_data["simulation_period"]
    ).astype(np.float32)

    error = committed_capacity.sub(predicted_generation).div(available_capacity)
    error = error.loc[error.index.date >= day].abs().sum(axis=1)
    trial = (
        markup.loc[error.index, :]
        if parameter == "markup"
        else hydro_avail_factor[error.index].to_frame("Trial")
    )
    return trial, error


def learn_parameter(parameter, dataset, config):
    states = (
        create_markup_state(
            load=dataset["demand"]["DA"],
            res=dataset["res_generation"],
            net_imports=dataset["net_imports"],
            capacity=dataset["available_capacity"],
            water_value=dataset.get("water_value"),
        )
        if parameter == "markup"
        else create_hydro_state(
            availability_factor=dataset["avail_factors"].filter(regex="_WAT$"),
            water_value=dataset["water_value"],
        )
    )

    def _to_single_class(data):
        return np.apply_along_axis(
            lambda x: functools.reduce(np.core.defchararray.add, x),
            1,
            data.astype(int).astype(str),
        ).reshape(-1, 1)

    clf = Pipeline(
        [
            (
                "discretize",
                KBinsDiscretizer(
                    n_bins=config["Calibration"].get("NumBins", 3),
                    encode="ordinal",
                    strategy="uniform",
                ),
            ),
            ("aggregate", FunctionTransformer(_to_single_class)),
            (
                "encode_ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    dtype=np.int16,
                ),
            ),
            (
                "impute_unknown",
                SimpleImputer(
                    missing_values=-1,
                    strategy="most_frequent",
                ),
            ),
        ]
    ).fit(states)

    states_to_clusters = pd.Series(
        data=clf.transform(states).squeeze(), index=states.index
    )
    safe_subset = np.sort(np.unique(states.index.date))[
        config["Simulation"]
        .get("Warmup", 2) : -config["Calibration"]
        .get("SimLength", 5)
    ]

    sample, _ = train_test_split(
        states.loc[np.isin(states.index.date, safe_subset)].sort_index(),
        train_size=config["Calibration"].get("NumSamples", 500),
        stratify=states_to_clusters.loc[
            np.isin(states_to_clusters.index.date, safe_subset)
        ].sort_index(),
    )

    sample = np.unique(sample.index.date)

    X_all = []
    y_all = []

    pbar = tqdm(total=len(sample))
    for day in sample:
        trial, error = run_sample(
            dataset=dataset, parameter=parameter, day=day, config=config
        )
        X_all.append(pd.concat((states.loc[trial.index, :], trial), axis=1))
        y_all.append(error)
        pbar.update(1)
    pbar.close()

    X_all = pd.concat(X_all, axis=0)
    y_all = pd.concat(y_all, axis=0)
    model = GradientBoostingRegressor(
        n_estimators=500,
        validation_fraction=0.2,
        n_iter_no_change=10,
    ).fit(X_all, y_all)

    n_trials = config["Calibration"].get("NumTrials", 50)
    click.echo("Assembling training data.")
    placeholder = pd.DataFrame(0, index=range(n_trials), columns=X_all.columns)

    if parameter == "markup":
        clusters = dataset["available_capacity"].columns
        target = pd.DataFrame(0, index=X_all.index, columns=clusters)
        pbar = tqdm(total=X_all.shape[0])

        for i, (_, row) in enumerate(X_all.iterrows()):
            placeholder.loc[:] = row[placeholder.columns].values
            trials = create_markup_trials(
                period=range(n_trials),
                clusters=clusters,
                active_clusters=config["Calibration"].get("HasMarkup"),
                min_value=config["Calibration"].get("MarkupMin"),
                max_value=config["Calibration"].get("MarkupMax"),
            )
            placeholder.loc[:, clusters] = trials.loc[:, clusters]
            prediction = model.predict(placeholder)
            target.iloc[i][clusters] = np.array(
                trials.iloc[np.argmin(prediction)][clusters]
            ).astype(float)
            pbar.update(1)
        pbar.close()

        model = MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=500,
                max_depth=2,
                validation_fraction=0.2,
                n_iter_no_change=10,
            )
        ).fit(states.loc[target.index], target)

    else:
        target = pd.Series(0, index=X_all.index)
        pbar = tqdm(total=X_all.shape[0])

        for i, (_, row) in enumerate(X_all.iterrows()):
            placeholder.loc[:] = row[placeholder.columns].values
            trials = create_hydro_trials(period=range(n_trials))
            placeholder.loc[:, "Trial"] = trials
            prediction = model.predict(placeholder)
            target.iloc[i] = np.array(trials.iloc[np.argmin(prediction)]).astype(float)
            pbar.update(1)
        pbar.close()

        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=2, validation_fraction=0.2, n_iter_no_change=10
        ).fit(states.loc[target.index], target)

    return model


################################################################################################
# CLI commands
################################################################################################


@click.group(name="eevalue")
def calib_cli():
    pass


@calib_cli.command("calibrate", help=CALIB_HELP)
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
@click.option(
    "--experiment-name",
    "-en",
    type=str,
    default="Default",
    help=EXP_NAME_HELP,
)
@click.option(
    "--tracking-uri",
    "-tu",
    type=str,
    default=None,
    envvar="MLFLOW_TRACKING_URI",
    help=TR_URI_HELP,
)
@click.option("--tag", "-t", "tags", type=str, multiple=True, help=TAG_ARG_HELP)
@click.option("--hydro/--no-hydro", default=True, help=HYDRO_FLAG_HELP)
@click.option("--markup/--no-markup", default=True, help=MARKUP_FLAG_HELP)
def calibrate(
    country,
    start_date,
    end_date,
    config,
    data_dir,
    experiment_name,
    tracking_uri,
    tags,
    hydro,
    markup,
):
    parameters = []
    if hydro:
        parameters.append("hydro")
    if markup:
        parameters.append("markup")

    if not parameters:
        raise ValueError(
            "The flags `--no-hydro` and `--no-markup` cannot be used at the same time"
        )

    period = create_period(start_date, end_date)
    config = load_config(config=config, country=country)
    data_dir = data_dir or DATA_DIR

    logger = logging.getLogger("eevalue:calibrate")
    logger.setLevel(20)

    tracking_uri = tracking_uri or f"{LOCAL_FILE_URI_PREFIX}./eevalue/tracking/mlruns"
    mlflow.set_tracking_uri(tracking_uri)

    tags = (
        {}
        if not tags
        else {key: val for key, val in [item.split("=", 1) for item in tags]}
    )
    tags.update(
        {"country": country, "action": "calibrate", "hydro": hydro, "markup": markup}
    )

    mlflow_client = MlflowClient()
    expt = mlflow_client.get_experiment_by_name(experiment_name)
    if expt is not None:
        experiment_id = expt.experiment_id
    else:
        logger.info(f"Experiment with name `{experiment_name}` not found. Creating it.")
        experiment_id = mlflow_client.create_experiment(
            name=experiment_name, artifact_location=tracking_uri
        )

    dataset = create_historical_data(
        period=period, country=country, config=config, data_dir=data_dir
    )

    with mlflow.start_run(experiment_id=experiment_id, tags=resolve_tags(tags)) as run:
        mlflow.log_params({**config["Calibration"], **config["Simulation"]})

        hydro_model = None
        if "hydro" in parameters:
            logger.info("Start learning parameter: `hydro`")
            hydro_model = learn_parameter("hydro", dataset, config)
            mlflow.sklearn.log_model(hydro_model, "calibration/models/hydro/predictor")

        if "markup" in parameters:
            logger.info("Start learning parameter: `markup`")
            if hydro_model is not None:
                states = create_hydro_state(
                    availability_factor=dataset["avail_factors"].filter(regex="_WAT$"),
                    water_value=dataset["water_value"],
                )
                hydro_avail_factor = pd.Series(
                    hydro_model.predict(states),
                    index=states.index,
                )
                hydro_cluster = (
                    dataset["avail_factors"].filter(regex="_WAT$").columns[0]
                )
                dataset["available_capacity"][hydro_cluster] = (
                    dataset["available_capacity"][hydro_cluster]
                    .div(dataset["avail_factors"][hydro_cluster])
                    .mul(hydro_avail_factor)
                )
                dataset["avail_factors"][hydro_cluster] = hydro_avail_factor

            markup_model = learn_parameter("markup", dataset, config)
            mlflow.sklearn.log_model(
                markup_model, "calibration/models/markup/predictor"
            )

        logger.info(f"run ID: {run.info.run_id}")

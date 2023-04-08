# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import os

import click
import mlflow
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

import eevalue.logging as logging
from eevalue.build_data.historical import create_historical_data
from eevalue.calibrate.inputs import create_hydro_state, create_markup_state
from eevalue.exceptions import NotEnoughData
from eevalue.settings import DATA_DIR
from eevalue.simulate.inputs import SimData
from eevalue.simulate.model import create_model, run_solver
from eevalue.simulate.postprocess import pyomo_to_pandas
from eevalue.utils import create_period, cvrmse, load_config

from .common import (
    CONFIG_HELP,
    COUNTRY_HELP,
    DATA_DIR_HELP,
    END_DATE_HELP,
    LOCAL_FILE_URI_PREFIX,
    START_DATE_HELP,
    TAG_ARG_HELP,
    TR_URI_HELP,
    get_artifact_path,
)

BACKTEST_HELP = """Run simulation on historical data and compare results."""
EXP_NAME_HELP = """The name of the experiment for experiment tracking. Ignored if `run-id` is
provided, since in this case, the command will update the provided run instead of creating a new one."""
RUN_ID_HELP = """The ID string for the run that contains the effective hydro availability factor
and the markup predictive models. If not provided, nominal availability will be used and no markup
will be added to the simulation. The predictive models are created through the `calibrate` command."""


@click.group(name="eevalue")
def bt_cli():
    pass


@bt_cli.command("backtest", help=BACKTEST_HELP)
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
    "--run-id",
    "-ri",
    type=str,
    default=None,
    help=RUN_ID_HELP,
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
def backtest(
    country,
    start_date,
    end_date,
    config,
    data_dir,
    run_id,
    experiment_name,
    tracking_uri,
    tags,
):
    period = create_period(start_date, end_date)
    config = load_config(config=config, country=country)
    data_dir = data_dir or DATA_DIR

    logger = logging.getLogger("eevalue:backtest")
    logger.setLevel(20)

    tracking_uri = tracking_uri or f"{LOCAL_FILE_URI_PREFIX}./eevalue/tracking/mlruns"
    mlflow.set_tracking_uri(tracking_uri)

    tags = (
        {}
        if not tags
        else {key: val for key, val in [item.split("=", 1) for item in tags]}
    )
    tags.update({"country": country, "action": "backtest"})

    try:
        dataset = create_historical_data(
            period=period, country=country, config=config, data_dir=data_dir
        )
    except NotEnoughData as ex:
        logger.error(str(ex))
        raise click.Abort()

    if run_id is not None:
        try:
            model = mlflow.sklearn.load_model(
                get_artifact_path(
                    run_id, tracking_uri, "calibration", "models", "hydro", "predictor"
                )
            )
        except MlflowException:
            pass
        else:
            states = create_hydro_state(
                availability_factor=dataset["avail_factors"].filter(regex="_WAT$"),
                water_value=dataset["water_value"],
            )
            hydro_avail_factor = pd.Series(
                model.predict(states),
                index=states.index,
            )
            hydro_cluster = dataset["avail_factors"].filter(regex="_WAT$").columns[0]
            dataset["available_capacity"][hydro_cluster] = (
                dataset["available_capacity"][hydro_cluster]
                .div(dataset["avail_factors"][hydro_cluster])
                .mul(hydro_avail_factor)
            )
            dataset["avail_factors"][hydro_cluster] = hydro_avail_factor

        try:
            model = mlflow.sklearn.load_model(
                get_artifact_path(
                    run_id, tracking_uri, "calibration", "models", "markup", "predictor"
                )
            )
        except MlflowException:
            pass
        else:
            states = create_markup_state(
                load=dataset["demand"]["DA"],
                res=dataset["res_generation"],
                net_imports=dataset["net_imports"],
                capacity=dataset["available_capacity"],
                water_value=dataset.get("water_value"),
            )
            markup = pd.DataFrame(
                model.predict(states),
                index=states.index,
                columns=dataset["available_capacity"].columns,
            )
            dataset["markup"] = markup

        experiment_id = None
    else:
        mlflow_client = MlflowClient()
        expt = mlflow_client.get_experiment_by_name(experiment_name)
        if expt is not None:
            experiment_id = expt.experiment_id
        else:
            logger.info(
                f"Experiment with name `{experiment_name}` not found. Creating it..."
            )
            experiment_id = mlflow_client.create_experiment(
                name=experiment_name, artifact_location=tracking_uri
            )

    _ = dataset.pop("available_capacity")
    _ = dataset.pop("water_value", None)
    committed_capacity = dataset.pop("committed_capacity")

    with mlflow.start_run(experiment_id=experiment_id, run_id=run_id) as run:
        mlflow.set_tags(tags)
        mlflow.log_params({**config["Simulation"]})

        def log_csv(filename, data):
            with open(filename, "w") as f:
                f.write(data.to_csv(index_label="ds", lineterminator="\n"))
                mlflow.log_artifact(filename, "backtest/results")
            os.remove(filename)

        predicted_generation = None
        # split simulation per 3 months (3x30x24 = 2160 hours) to speed up
        for subset in np.array_split(
            dataset["simulation_period"],
            math.ceil(len(dataset["simulation_period"]) / 2160),
        ):
            dataset["simulation_period"] = subset
            sim_data = SimData(**dataset, action="backtest")
            sets, parameters = sim_data.build()
            model = create_model(sets, parameters, action="backtest")
            model = run_solver(model, solver=config["Simulation"].get("Solver"))
            predicted_generation = pd.concat(
                (predicted_generation, pyomo_to_pandas(model, "Power", period=subset)),
                axis=0,
            )

        predicted_generation = predicted_generation.sort_index()
        log_csv("actual.csv", committed_capacity)
        log_csv("predicted.csv", predicted_generation)

        for col in committed_capacity.columns:
            error = cvrmse(committed_capacity[col], predicted_generation[col])
            logger.info(f"CV of RMSE for {col} generation: {error}")
            mlflow.log_metric(f"{col}-CVRMSE", error)

        logger.info(f"run ID: {run.info.run_id}")

# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
import pickle

import blosc
import click
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from tqdm.auto import tqdm

import eevalue.logging as logging
from eevalue.build_data.forward import sample_yearly
from eevalue.simulate.inputs import SimData
from eevalue.simulate.model import create_model, run_solver
from eevalue.simulate.postprocess import pyomo_to_pandas
from eevalue.utils import as_series, create_period, load_config

from .common import (
    CONFIG_HELP,
    END_DATE_HELP,
    LOCAL_FILE_URI_PREFIX,
    START_DATE_HELP,
    TAG_ARG_HELP,
    TR_URI_HELP,
    get_artifact_path,
)

EXP_NAME_HELP = """The name of the experiment for experiment tracking."""
LOAD_MODE_HELP = """Parameter indicating the presence and type of the load modifying
resources. It can be 'up' or 'down'."""
LS_HELP = (
    """Flag to indicate whether the model should include load shaping resources."""
)
REPLAY_HELP = """Re-run scenarios adding load shaping and/or storage resources."""
RUN_ID_HELP = """The ID string for the run that contains the scenarios to re-run."""
STORAGE_HELP = """Flag to indicate whether the model should include electricity storage
(for demand flexibility)."""


################################################################################################
# CLI commands
################################################################################################


@click.group(name="eevalue")
def replay_cli():
    pass


@replay_cli.command("replay", help=REPLAY_HELP)
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
    "--run-id",
    "-ri",
    type=str,
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
@click.option("--storage/--no-storage", default=False, help=STORAGE_HELP)
@click.option("--load-shaping/--no-load-shaping", default=False, help=LS_HELP)
@click.option(
    "--load-shape-mode",
    "-lsm",
    type=str,
    default=None,
    help=LOAD_MODE_HELP,
)
@click.option("--tag", "-t", "tags", type=str, multiple=True, help=TAG_ARG_HELP)
def replay(
    start_date,
    end_date,
    config,
    run_id,
    experiment_name,
    tracking_uri,
    storage,
    load_shaping,
    load_shape_mode,
    tags,
):
    if (not storage) and (not load_shaping):
        raise ValueError("`storage` or `load_shaping` or both should be enabled")

    if load_shaping and (load_shape_mode is None):
        raise ValueError(
            "If `load_shaping` is enabled, a `load_shape_mode` must be provided."
        )

    if load_shaping and (load_shape_mode not in ("up", "down")):
        raise ValueError(
            f"`load_shape_mode` can be either 'up' or 'down'. Got {load_shape_mode}"
        )

    period = create_period(start_date, end_date)

    logger = logging.getLogger("eevalue:replay")
    logger.setLevel(20)

    tracking_uri = tracking_uri or f"{LOCAL_FILE_URI_PREFIX}./eevalue/tracking/mlruns"
    mlflow.set_tracking_uri(tracking_uri)

    country = mlflow.get_run(run_id).data.tags["country"]
    config = load_config(config=config, country=country)

    tags = (
        {}
        if not tags
        else {key: val for key, val in [item.split("=", 1) for item in tags]}
    )
    tags.update(
        {
            "country": country,
            "action": "replay",
            "reference_run_id": run_id,
            "storage": storage,
            "load_shaping": load_shaping,
            "load_shape_mode": load_shape_mode,
        }
    )

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

    logger.info(
        "Scenario replay has started with storage "
        + ("ENABLED" if storage else "NOT ENABLED")
        + " and load shaping "
        + ("ENABLED" if load_shaping else "NOT ENABLED")
    )

    with mlflow.start_run(experiment_id=experiment_id) as run:
        mlflow.set_tags(tags)
        mlflow.log_params({**config["Simulation"]})

        def log_csv(filename, data):
            data.to_csv(
                filename,
                index_label="ds",
                compression={"method": "gzip", "compresslevel": 1, "mtime": 1},
            )
            mlflow.log_artifact(filename, "forward/results")
            os.remove(filename)

        warmup = config["Scenarios"]["Warmup"]

        if storage:
            scen_storage_capacity = sample_yearly(
                period=period,
                filepath=config["Scenarios"]["Storage"]["Capacity"]["FilePath"],
                multiplier=config["Scenarios"]["Storage"]["Capacity"]["Multiplier"],
            )

        inputs_path = get_artifact_path(run_id, tracking_uri, "forward", "inputs")
        results_path = get_artifact_path(run_id, tracking_uri, "forward", "results")

        filepaths = glob.glob(inputs_path + "/scen_inputs_*")
        with tqdm(total=len(filepaths)) as pbar:
            for filename in filepaths:
                basename = os.path.basename(filename)
                i, j = basename.split(".")[0].rsplit("_", 2)[-2:]

                with open(filename, "rb") as f:
                    dataset = pickle.loads(blosc.decompress(f.read()))

                if storage:
                    dataset["storage_charge_ef"] = config["Scenarios"]["Storage"][
                        "StorageChargingEfficiency"
                    ]
                    dataset["storage_discharge_ef"] = config["Scenarios"]["Storage"][
                        "StorageDischargeEfficiency"
                    ]
                    dataset["storage_capacity"] = scen_storage_capacity()[0]
                    dataset["storage_charge_cap"] = (
                        dataset["storage_capacity"]
                        / config["Scenarios"]["Storage"]["StorageChargingCapacity"]
                    )
                    dataset["storage_min"] = (
                        dataset["storage_capacity"]
                        * config["Scenarios"]["Storage"]["StorageMinimum"]
                    )
                    dataset["storage_final_min"] = (
                        dataset["storage_capacity"]
                        * config["Scenarios"]["Storage"]["StorageFinalMin"]
                    )
                    dataset["storage_initial"] = dataset["storage_min"]

                if load_shaping:
                    filenames = (
                        [f"curtailed_power_{i}_{j}.csv", f"ll_ramp_down_{i}_{j}.csv"]
                        if load_shape_mode == "up"
                        else [f"ll_max_power_{i}_{j}.csv", f"ll_ramp_up_{i}_{j}.csv"]
                    )

                    max_load_shape = None
                    for name in filenames:
                        data = as_series(
                            pd.read_csv(
                                os.path.join(results_path, name), compression="gzip"
                            ).set_index("ds")
                        )
                        max_load_shape = (
                            data
                            if (max_load_shape is None)
                            else max_load_shape.mask(data > max_load_shape, data)
                        )

                    dataset["maximum_load_shaping"] = (
                        np.random.uniform(low=0.1, high=0.9) * max_load_shape.sum()
                    )

                # Run simulation
                sim_data = SimData(
                    **dataset, action="forward", load_shape_mode=load_shape_mode
                )
                sets, parameters = sim_data.build()
                model = create_model(
                    sets, parameters, action="forward", load_shape_mode=load_shape_mode
                )
                model = run_solver(model, solver=config["Simulation"]["Solver"])

                ll_max_power = pyomo_to_pandas(
                    model, "LLMaxPower", period=dataset["simulation_period"]
                )["LLMaxPower"].iloc[24 * warmup :]
                ll_ramp_down = (
                    pyomo_to_pandas(
                        model, "LLRampDown", period=dataset["simulation_period"]
                    )
                    .sum(axis=1)
                    .iloc[24 * warmup :]
                )
                ll_ramp_up = (
                    pyomo_to_pandas(
                        model, "LLRampUp", period=dataset["simulation_period"]
                    )
                    .sum(axis=1)
                    .iloc[24 * warmup :]
                )
                curtailed_power = pyomo_to_pandas(
                    model, "CurtailedPower", period=dataset["simulation_period"]
                )["CurtailedPower"].iloc[24 * warmup :]
                power = pyomo_to_pandas(
                    model, "Power", period=dataset["simulation_period"]
                ).iloc[24 * warmup :, :]

                if storage:
                    storage_input = pyomo_to_pandas(
                        model, "StorageInput", period=dataset["simulation_period"]
                    )["StorageInput"].iloc[24 * warmup :]
                    storage_output = pyomo_to_pandas(
                        model, "StorageOutput", period=dataset["simulation_period"]
                    )["StorageOutput"].iloc[24 * warmup :]

                if load_shaping:
                    load_mode_up = pyomo_to_pandas(
                        model, "LoadModeUp", period=dataset["simulation_period"]
                    )["LoadModeUp"].iloc[24 * warmup :]
                    load_mode_down = pyomo_to_pandas(
                        model, "LoadModeDown", period=dataset["simulation_period"]
                    )["LoadModeDown"].iloc[24 * warmup :]

                log_csv(f"ll_max_power_{i}_{j}.csv", ll_max_power)
                log_csv(f"ll_ramp_down_{i}_{j}.csv", ll_ramp_down)
                log_csv(f"ll_ramp_up_{i}_{j}.csv", ll_ramp_up)
                log_csv(f"curtailed_power_{i}_{j}.csv", curtailed_power)
                log_csv(f"power_{i}_{j}.csv", power)
                if storage:
                    log_csv(f"storage_input_{i}_{j}.csv", storage_input)
                    log_csv(f"storage_output_{i}_{j}.csv", storage_output)
                if load_shaping:
                    log_csv(f"load_mode_up_{i}_{j}.csv", load_mode_up)
                    log_csv(f"load_mode_down_{i}_{j}.csv", load_mode_down)

                pbar.update(1)

        logger.info(f"run ID: {run.info.run_id}")

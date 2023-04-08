# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import datetime
import os
import pickle

import blosc
import click
import mlflow
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_is_fitted
from tqdm.auto import trange

import eevalue.logging as logging
from eevalue.build_data.forward import (
    sample_components,
    sample_historical,
    sample_yearly,
)
from eevalue.build_data.historical import build_availability
from eevalue.calibrate.inputs import create_hydro_state, create_markup_state
from eevalue.preprocess.clusters import group_plants
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
    LOCAL_FILE_URI_PREFIX,
    START_DATE_HELP,
    TAG_ARG_HELP,
    TR_URI_HELP,
    get_artifact_path,
)

EXP_NAME_HELP = """The name of the experiment for experiment tracking."""
RUN_ID_HELP = """The ID string for the run that contains the effective hydro availability factor
and the markup predictive models. If not provided, nominal availability will be used and no markup
will be added to the simulation. The predictive models are created through the `calibrate` command."""
SIMUL_HELP = """Simulate the model using forward scenario generation."""
STORAGE_HELP = """Flag to indicate whether the model should include electricity storage (for demand
flexibility)."""


################################################################################################
# Utility functions
################################################################################################


def sample_dataset(dataset: dict, to_ignore: pd.Series, warmup: int = 1):
    """Create a sample from the provided dataset.

    Args:
        dataset (dict): The input dataset.
        to_ignore (pandas.Series): Time series indicating whether a day should be ignored.
        warmup (int, optional): The number of previous days to include in the sample. Defaults to 1.

    Returns:
        dict: The sample from the provided dataset.
    """

    period = dataset["simulation_period"]
    period = period[~np.isin(period.date, to_ignore[to_ignore].index.date)]

    if period.size > 0:
        if warmup > 0:
            period = pd.date_range(
                start=period.min().date() - datetime.timedelta(days=warmup),
                end=period.min().date(),
                freq="H",
                inclusive="left",
            ).union(period)

        sample = copy.deepcopy(dataset)
        sample["simulation_period"] = period

        for name in [
            "avail_factors",
            "avail_factor_pv",
            "avail_factor_wind",
            "capacity_pv",
            "capacity_wind",
            "demand",
            "fuel_price",
            "markup",
            "max_exports",
            "max_imports",
            "permit_price",
        ]:
            if name in dataset:
                sample[name] = dataset[name].reindex(period, copy=True)
                if sample[name].isnull().values.any():
                    sample[name] = sample[name].groupby(lambda x: x.time).bfill()
        return sample
    else:
        return {}


################################################################################################
# CLI commands
################################################################################################


@click.group(name="eevalue")
def sim_cli():
    pass


@sim_cli.command("simulate", help=SIMUL_HELP)
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
@click.option("--storage/--no-storage", default=False, help=STORAGE_HELP)
@click.option("--tag", "-t", "tags", type=str, multiple=True, help=TAG_ARG_HELP)
def simulate(
    country,
    start_date,
    end_date,
    config,
    data_dir,
    run_id,
    experiment_name,
    tracking_uri,
    storage,
    tags,
):
    period = create_period(start_date, end_date)
    config = load_config(config=config, country=country)
    data_dir = data_dir or DATA_DIR

    logger = logging.getLogger("eevalue:simulate")
    logger.setLevel(20)

    tracking_uri = tracking_uri or f"{LOCAL_FILE_URI_PREFIX}./eevalue/tracking/mlruns"
    mlflow.set_tracking_uri(tracking_uri)

    tags = (
        {}
        if not tags
        else {key: val for key, val in [item.split("=", 1) for item in tags]}
    )
    tags.update(
        {"country": country, "action": "simulation", "calibration_run_id": run_id}
    )

    n_inter_trials = config["Scenarios"]["NumInterTrials"]
    n_intra_trials = config["Scenarios"]["NumIntraTrials"]
    warmup = config["Scenarios"]["Warmup"]

    hydro_model = None
    markup_model = None

    if run_id is not None:
        try:
            hydro_model = mlflow.sklearn.load_model(
                get_artifact_path(
                    run_id, tracking_uri, "calibration", "models", "hydro", "predictor"
                )
            )
        except MlflowException:
            pass

        try:
            markup_model = mlflow.sklearn.load_model(
                get_artifact_path(
                    run_id, tracking_uri, "calibration", "models", "markup", "predictor"
                )
            )
        except MlflowException:
            pass

    path = data_dir / "components" / country
    with open(path / "pca.pickle", "rb") as f:
        pca = pickle.load(f)
    with open(path / "features.pickle", "rb") as f:
        feature_mapping = pickle.load(f)

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

    with mlflow.start_run(experiment_id=experiment_id) as run:
        mlflow.set_tags(tags)
        mlflow.log_params({**config["Simulation"]})

        def reshape(data, column=None):
            data = data.stack()
            data.index = data.index.map(
                lambda x: datetime.datetime.combine(x[0], datetime.time(x[1]))
            )
            if column is not None:
                data = data.to_frame(column)
            return data

        def log_csv(filename, data):
            data.to_csv(
                filename,
                index_label="ds",
                compression={"method": "gzip", "compresslevel": 1, "mtime": 1},
            )
            mlflow.log_artifact(filename, "forward/results")
            os.remove(filename)

        def log_dict(filename, data):
            with open(filename, "wb") as f:
                f.write(blosc.compress(pickle.dumps(data)))
            mlflow.log_artifact(filename, "forward/inputs")
            os.remove(filename)

        plan = pd.read_csv(config["Scenarios"]["Capacity"]["FilePath"])
        plan = plan.loc[plan["ds"] <= period.year.max()]
        clustered_plants = group_plants(
            country,
            [period.year.min()],
            plan=None if plan.empty else plan,
        )[period.year.min()].reset_index()

        scen_hourly = sample_components(
            period=period, country=country, data_dir=data_dir
        )
        scen_demand_level = sample_yearly(
            period=period,
            filepath=config["Scenarios"]["Demand"]["FilePath"],
            multiplier=config["Scenarios"]["Demand"]["Multiplier"],
        )
        scen_capacity_pv = sample_yearly(
            period=period, filepath=config["Scenarios"]["CapacityPV"]["FilePath"]
        )
        scen_capacity_wind = sample_yearly(
            period=period, filepath=config["Scenarios"]["CapacityWind"]["FilePath"]
        )
        avail_factors_hist = build_availability(
            create_period(
                config["Scenarios"]["AF"]["StartDate"],
                config["Scenarios"]["AF"]["EndDate"],
            ),
            country,
            data_dir,
        )
        scen_avail_factors = {
            cluster: sample_historical(
                period=period,
                data=avail_factors_hist[cluster],
                group=config["Scenarios"]["AF"]["Group"],
            )
            for cluster in avail_factors_hist.columns
        }
        scen_permit_price = sample_yearly(
            period=period,
            filepath=config["Scenarios"]["CarbonPrice"]["FilePath"],
            multiplier=config["Scenarios"]["CarbonPrice"]["Multiplier"],
        )
        scen_fuel_price = {
            fuel: sample_yearly(
                period=period,
                filepath=config["Scenarios"]["FuelPrice"][fuel]["FilePath"],
                multiplier=config["Scenarios"]["FuelPrice"][fuel]["Multiplier"],
            )
            if fuel in config["Scenarios"]["FuelPrice"]
            else lambda: pd.Series(0, index=period)
            for fuel in clustered_plants["Fuel"].unique()
        }

        if storage:
            scen_storage_capacity = sample_yearly(
                period=period,
                filepath=config["Scenarios"]["Storage"]["Capacity"]["FilePath"],
                multiplier=config["Scenarios"]["Storage"]["Capacity"]["Multiplier"],
            )

        dataset = {}
        dataset["simulation_period"] = period
        dataset["clustered_plants"] = clustered_plants

        # Non-time series
        dataset["reserve_technologies"] = config["Market"].get(
            "ReserveParticipation",
            dataset["clustered_plants"]["Technology"].unique().tolist(),
        )
        dataset["cost_curtailment"] = config["Market"]["CostCurtailment"]
        dataset["voll"] = config["Market"]["VOLL"]

        if storage:
            dataset["storage_charge_ef"] = config["Scenarios"]["Storage"][
                "StorageChargingEfficiency"
            ]
            dataset["storage_discharge_ef"] = config["Scenarios"]["Storage"][
                "StorageDischargeEfficiency"
            ]

        for i in trange(n_inter_trials, desc="Yearly variation", position=0):
            demand_level = (
                scen_demand_level()
                .groupby(lambda x: x.year)
                .apply(lambda x: 1000 * x / len(x))
            )

            dataset["capacity_pv"] = scen_capacity_pv()
            dataset["capacity_wind"] = scen_capacity_wind()
            dataset["permit_price"] = scen_permit_price()

            fuels = []
            for fuel in clustered_plants["Fuel"].unique():
                fuels.append(scen_fuel_price.get(fuel)().to_frame(fuel))
                if isinstance(fuels[-1].index[0], tuple):
                    fuels[-1].index = fuels[-1].index.droplevel()

            dataset["fuel_price"] = pd.concat(fuels, axis=1)

            if storage:
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

            clf_one_class = LocalOutlierFactor(n_neighbors=10, novelty=True)
            scaler = MinMaxScaler(feature_range=(0, 1))
            pool = None

            for j in trange(
                n_intra_trials, desc="Daily variation", position=1, leave=False
            ):
                pca_samples = scen_hourly()
                index = pca_samples.index

                to_ignore = None
                try:
                    check_is_fitted(clf_one_class)
                except NotFittedError:
                    pass
                else:
                    scores = clf_one_class.decision_function(
                        pca_samples.values
                    ).reshape(-1, 1)
                    scores = pd.Series(
                        data=scaler.fit_transform(scores).squeeze(), index=index
                    )
                    to_ignore = scores > 0.5

                samples = pca.inverse_transform(pca_samples)
                features = {}
                for start, feature in zip(
                    range(0, samples.shape[1], 24), feature_mapping["names"]
                ):
                    features[feature] = samples[:, start : start + 24]

                for key, val in feature_mapping.items():
                    if key == "names":
                        continue
                    # demand, avail_factor_pv, avail_factor_wind, max_exports, max_imports, water_value
                    dataset[key] = (
                        pd.concat(
                            [
                                reshape(
                                    pd.DataFrame(
                                        data=features[item],
                                        columns=range(24),
                                        index=index,
                                    ),
                                    column=item,
                                )
                                for item in val
                            ],
                            axis=1,
                        )
                        if isinstance(val, list)
                        else reshape(
                            pd.DataFrame(
                                data=features[val], columns=range(24), index=index
                            )
                        )
                    )

                    if key in ("demand", "max_exports", "max_imports", "water_value"):
                        dataset[key] = dataset[key].clip(lower=0)
                    elif key in ("avail_factor_pv", "avail_factor_wind"):
                        dataset[key] = dataset[key].clip(lower=0, upper=1)

                if isinstance(demand_level.index[0], tuple):
                    demand_level.index = demand_level.index.droplevel()

                dataset["demand"] = dataset["demand"].mul(demand_level).to_frame("DA")
                dataset["demand"]["2U"] = 0
                for _, grouped in dataset["demand"]["DA"].groupby(
                    np.arange(len(dataset["demand"])) // 24
                ):
                    dataset["demand"].loc[
                        dataset["demand"]["2U"].index.isin(grouped.index), "2U"
                    ] = (np.sqrt(10 * grouped.max() + 22500) - 150)
                dataset["demand"]["2D"] = 0.5 * dataset["demand"]["2U"]

                dataset["avail_factors"] = pd.concat(
                    [
                        scen_avail_factors.get(cluster)().to_frame(cluster)
                        for cluster in avail_factors_hist.columns
                    ],
                    axis=1,
                )

                if "WAT" in dataset["fuel_price"].columns:
                    var_cost_water = 0
                    available_capacity = pd.DataFrame(
                        0,
                        index=dataset["avail_factors"].index,
                        columns=dataset["avail_factors"].columns,
                    )
                    for _, (cluster, cap) in clustered_plants[
                        ["Cluster", "TotalCapacity"]
                    ].iterrows():
                        available_capacity[cluster] = dataset["avail_factors"][
                            cluster
                        ].mul(cap)

                    for cluster in available_capacity.columns:
                        _, fuel = cluster.split("_")
                        if fuel == "WAT":
                            continue

                        efficiency, emission_rate = clustered_plants.loc[
                            clustered_plants["Cluster"] == cluster
                        ][["Efficiency", "CO2Intensity"]].iloc[0, :]

                        if isinstance(dataset["permit_price"].index[0], tuple):
                            dataset["permit_price"].index = dataset[
                                "permit_price"
                            ].index.droplevel()

                        var_cost = dataset["fuel_price"][fuel].div(
                            efficiency
                        ) + dataset["permit_price"].mul(emission_rate)
                        var_cost_water += var_cost.mul(available_capacity[cluster])

                    var_cost_water = (
                        var_cost_water.div(available_capacity.sum(axis=1))
                        .fillna(method="ffill")
                        .fillna(method="bfill")
                    )
                    dataset["fuel_price"]["WAT"] = var_cost_water

                if hydro_model is not None:
                    states = create_hydro_state(
                        availability_factor=dataset["avail_factors"].filter(
                            regex="_WAT$"
                        ),
                        water_value=dataset["water_value"],
                    )
                    hydro_avail_factor = pd.Series(
                        hydro_model.predict(states),
                        index=states.index,
                    )
                    hydro_cluster = (
                        dataset["avail_factors"].filter(regex="_WAT$").columns[0]
                    )
                    dataset["avail_factors"][hydro_cluster] = hydro_avail_factor

                if markup_model is not None:
                    states = create_markup_state(
                        load=dataset["demand"]["DA"],
                        res=dataset["capacity_pv"] * dataset["avail_factor_pv"]
                        + dataset["capacity_wind"] * dataset["avail_factor_wind"],
                        net_imports=dataset["max_exports"] - dataset["max_imports"],
                        capacity=dataset["avail_factors"].apply(
                            lambda x: x.mul(
                                dataset["clustered_plants"][
                                    ["Cluster", "TotalCapacity"]
                                ]
                                .set_index("Cluster")
                                .loc[x.name, "TotalCapacity"]
                            )
                        ),
                        water_value=dataset["water_value"],
                    )
                    markup = pd.DataFrame(
                        markup_model.predict(states),
                        index=states.index,
                        columns=dataset["avail_factors"].columns,
                    )
                    dataset["markup"] = markup

                # Run simulation
                _ = dataset.pop("water_value", None)

                if to_ignore is not None:
                    test_dataset = sample_dataset(dataset, to_ignore, warmup=warmup)
                else:
                    test_dataset = dataset

                if test_dataset:
                    test_period = test_dataset["simulation_period"]
                    sim_data = SimData(**test_dataset, action="forward")
                    sets, parameters = sim_data.build()
                    model = create_model(sets, parameters, action="forward")
                    model = run_solver(model, solver=config["Simulation"]["Solver"])

                    ll_max_power = pyomo_to_pandas(
                        model, "LLMaxPower", period=test_period
                    )["LLMaxPower"].iloc[24 * warmup :]
                    ll_ramp_down = (
                        pyomo_to_pandas(model, "LLRampDown", period=test_period)
                        .sum(axis=1)
                        .iloc[24 * warmup :]
                    )
                    ll_ramp_up = (
                        pyomo_to_pandas(model, "LLRampUp", period=test_period)
                        .sum(axis=1)
                        .iloc[24 * warmup :]
                    )
                    curtailed_power = pyomo_to_pandas(
                        model, "CurtailedPower", period=test_period
                    )["CurtailedPower"].iloc[24 * warmup :]
                    power = pyomo_to_pandas(model, "Power", period=test_period).iloc[
                        24 * warmup :, :
                    ]
                    if storage:
                        storage_input = pyomo_to_pandas(
                            model, "StorageInput", period=test_period
                        )["StorageInput"].iloc[24 * warmup :]
                        storage_output = pyomo_to_pandas(
                            model, "StorageOutput", period=test_period
                        )["StorageOutput"].iloc[24 * warmup :]

                    is_novelty = np.logical_or(
                        np.logical_or(
                            ll_max_power.groupby(lambda x: x.date).sum() > 0,
                            ll_ramp_down.groupby(lambda x: x.date).sum() > 0,
                        ),
                        ll_ramp_up.groupby(lambda x: x.date).sum() > 0,
                    )

                    pool = pd.concat(
                        (pool, pca_samples.loc[is_novelty[~is_novelty].index]),
                        axis=0,
                        ignore_index=True,
                    )
                    clf_one_class.fit(pool.values)
                else:
                    ll_max_power = pd.Series(0, index=period)
                    ll_ramp_down = pd.Series(0, index=period)
                    ll_ramp_up = pd.Series(0, index=period)
                    curtailed_power = pd.Series(0, index=period)

                if test_dataset:
                    log_dict(f"scen_inputs_{i}_{j}.dat", test_dataset)
                    log_csv(f"power_{i}_{j}.csv", power)
                    if storage:
                        log_csv(f"storage_input_{i}_{j}.csv", storage_input)
                        log_csv(f"storage_output_{i}_{j}.csv", storage_output)

                log_csv(f"ll_max_power_{i}_{j}.csv", ll_max_power)
                log_csv(f"ll_ramp_down_{i}_{j}.csv", ll_ramp_down)
                log_csv(f"ll_ramp_up_{i}_{j}.csv", ll_ramp_up)
                log_csv(f"curtailed_power_{i}_{j}.csv", curtailed_power)

        logger.info(f"run ID: {run.info.run_id}")

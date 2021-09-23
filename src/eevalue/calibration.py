# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from datetime import timedelta

import optuna

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
optuna.logging.set_verbosity(optuna.logging.ERROR)

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from eevalue.compile_data import create_dataset, subsample_dataset
from eevalue.model import ModelData, build_model_params, create_model, run_model
from eevalue.postprocessing import pyomo_to_pandas



def sample_from_period(period, days=30):
    dt = period.to_series().resample("D").first().index
    selected = np.random.choice(range(len(dt) - days - 1))
    return period[
        (period >= dt[selected]) & (period < dt[selected] + timedelta(days=days))
    ]


def loss_function(trial, dataset, clusters, solver):
    sample = sample_from_period(dataset["time_series"]["simulation_period"])
    dataset = subsample_dataset(dataset, sample)

    dataset.pop("available_capacity")
    dataset.pop("net_load")
    committed_capacity = dataset.pop("committed_capacity")

    markup = None
    plants = dataset["clustered_plants"]
    for cl in clusters:
        markup = pd.concat(
            (
                markup,
                pd.DataFrame(
                    data=trial.suggest_int(f"markup_{cl}", 0, 100),
                    index=committed_capacity.index,
                    columns=[cl],
                ),
            ),
            axis=1,
        )
        plants.loc[plants["Cluster"] == cl, "RampUpRate"] *= trial.suggest_float(
            f"rur_{cl}", 0.1, 1
        )

    dataset["markup"] = markup
    model_data = ModelData(**dataset, calibration=True)
    sets, parameters = build_model_params(model_data, calibration=True)
    model = create_model(sets, parameters, calibration=True)
    model = run_model(model, solver=solver)
    predicted_generation = pyomo_to_pandas(model, "Power").set_index(sample)

    n_obs = len(committed_capacity)
    warmup = int(0.1 * n_obs)
    error = mean_squared_error(
        committed_capacity.iloc[warmup:, :], predicted_generation.iloc[warmup:, :]
    )
    return error


def calibrate(country, period, solver="glpk", budget=200, has_hydro=True):
    dataset = create_dataset(
            country=country,
            period=period,
            calibration=True,
            exclude=["ramping_events", "capacity_margin", "water_value"],
        )
    
    if has_hydro:
        fuel_prices = dataset["time_series"]["fuel_price"]
        if np.allclose(fuel_prices["WAT"], 0):
            fuel_prices["WAT"] = fuel_prices[fuel_prices > 0.01].min(axis=1)

    committed_capacity = dataset["time_series"]["committed_capacity"]
    clusters = committed_capacity.columns.to_list()

    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(multivariate=True),
    )
   
    study.optimize(
        lambda trial: loss_function(trial, dataset, clusters, solver),
        n_trials=budget,
        show_progress_bar=True,
    )
    return study

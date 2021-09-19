# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from concurrent import futures
from datetime import timedelta

import nevergrad as ng
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder

from eevalue._polynomial import SplineTransformer
from eevalue.compile_data import create_dataset, subsample_dataset
from eevalue.model import ModelData, build_model_params, create_model, run_model
from eevalue.postprocessing import pyomo_to_pandas


def sample_from_period(period, days=30):
    dt = period.to_series().resample("D").first().index
    selected = np.random.choice(range(len(dt) - days - 1))
    return period[
        (period >= dt[selected]) & (period < dt[selected] + timedelta(days=days))
    ]


def loss_function(dataset, features, period, encoder, solver, *weights):
    sample = sample_from_period(period)
    dataset = subsample_dataset(dataset, sample)

    dataset.pop("available_capacity")
    dataset.pop("net_load")
    committed_capacity = dataset.pop("committed_capacity")

    markup_input = pd.concat([dataset.pop(name) for name in features], axis=1)
    markup_input = encoder.transform(markup_input)

    markup = pd.DataFrame(
        data=np.concatenate([10 * np.matmul(markup_input, w) for w in weights], axis=1),
        index=committed_capacity.index,
        columns=committed_capacity.columns,
    )

    dataset["markup"] = markup
    model_data = ModelData(**dataset, calibration=True)
    sets, parameters = build_model_params(model_data, calibration=True)
    model = create_model(sets, parameters, calibration=True)
    results = run_model(model, solver=solver)
    predicted_generation_cl = pyomo_to_pandas(results, "Power").set_index(sample)

    plants = dataset["clustered_plants"]
    combinations = set(zip(plants["Technology"], plants["Fuel"]))

    predicted_generation = None
    for item in combinations:
        t, f = item
        clusters = plants[(plants["Technology"] == t) & (plants["Fuel"] == f)][
            "Cluster"
        ].tolist()
        pred = pd.Series(0, index=sample)
        for cl in clusters:
            pred += predicted_generation_cl[cl]
        predicted_generation = pd.concat(
            [predicted_generation, pred.to_frame(f"{t}_{f}")], axis=1
        )

    n_obs = len(committed_capacity)
    warmup = int(0.1 * n_obs)
    error = np.mean(
        np.abs(
            committed_capacity.iloc[warmup:, :] - predicted_generation.iloc[warmup:, :]
        ).sum(axis=1)
    )
    return error


def calibrate(
    country, period, solver="glpk", budget=500, num_workers=1, has_hydro=True
):
    dataset = create_dataset(country=country, period=period, calibration=True)

    if has_hydro:
        features = ["capacity_margin", "ramping_events", "water_value"]
    else:
        features = ["capacity_margin", "ramping_events"]

    markup_input = pd.concat(
        [dataset["time_series"].get(name) for name in features], axis=1
    )

    categorical_pipeline = Pipeline(
        [
            (
                "select",
                ColumnTransformer(
                    [
                        (
                            "select",
                            "passthrough",
                            ["RU", "RD"],
                        )
                    ],
                    remainder="drop",
                ),
            ),
            ("encode", OneHotEncoder(sparse=False)),
        ]
    )

    numerical_pipeline = Pipeline(
        [
            (
                "select",
                ColumnTransformer(
                    [
                        (
                            "select",
                            "passthrough",
                            ["capacity_margin", "water_value"]
                            if has_hydro
                            else ["capacity_margin"],
                        )
                    ],
                    remainder="drop",
                ),
            ),
            (
                "encode",
                SplineTransformer(
                    n_knots=5,
                    degree=3,
                    knots="uniform",
                    extrapolation="constant",
                    include_bias=True,
                ),
            ),
        ]
    )


    encoder = FeatureUnion(
        [
            ("categorical", categorical_pipeline),
            ("numerical", numerical_pipeline),
        ]
    )
    encoder = encoder.fit(markup_input)

    n_output_features = 0
    pipelines = dict(encoder.transformer_list)
    for item in pipelines["categorical"]["encode"].categories_:
        n_output_features += len(np.unique(item))
    n_output_features += pipelines["numerical"]["encode"].n_features_out_

    weights = []
    for _ in range(dataset["non_time_series"]["clustered_plants"].shape[0]):
        weights.append(ng.p.Array(shape=(n_output_features, 1)))

    instru = ng.p.Instrumentation(dataset, features, period, encoder, solver, *weights)

    optimizer = ng.optimizers.NGOpt(
        parametrization=instru, budget=budget, num_workers=num_workers
    )

    if (num_workers is not None) and (num_workers > 1):
        with futures.ProcessPoolExecutor(max_workers=optimizer.num_workers) as executor:
            recommendation = optimizer.minimize(
                loss_function, executor=executor, batch_mode=False
            )
    else:
        recommendation = optimizer.minimize(loss_function)

    return recommendation.value

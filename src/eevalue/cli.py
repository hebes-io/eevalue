# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import pickle
from datetime import timedelta

import click
import numpy as np
import pandas as pd

from eevalue.calibration import calibrate
from eevalue.compile_data import create_dataset
from eevalue.model import ModelData, build_model_params, create_model, run_model
from eevalue.postprocessing import pyomo_to_pandas
from eevalue.settings import DATA_PATH


CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

CALIBRATE_HELP = """Optimize markup using calibration period data."""
VALID_HELP = """Validate the optimized markup."""

COUNTRY_ARG_HELP = """The name (ISO 3166 alpha-2 codes, except for Greece, for which
the abbreviation EL is used) of the country to run the command for."""
DS_ARG_HELP = """The first date of the calibration, validation or simulation period
(%d-%m-%Y, like 1-1-2020)."""
DE_ARG_HELP = """The last date of the calibration, validation or simulation period
(%d-%m-%Y, like 31-12-2020)."""
BUDGET_ARG_HELP = """The budget for the calibration (number of optimization trials)."""
HYDRO_ARG_HELP = (
    """Flag which indicates that the modelled system has not hydroelectric plants."""
)
SOLVER_ARG_HELP = """The name of the solver (use pyomo help --solvers to see the
supported solvers)."""


@click.group(context_settings=CONTEXT_SETTINGS, name=__file__)
def cli():
    """Command line tool for running the eevalue package."""


@cli.command("calibrate", help=CALIBRATE_HELP)
@click.option("--country", "-c", type=str, required=True, help=COUNTRY_ARG_HELP)
@click.option(
    "--date-start",
    "-ds",
    type=click.DateTime(formats=["%d-%m-%Y"]),
    required=True,
    help=DS_ARG_HELP,
)
@click.option(
    "--date-end",
    "-de",
    type=click.DateTime(formats=["%d-%m-%Y"]),
    required=True,
    help=DE_ARG_HELP,
)
@click.option("--budget", type=int, default=200, help=BUDGET_ARG_HELP)
@click.option("--has-hydro", type=bool, default=True, help=HYDRO_ARG_HELP)
@click.option("--solver", type=str, default="glpk", help=SOLVER_ARG_HELP)
def run_calibration(country, date_start, date_end, budget, has_hydro, solver):
    date_end = date_end + timedelta(days=1)
    period = pd.date_range(start=date_start, end=date_end, freq="H", closed="left")

    opt_values = calibrate(
        country,
        period,
        solver=solver,
        budget=budget,
        has_hydro=has_hydro,
    )
    file_path = os.path.join(DATA_PATH, f"Opt/{country}", "markup.pickle")
    with open(file_path, "wb") as f:
        pickle.dump(opt_values, f)


@cli.command("validate", help=VALID_HELP)
@click.option("--country", "-c", type=str, required=True, help=COUNTRY_ARG_HELP)
@click.option(
    "--date-start",
    "-ds",
    type=click.DateTime(formats=["%d-%m-%Y"]),
    required=True,
    help=DS_ARG_HELP,
)
@click.option(
    "--date-end",
    "-de",
    type=click.DateTime(formats=["%d-%m-%Y"]),
    required=True,
    help=DE_ARG_HELP,
)
@click.option("--has-hydro", type=bool, default=True, help=HYDRO_ARG_HELP)
@click.option("--solver", type=str, default="gurobi", help=SOLVER_ARG_HELP)
def run_validation(country, date_start, date_end, has_hydro, solver):
    file_path = os.path.join(DATA_PATH, f"Opt/{country}", "markup.pickle")
    with open(file_path, "rb") as f:
        results = pickle.load(f)

    date_end = date_end + timedelta(days=1)
    period = pd.date_range(start=date_start, end=date_end, freq="H", closed="left")

    dataset = create_dataset(
        country=country,
        period=period,
        calibration=True,
        return_flat=True,
        exclude=["capacity_margin", "ramping_events", "water_value"],
    )

    if has_hydro:
        fuel_prices = dataset["fuel_price"]
        if np.allclose(fuel_prices["WAT"], 0):
            fuel_prices["WAT"] = fuel_prices[fuel_prices > 0.01].min(axis=1)

    dataset.pop("available_capacity")
    dataset.pop("net_load")
    committed_capacity = dataset.pop("committed_capacity")

    markup = None
    plants = dataset["clustered_plants"]
    for key, value in results.best_trial.params.items():
        if key.startswith("markup_"):
            cl = key.split("_", 1)[-1]
            markup = pd.concat(
                (
                    markup,
                    pd.DataFrame(
                        data=value, index=committed_capacity.index, columns=[cl]
                    ),
                ),
                axis=1,
            )
        elif key.startswith("rur_"):
            cl = key.split("_", 1)[-1]
            plants.loc[plants["Cluster"] == cl, "RampUpRate"] *= value

    dataset["markup"] = markup
    model_data = ModelData(**dataset, calibration=True)
    sets, parameters = build_model_params(model_data, calibration=True)
    model = create_model(sets, parameters, calibration=True)
    model = run_model(model, solver=solver)
    predicted_generation = pyomo_to_pandas(model, "Power").set_index(period)
    predicted_generation = predicted_generation.rename(
        columns={col: "_".join([col, "pred"]) for col in predicted_generation.columns}
    )

    file_path = os.path.join(DATA_PATH, f"Validation/{country}", "results.csv")
    pd.concat((committed_capacity, predicted_generation), axis=1).to_csv(
        file_path, index=True
    )

# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from functools import reduce

import click
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LassoCV

import eevalue.logging as logging
from eevalue.cli.common import (
    LOCAL_FILE_URI_PREFIX,
    TAG_ARG_HELP,
    TR_URI_HELP,
    get_artifact_path,
)
from eevalue.simulate.postprocess import assemble_results

BASE_RUN_ID_HELP = """The ID string for the run that contains the reference scenarios (from
a forward simulation)."""
COMP_HELP = """Compare a reference and a counterfactual case to quantify the impact of load
modifying resources on the selected system state."""
COUNTER_RUN_ID_HELP = """The ID string for the run that contains the counterfactual scenarios
(from a replay simulation)."""
EXP_NAME_HELP = """The name of the experiment for experiment tracking."""
SYS_STATE_HELP = """The name of the system state to compare. It can be one of 'deficit', 'surplus',
'power'. For the specific case of 'power', it can also be provided as 'power.NAME, where NAME is
the name of a specific technology cluster."""


################################################################################################
# Utility functions
################################################################################################


def get_load_change(run_id, tracking_uri):
    path = get_artifact_path(run_id, tracking_uri, "forward", "results")
    result = pd.merge(
        assemble_results("LoadModeUp", path),
        assemble_results("LoadModeDown", path),
        left_on=["ds", "scen"],
        right_on=["ds", "scen"],
    )
    result["load_change"] = result["LoadModeUp"] - result["LoadModeDown"]
    result = result.drop(["LoadModeUp", "LoadModeDown"], axis=1)
    return result


def get_system_state(
    state: str, path: str, select_only: str = None, suffix: str = "pre"
):
    effect_mapping = {
        "deficit": ["LLMaxPower", "LLRampUp"],
        "surplus": ["LLRampDown", "CurtailedPower"],
        "power": "Power",
    }

    if isinstance(effect_mapping[state], list):
        result = reduce(
            lambda left, right: pd.merge(
                left, right, left_on=["ds", "scen"], right_on=["ds", "scen"]
            ),
            [assemble_results(name, path) for name in effect_mapping[state]],
        )
    else:
        result = assemble_results(effect_mapping[state], path)

    if select_only is None:
        result[f"{state}_{suffix}"] = result[
            [col for col in result.columns if col not in ("ds", "scen")]
        ].sum(axis=1)
        result = result.drop(
            [
                col
                for col in result.columns
                if col not in ("ds", "scen", f"{state}_{suffix}")
            ],
            axis=1,
        )
    else:
        result = result.drop(
            [col for col in result.columns if col not in ("ds", "scen", select_only)],
            axis=1,
        )
        result = result.rename(columns={select_only: f"{state}_{suffix}"})

    return result


def get_state_change(state, base_run_id, counter_run_id, tracking_uri):
    select_only = None
    if "." in state:
        state, select_only = state.split(".", 1)

    state_pre = get_system_state(
        state,
        get_artifact_path(base_run_id, tracking_uri, "forward", "results"),
        select_only=select_only,
        suffix="pre",
    )
    state_post = get_system_state(
        state,
        get_artifact_path(counter_run_id, tracking_uri, "forward", "results"),
        select_only=select_only,
        suffix="post",
    )
    state_change = state_pre.merge(
        state_post, left_on=["ds", "scen"], right_on=["ds", "scen"]
    )
    state_change["state_change"] = (
        state_change[f"{state}_post"] - state_change[f"{state}_pre"]
    )
    state_change = state_change.drop([f"{state}_post", f"{state}_pre"], axis=1)
    return state_change


def fit_predictor(
    load_change: pd.DataFrame, state_change: pd.DataFrame, sys_state: str
):
    hour_of_year = (load_change["ds"].dt.dayofyear - 1) * 24 + load_change["ds"].dt.hour
    load_change["hour_of_year"] = hour_of_year
    X = load_change.pivot(
        index="scen", columns="hour_of_year", values="load_change"
    ).fillna(value=0)

    if sys_state == "deficit":
        y = state_change.groupby("scen")["state_change"].min()[X.index]
        positive = True
        multiplier = 1
    elif sys_state == "surplus":
        y = state_change.groupby("scen")["state_change"].sum()[X.index]
        positive = True
        multiplier = -1
    elif sys_state == "power":
        y = state_change.groupby("scen")["state_change"].sum()[X.index]
        positive = True
        multiplier = 1
    else:
        y = state_change.groupby("scen")["state_change"].sum()[X.index]
        positive = False
        multiplier = 1

    y = y.mul(multiplier)
    model = LassoCV(
        fit_intercept=True, positive=positive, cv=3, max_iter=5000, tol=0.01
    ).fit(X, y)
    coef = pd.DataFrame(
        data=multiplier * model.coef_, columns=["coef"], index=X.columns
    )
    return coef / coef.sum()


################################################################################################
# CLI commands
################################################################################################


@click.group(name="eevalue")
def comp_cli():
    pass


@comp_cli.command("compare", help=COMP_HELP)
@click.option("--sys-state", "-sys", type=str, required=True, help=SYS_STATE_HELP)
@click.option(
    "--base-run-id",
    "-bri",
    type=str,
    help=BASE_RUN_ID_HELP,
)
@click.option(
    "--counter-run-id",
    "-cri",
    type=str,
    help=COUNTER_RUN_ID_HELP,
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
def compare(
    sys_state,
    base_run_id,
    counter_run_id,
    experiment_name,
    tracking_uri,
    tags,
):
    if ("." in sys_state) and (sys_state.split(".", 1)[0] != "power"):
        raise ValueError(
            "If '.' is part of `sys_state`, it is expected as 'power.*'. "
            f"Got {sys_state.split('.', 1)[0]}.*"
        )
    elif ("." not in sys_state) and (sys_state not in ["deficit", "surplus", "power"]):
        raise ValueError(
            "`sys_state` can be one of 'deficit', 'surplus' or 'power'. "
            f"Got {sys_state}."
        )

    logger = logging.getLogger("eevalue:compare")
    logger.setLevel(20)

    tracking_uri = tracking_uri or f"{LOCAL_FILE_URI_PREFIX}./eevalue/tracking/mlruns"
    mlflow.set_tracking_uri(tracking_uri)

    counter_run = mlflow.get_run(counter_run_id)
    load_shaping = counter_run.data.tags.get("load_shaping", False)
    if not load_shaping:
        raise ValueError(
            "The run that corresponds to the `counter_run_id` must have load "
            "modifying resources enabled"
        )

    mode = counter_run.data.tags.get("load_shape_mode", "")
    if (sys_state == "deficit") and (mode not in ["down", "flex"]):
        raise ValueError(
            "Comparison for capacity deficit needs a counterfactual run with "
            f"`load_shape_mode` 'down' or 'flex'. Got {mode}"
        )
    if (sys_state == "surplus") and (mode not in ["up", "flex"]):
        raise ValueError(
            "Comparison for capacity surplus needs a counterfactual run with "
            f"`load_shape_mode` 'up' or 'flex'. Got {mode}"
        )

    tags = (
        {}
        if not tags
        else {key: val for key, val in [item.split("=", 1) for item in tags]}
    )
    tags.update(
        {
            "country": counter_run.data.tags["country"],
            "action": "compare",
            "base_run_id": base_run_id,
            "counter_run_id": counter_run_id,
            "state": sys_state,
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

    with mlflow.start_run(experiment_id=experiment_id) as run:
        mlflow.set_tags(tags)

        def log_csv(filename, data):
            data.to_csv(
                filename,
                index_label="hour_of_year",
            )
            mlflow.log_artifact(filename, f"compare/{sys_state}/results")
            os.remove(filename)

        load_change = get_load_change(counter_run_id, tracking_uri)
        state_change = get_state_change(
            sys_state, base_run_id, counter_run_id, tracking_uri
        )

        results = fit_predictor(load_change, state_change, sys_state)
        log_csv("results.csv", results)
        logger.info(f"run ID: {run.info.run_id}")

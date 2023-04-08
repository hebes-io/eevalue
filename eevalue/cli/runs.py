# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import datetime

import click
import rich
from mlflow.tracking import MlflowClient
from rich.table import Table

import eevalue.logging as logging

from .common import (
    LOCAL_FILE_URI_PREFIX,
    MAX_RESULTS_HELP,
    RUNS_HELP,
    TAG_ARG_HELP,
    TR_URI_HELP,
)

LOCAL_TIMEZONE = datetime.datetime.now().astimezone().tzinfo
EXP_NAME_HELP = """The name of the experiment for experiment tracking."""


@click.group(name="eevalue")
def runs_cli():
    pass


@runs_cli.command("runs", help=RUNS_HELP)
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
@click.option(
    "--max-results",
    "-mr",
    type=int,
    default=10,
    help=MAX_RESULTS_HELP,
)
@click.option("--tag", "-t", "tags", type=str, multiple=True, help=TAG_ARG_HELP)
def runs(experiment_name, tracking_uri, max_results, tags):
    logger = logging.getLogger("eevalue:runs")
    logger.setLevel(20)

    tracking_uri = tracking_uri or f"{LOCAL_FILE_URI_PREFIX}./eevalue/tracking/mlruns"
    tags = (
        {}
        if not tags
        else {key: val for key, val in [item.split("=", 1) for item in tags]}
    )

    mlflow_client = MlflowClient(tracking_uri)
    expt = mlflow_client.get_experiment_by_name(experiment_name)
    if expt is not None:
        experiment_id = expt.experiment_id
    else:
        raise ValueError("")

    query = " and ".join([f"tags.{key} = '{val}'" for key, val in tags.items()])
    runs = mlflow_client.search_runs(
        experiment_ids=experiment_id, max_results=max_results, filter_string=query
    )

    table = Table(show_header=True, header_style="bold #2070b2", box=rich.box.SIMPLE)
    table.add_column("Run id")
    table.add_column("Action")
    table.add_column("Start time")
    table.add_column("End time")

    for run in runs:
        table.add_row(
            run.info.run_id,
            run.data.tags["action"],
            datetime.datetime.fromtimestamp(
                run.info.start_time / 1000.0, tz=LOCAL_TIMEZONE
            ).strftime("%Y-%m-%d %H:%M:%S"),
            datetime.datetime.fromtimestamp(
                run.info.end_time / 1000.0, tz=LOCAL_TIMEZONE
            ).strftime("%Y-%m-%d %H:%M:%S"),
        )

    click.secho(rich.print(table))

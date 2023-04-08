# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
import pickle

import click
import mlflow
import numpy as np
import pandas as pd
from scipy.stats import cumfreq
from tqdm.auto import tqdm

import eevalue.logging as logging
from eevalue.utils import as_series

from .common import LOCAL_FILE_URI_PREFIX, TR_URI_HELP, get_artifact_path

ORDC_HELP = """Estimate and store the operating reserve demand curve (ORDC) of the
simulated system."""
RUN_ID_HELP = """The ID string for the run that contains the simulated scenarios for
the relation between available capacity and probability of capacity deficit."""


@click.group(name="eevalue")
def ordc_cli():
    pass


@ordc_cli.command("ordc", help=ORDC_HELP)
@click.option(
    "--run-id",
    "-ri",
    required=True,
    type=str,
    help=RUN_ID_HELP,
)
@click.option(
    "--tracking-uri",
    "-tu",
    type=str,
    default=None,
    envvar="MLFLOW_TRACKING_URI",
    help=TR_URI_HELP,
)
def ordc(run_id, tracking_uri):
    logger = logging.getLogger("eevalue:backtest")
    logger.setLevel(20)

    tracking_uri = tracking_uri or f"{LOCAL_FILE_URI_PREFIX}./eevalue/tracking/mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    inputs_path = get_artifact_path(run_id, tracking_uri, "forward", "inputs")
    results_path = get_artifact_path(run_id, tracking_uri, "forward", "results")

    with mlflow.start_run(run_id=run_id) as run:

        def log_dict(filename, data):
            with open(filename, "wb") as f:
                f.write(pickle.dumps(data))
            mlflow.log_artifact(filename, "forward/ordc")
            os.remove(filename)

        all_data = []

        with tqdm() as pbar:
            for filename in glob.glob(inputs_path + "/scen_inputs_*"):
                basename = os.path.basename(filename)
                i, j = basename.split(".")[0].rsplit("_", 2)[-2:]

                max_deficit = np.maximum(
                    as_series(
                        pd.read_csv(
                            os.path.join(results_path, f"ll_max_power_{i}_{j}.csv"),
                            compression="gzip",
                        ).set_index("ds")
                    ),
                    as_series(
                        pd.read_csv(
                            os.path.join(results_path, f"ll_ramp_up_{i}_{j}.csv"),
                            compression="gzip",
                        ).set_index("ds")
                    ),
                )
                all_data.append(max_deficit.max())
                pbar.update(1)

        res = cumfreq(all_data, numbins=int(len(all_data) / 2))
        res = pd.Series(
            data=res.cumcount,
            index=res.lowerlimit
            + np.linspace(0, res.binsize * res.cumcount.size, res.cumcount.size),
        )
        res = 1 - (res / res.max())
        log_dict(
            "curve.dat", dict(data=res.to_frame("LOLP"), reference=np.mean(all_data))
        )

    logger.info(f"run ID: {run.info.run_id}")

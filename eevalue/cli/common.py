# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from urllib.parse import urlparse

from mlflow.tracking import MlflowClient

LOCAL_FILE_URI_PREFIX = "file:"

CONFIG_HELP = """The path to a YAML configuration file to load parameters from. If not
provided, `eevalue/config/{country}/config.yml` will be used."""
COUNTRY_HELP = """The name of the country to run the command for (in ISO 3166
alpha-2 codes, except for Greece, for which the abbreviation EL is used)."""
DATA_DIR_HELP = """The directory where artifacts produced during preprocessing are stored.
If not provided, the function will first search for a `EEVALUE_DATA_DIR` environmemt variable,
and if not set, the value in `eevalue.settings.DATA_DIR` will be used."""
END_DATE_HELP = (
    """The last date of the simulation period (in %d-%m-%Y format, like 31-12-2020)."""
)
HYDRO_FLAG_HELP = """Flag to indicate whether calibration should learn the effective availability
of hydropower."""
MARKUP_FLAG_HELP = (
    """Flag to indicate whether calibration should learn a cost markup function."""
)
MAX_RESULTS_HELP = """Maximum number of runs to print."""
RUNS_HELP = "Print out the stored MLflow runs (filtered by tags)."
START_DATE_HELP = (
    """The first date of the simulation period (in %d-%m-%Y format, like 1-1-2020)."""
)
TAG_ARG_HELP = """One or more tags for the run. The tags should be provided in
`key=value` format, for example `-t country=EL`."""
TR_URI_HELP = """Address of local or remote tracking server. If not provided, defaults to
`MLFLOW_TRACKING_URI` environment variable if set, otherwise to `./eevalue/tracking/mlruns`."""


def get_artifact_path(run_id, tracking_uri, *args):
    mlflow_client = MlflowClient(tracking_uri)
    run = mlflow_client.get_run(run_id)
    return os.path.abspath(
        os.path.join(
            run.info.artifact_uri.split(LOCAL_FILE_URI_PREFIX + "///")[-1], *args
        )
    )

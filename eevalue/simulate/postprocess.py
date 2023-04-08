# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
from datetime import datetime

import pandas as pd
import pyomo


def get_sets(model, varname):
    """Get sets that correspond to a pyomo model's variable or parameter.

    Args:
        model (pyomo.environ.ConcreteModel): An optimized pyomo model instance.
        varname (str): The name of the variable or parameter.

    Returns:
        list: A list with the sets that correspond to the variable or parameter.
    """
    var = getattr(model, varname)

    if var.dim() > 1:
        sets = [pset.getname() for pset in var.index_set().subsets()]
    else:
        sets = [var.index_set().name]
    return sets


def get_set_members(model, sets):
    """Get set members for a list of sets.

    Args:
        model (pyomo.environ.ConcreteModel): An optimized pyomo model instance.
        sets (list): List of strings with the set names.

    Returns:
        list: A list with the set members.
    """
    sm = []
    for s in sets:
        sm.append([v for v in getattr(model, s).data()])
    return sm


def pyomo_to_pandas(model, varname, period=None):
    """Convert a pyomo variable or parameter into a pandas dataframe.

    Args:
        model (pyomo.environ.ConcreteModel): An optimized pyomo model instance.
        varname (str): The name of the variable or parameter.
        period (pandas.DatetimeIndex with freq='H', optional): The period over which
            to gather the data. Defaults to None.

    Returns:
        pandas.DataFrame: The variable or parameter time series.
    """
    setnames = get_sets(model, varname)
    sets = get_set_members(model, setnames)
    var = getattr(model, varname)

    if len(sets) != var.dim():
        raise ValueError(
            "The number of provided set lists ("
            + str(len(sets))
            + ") does not match the dimensions of the variable ("
            + str(var.dim())
            + ")"
        )

    if var.dim() == 1:
        [SecondSet] = sets
        out = pd.DataFrame(columns=[var.name], index=SecondSet)
        data = var.get_values()
        for idx in data:
            out[var.name][idx] = data[idx]

        if period is not None:
            out = out.set_index(period)
    elif var.dim() == 2:
        [FirstSet, SecondSet] = sets
        out = pd.DataFrame(columns=FirstSet, index=SecondSet)
        if isinstance(var, pyomo.core.base.var.IndexedVar):
            var = var.get_values()

        for idx in var:
            out[idx[0]][idx[1]] = var[idx]

        if period is not None:
            out = out.set_index(period)
    else:
        raise ValueError("The function only accepts one or two-dimensional variables")

    return out


def assemble_results(target: str, path: str) -> pd.DataFrame:
    """Assemble results from forward simulations.

    Args:
        target (str): The data to assemble. It can be one of "LLMaxPower", "LLRampDown",
            "LLRampUp", "CurtailedPower", "Power", "LoadModeUp", "LoadModeDown",
            "StorageInput", "StorageOutput,
        path (str): The path to the relevant files. The files are expected to be gzipped
            csv files.

    Returns:
        pandas.DataFrame: The dataframe with the assembled data.
    """
    dataset = []
    file_mapping = {
        "LLMaxPower": "ll_max_power",
        "LLRampDown": "ll_ramp_down",
        "LLRampUp": "ll_ramp_up",
        "CurtailedPower": "curtailed_power",
        "Power": "power",
        "LoadModeUp": "load_mode_up",
        "LoadModeDown": "load_mode_down",
        "StorageInput": "storage_input",
        "StorageOutput": "storage_output",
    }

    filename = file_mapping[target]

    for file in glob.glob(path + f"/{filename}_*"):
        basename = os.path.basename(file)
        scen = "_".join(basename.split(".")[0].rsplit("_", 2)[-2:])
        data = pd.read_csv(file, compression="gzip")
        data["scen"] = scen
        dataset.append(data)

    dataset = pd.concat(dataset, axis=0, ignore_index=True)
    dataset["ds"] = dataset["ds"].map(
        lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    )

    columns = dataset.columns
    if (len(columns) == 3) and (target not in columns):
        dataset = dataset.rename(
            columns={col: target for col in columns if col not in ("ds", "scen")}
        )

    return dataset

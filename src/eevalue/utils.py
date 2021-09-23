# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
from sklearn.utils.validation import column_or_1d


def as_list(val):
    """Helper function, always returns a list of the input value."""
    if isinstance(val, str):
        return [val]
    if hasattr(val, "__iter__"):
        return list(val)
    if val is None:
        return []
    return [val]


def as_series(x):
    """Helper function to cast an iterable to a Pandas Series object."""
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0]
    else:
        return pd.Series(column_or_1d(x))

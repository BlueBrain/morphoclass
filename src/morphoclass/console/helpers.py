# Copyright © 2022-2022 Blue Brain Project/EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Helper functions for morphoclass checkpoints."""
from __future__ import annotations

from typing import Callable

import click


def validate_params(ctx, param, value):
    """Validate click input for method parameters.

    Parameters
    ----------
    ctx : click.Context
        Click context.
    param : str
        Parameter name.
    value : str
        Value to check if it has the correct parameres from.

    Returns
    -------
    val : dict
        Converted value.

    Raises
    ------
    ValueError
        If format is not parsable to dictionary.
    """
    try:
        val = params_dict(value)
        return val
    except ValueError:
        raise click.BadParameter("Format of parameters must be 'a=1 b=2 c=somestr'")


def params_dict(value):
    """Covert string to dictionary.

    This conversion is needed for converting method parameters given as string
    in the command line. E.g., random_state=113587 objective="multi:softmax".

    Parameters
    ----------
    value : str
        Input parameter to be converted to corresponding type, or
        leave it as a string as a last option.

    Returns
    -------
    dict
        Value converted to a dictionary.

    Raises
    ------
    ValueError
        If any of the partitions is empty string, this is not the dict type.
    """
    d = {}
    if value is not None:
        for p in value.split(" "):
            k, e, v = p.partition("=")
            if k == "" or e == "" or v == "":
                raise ValueError
            d[k] = convert(v)
    return d


def convert(val):
    """Convert value from str to python type.

    Parameters
    ----------
    val : str
        Input parameter to be converted to corresponding type, or
        leave it as a string as a last option.

    Returns
    -------
    int, float, str, dict
        Value converted to a corresponding type.
    """
    constructors: list[Callable] = [int, float, params_dict, str]
    for c in constructors:
        if val is None or val == "None":
            return None
        try:
            return c(val)
        except (ValueError, IndexError):
            pass


def ctx2dict(ctx):
    """Convert click context to dictionary.

    Parameters
    ----------
    ctx : click.Context, list
        Context object with unspecified parameters (used as kwargs for models, etc.).
        E.g., ['--a', '1', '--b', 'sth', '--c', 'd=1 g=2']

    Returns
    -------
    config_params : dict
        Context content as dictionary.

    Note
    ----
    It won't happen that the keys start with only one dash, since
    we didn't specify these parameters explicitly with their shorter names.
    """
    keys = ctx.args[::2]
    # remove prefix dashes, and replace rest with underscores
    keys = [k[2:].replace("-", "_") for k in keys]
    values = ctx.args[1::2]

    params = dict(zip(keys, values))
    config_params = params2dict(params)

    return config_params


def params2dict(params):
    """Convert dictionary and nested dictionaries with string values to correct type.

    Parameters
    ----------
    params : dict
        Dictionary of string values and/or nested dictionaries with string values.

    Returns
    -------
    config_params : dict
        Dictionary with correct value types.
    """
    config_params = {}
    for key, val in params.items():
        config_params[key] = convert(val)

        # if the key is set of empty parameters, return empty dict
        if key.endswith("_params") and config_params[key] == "":
            config_params[key] = {}

    return config_params

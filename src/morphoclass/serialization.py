# Copyright © 2022 Blue Brain Project/EPFL
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
"""Module with tools and utilities for data and model serialization."""
from __future__ import annotations

import time
from pathlib import Path


def get_timestamp(t=None):
    """Generate a timestamp.

    Parameters
    ----------
    t : time.struct_time (optional)
        A given time. If not provided then `time.localtime()` is taken.

    Returns
    -------
    str
        A timestamp.
    """
    if t is None:
        t = time.localtime()
    return time.strftime("%Y%m%d_%H%M%S", t)


def get_experiment_directory(log_base_dir, experiment_name):
    """Create an empty directory for for logging from an experiment.

    Parameters
    ----------
    log_base_dir : str or pathlib.Path
        The base directory for logging. The experiment directory will be a
        subdirectory of it.
    experiment_name : str
        The name of the experiment.

    Returns
    -------
    pathlib.Path
        The created empty experiment logging directory.
    """
    log_base_dir = Path(log_base_dir)
    if not log_base_dir.exists():
        raise ValueError(f"Log base directory does not exist ({log_base_dir})")

    timestamp = get_timestamp()
    log_dir_name = f"{timestamp}_{experiment_name}"
    log_dir = log_base_dir / log_dir_name
    if log_dir.exists():
        raise ValueError(
            f"Log directory for given experiment already exists({log_dir})"
        )
    log_dir.mkdir()

    return log_dir

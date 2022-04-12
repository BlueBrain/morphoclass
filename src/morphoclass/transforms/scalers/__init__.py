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
"""Various feature scalers for morphology node features."""
from __future__ import annotations

import sys

from morphoclass.transforms.scalers.abstract_scaler import AbstractFeatureScaler
from morphoclass.transforms.scalers.manual_scaler import FeatureManualScaler
from morphoclass.transforms.scalers.min_max_scaler import FeatureMinMaxScaler
from morphoclass.transforms.scalers.robust_scaler import FeatureRobustScaler
from morphoclass.transforms.scalers.standard_scaler import FeatureStandardScaler


def scaler_from_config(config):
    """Reconstruct scaler from a config.

    Parameters
    ----------
    config : dict
        The configuration returned by `get_config` of a scaler class.

    Returns
    -------
    obj : object
        The reconstructed scaler.
    """
    scaler_cls = getattr(sys.modules[__name__], config["scaler_cls"])
    scaler_obj = scaler_cls(*config["scaler_args"], **config["scaler_kwargs"])
    scaler_obj.reconstruct(config["params"])

    return scaler_obj


__all__ = [
    "AbstractFeatureScaler",
    "FeatureManualScaler",
    "FeatureMinMaxScaler",
    "FeatureRobustScaler",
    "FeatureStandardScaler",
    "scaler_from_config",
]

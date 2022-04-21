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
"""Implementation of the manual scaler transform."""
from __future__ import annotations

import torch

from morphoclass.transforms.scalers import AbstractFeatureScaler


class FeatureManualScaler(AbstractFeatureScaler):
    """Scaler that shifts and scales the features by fixed values.

    The new features are computed by features -> (features - shift) / scale

    Parameters
    ----------
    feature_indices
        List of indices of the feature maps to which to apply the scaling.
    shift : float
        The fixed offset subtracted from the features
    scale : float
        The fixed scale by which the shifted features are divided.
    kwargs
        Additional keyword argument to pass through to the `AbstractFeatureScaler`
        base class.
    """

    def __init__(self, feature_indices, shift=0, scale=1, **kwargs):
        super().__init__(feature_indices, **kwargs)
        self.shift = torch.tensor(shift, dtype=torch.get_default_dtype())
        self.scale = torch.tensor(scale, dtype=torch.get_default_dtype())

    def _fit(self, features):
        pass

    def _transform(self, features):
        return (features - self.shift) / self.scale

    def get_config(self):
        """Generate the configuration necessary for reconstructing the scaler.

        Returns
        -------
        config : dict
            The configuration of the scaler. It should contain all
            information necessary for reconstructing the scaler
            using the `scaler_from_config` function.
        """
        config = {
            "scaler_cls": self.__class__.__name__,
            "scaler_args": [self.feature_indices],
            "scaler_kwargs": {
                "scale": self.scale.tolist(),
                "shift": self.shift.tolist(),
            },
            "params": {},
        }

        return config

    def reconstruct(self, params):
        """Reconstruct the configuration from parameters.

        Parameters
        ----------
        params : dict
            The parameters found in `config["params"]` with the `config`
            being the dictionary returned by `get_config`.
        """
        if "scale" in params:
            self.scale = torch.tensor(params["scale"])
        if "shift" in params:
            self.shift = torch.tensor(params["shift"])

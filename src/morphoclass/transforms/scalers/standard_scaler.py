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
"""Implementation of the standard scaler transform."""
from __future__ import annotations

import torch
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

from morphoclass.transforms.scalers import AbstractFeatureScaler


class FeatureStandardScaler(AbstractFeatureScaler):
    """Scaler that removes the mean and standard deviation.

    Internally the `StandardScaler` from scikit-learn is applied

    Parameters
    ----------
    feature_indices
        List of indices of the feature maps to which to apply the scaling.
    with_mean : bool (optional)
        Whether or not to shift by the mean value. This value is passed through
        to the `StandardScaler` class in sklearn.
    with_std : bool (optional)
        Whether or not to scale by the standard deviation. This value is passed
        through to the `StandardScaler` class in sklearn.
    kwargs
        Additional keyword argument to pass through to the `AbstractFeatureScaler`
        base class.
    """

    def __init__(self, feature_indices, with_mean=True, with_std=True, **kwargs):
        super().__init__(feature_indices, **kwargs)
        self.with_mean = with_mean
        self.with_std = with_std

        self.scaler = StandardScaler(
            with_mean=self.with_mean, with_std=self.with_std, copy=False
        )
        self.mean = None
        self.scale = None

    def _fit(self, features):
        self.scaler.fit(features)
        self.mean = torch.tensor(self.scaler.mean_, dtype=torch.get_default_dtype())
        self.scale = torch.tensor(self.scaler.scale_, dtype=torch.get_default_dtype())

    def _transform(self, features):
        if self.scaler.with_mean:
            if self.mean is None:
                raise NotFittedError("The scaler has to be fitted first")
            features -= self.mean.to(features.dtype)
        if self.scaler.with_std:
            if self.scale is None:
                raise NotFittedError("The scaler has to be fitted first")
            features /= self.scale.to(features.dtype)
        return features
        # return self.scaler.transform(features)

    def get_config(self):
        """Generate the configuration necessary for reconstructing the scaler.

        Returns
        -------
        config : dict
            The configuration of the scaler. It should contain all
            information necessary for reconstructing the scaler
            using the `scaler_from_config` function.
        """
        params = {}
        if self.scale is not None:
            params["scale"] = self.scale.tolist()
        if self.mean is not None:
            params["mean"] = self.mean.tolist()

        config = {
            "scaler_cls": self.__class__.__name__,
            "scaler_args": [self.feature_indices],
            "scaler_kwargs": {"with_mean": self.with_mean, "with_std": self.with_std},
            "params": params,
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
        if "mean" in params:
            self.mean = torch.tensor(params["mean"])

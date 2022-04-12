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
"""Implementation of the robust scaler transform."""
from __future__ import annotations

import torch
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import RobustScaler

from morphoclass.transforms.scalers import AbstractFeatureScaler


class FeatureRobustScaler(AbstractFeatureScaler):
    """Scaler that is robust against outliers in data.

    Internally the `RobustScaler` from scikit-learn is applied.

    Parameters
    ----------
    feature_indices
        List of indices of the feature maps to which to apply the scaling.
    with_centering : bool (optional)
        If True, center the data before scaling. This value is passed through
        to the `RobustScaler` class in sklearn.
    with_scaling : bool (optional)
        If True, scale the data to interquartile range. This value is passed
        through to the `RobustScaler` class in sklearn.
    kwargs
        Additional keyword argument to pass through to the `AbstractFeatureScaler`
        base class.
    """

    def __init__(
        self, feature_indices, with_centering=True, with_scaling=True, **kwargs
    ):
        super().__init__(feature_indices, **kwargs)
        self.with_centering = with_centering
        self.with_scaling = with_scaling

        self.scaler = RobustScaler(
            with_centering=self.with_centering,
            with_scaling=self.with_scaling,
            copy=False,
        )
        self.center = None
        self.scale = None

    def _fit(self, features):
        self.scaler.fit(features)

        if self.scaler.center_ is not None:
            self.center = torch.tensor(
                self.scaler.center_, dtype=torch.get_default_dtype()
            )
        if self.scaler.scale_ is not None:
            self.scale = torch.tensor(
                self.scaler.scale_, dtype=torch.get_default_dtype()
            )

    def _transform(self, features):
        if self.with_centering:
            if self.center is None:
                raise NotFittedError("The scaler has to be fitted first")
            features -= self.center.to(features.dtype)
        if self.with_scaling:
            if self.scale is None:
                raise NotFittedError("The scaler has to be fitted first")
            features /= self.scale.to(features.dtype)

        return features

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
        if self.center is not None:
            params["center"] = self.center.tolist()

        config = {
            "scaler_cls": self.__class__.__name__,
            "scaler_args": [self.feature_indices],
            "scaler_kwargs": {
                "with_centering": self.with_centering,
                "with_scaling": self.with_scaling,
            },
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
        if "center" in params:
            self.center = torch.tensor(params["center"])

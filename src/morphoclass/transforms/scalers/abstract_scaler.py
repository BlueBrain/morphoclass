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
"""The abstract base class for all feature scalers."""
from __future__ import annotations

import abc

from morphoclass.data import MorphologyDataLoader
from morphoclass.transforms.helper import require_field


class AbstractFeatureScaler(abc.ABC):
    """Base abstract class for all feature scalers.

    All derived classes must implement the `_fit` and `_transform` methods

    Parameters
    ----------
    feature_indices
        List of indices of the feature maps to which to apply the scaling.
    is_global_feature
        Apply scaler to global features rather than node features.
    """

    def __init__(self, feature_indices, is_global_feature=False):
        if isinstance(feature_indices, int):
            feature_indices = [feature_indices]
        self.feature_indices = feature_indices
        self.feature_field = "u" if is_global_feature else "x"

    def fit(self, dataset, idx=None):
        """Fit the scaler to data.

        Parameters
        ----------
        dataset : morphoclass.data.MorphologyDataset
            Data to which to fit the scaler.
        idx
            Selects a subset of samples in the dataset to which to fit the data.
        """
        if idx is None:
            idx = list(range(len(dataset)))

        subset = dataset.index_select(idx)
        loader = MorphologyDataLoader(dataset=subset, batch_size=len(idx))
        batch = next(iter(loader))
        all_features = getattr(batch, self.feature_field)
        self._fit(all_features[:, self.feature_indices])

    @abc.abstractmethod
    def _fit(self, features):
        """Fit the scaler to the given features.

        Parameters
        ----------
        features
            The features to which to fit the scaler.
        """

    @require_field("x")
    def __call__(self, data):
        """Apply the fitted scaler to data.

        Parameters
        ----------
        data
            The input morphology data.

        Returns
        -------
        data
            The data with the appropriate features scaled.
        """
        all_features = getattr(data, self.feature_field)
        scaled_values = self._transform(all_features[:, self.feature_indices])
        all_features[:, self.feature_indices] = scaled_values

        return data

    @abc.abstractmethod
    def _transform(self, features):
        """Apply the scaling to the given features.

        Parameters
        ----------
        features
            The features to which to apply the scaling.

        Returns
        -------
        features
            Features with the scaling applied.
        """

    @abc.abstractmethod
    def get_config(self):
        """Generate the configuration necessary for reconstructing the scaler.

        Returns
        -------
        config : dict
            The configuration of the scaler. It should contain all
            information necessary for reconstructing the scaler
            using the `scaler_from_config` function.
        """

    @abc.abstractmethod
    def reconstruct(self, params):
        """Reconstruct the configuration from parameters.

        Parameters
        ----------
        params : dict
            The parameters found in `config["params"]` with the `config`
            being the dictionary returned by `get_config`.
        """

    def __repr__(self):
        """Compute the repr."""
        return f"{self.__class__.__name__}(feature_indices={self.feature_indices})"

"""Implementation of the min-max scaler transform."""
from __future__ import annotations

import torch
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import MinMaxScaler

from morphoclass.transforms.scalers import AbstractFeatureScaler


class FeatureMinMaxScaler(AbstractFeatureScaler):
    """Scaler that scales the features to a given range.

    Internally the `MinMaxScaler` from scikit-learn is applied

    Parameters
    ----------
    feature_indices
        List of indices of the feature maps to which to apply the scaling.
    feature_range : sequence (optional)
        The feature range to which to scale the features. This value is passed
        through to the `MinMaxScaler` class in sklearn.
    take_abs : bool (optional)
        If true then the scaler will be fitted on the absolute values of the
        features. This way a feature range of (0, 1) would translate to an
        effective range of (-1, 1). This is useful when it's necessary to avoid
        shifting the features away from the origin.
    kwargs
        Additional keyword argument to pass through to the `AbstractFeatureScaler`
        base class.
    """

    def __init__(self, feature_indices, feature_range=(0, 1), take_abs=False, **kwargs):
        super().__init__(feature_indices, **kwargs)
        self.feature_range = feature_range
        self.take_abs = take_abs

        self.scaler = MinMaxScaler(feature_range=feature_range, copy=False)
        self.scale = None
        self.min = None

    def _fit(self, features):
        if self.take_abs:
            self.scaler.fit(features.abs())
        else:
            self.scaler.fit(features)

        self.scale = torch.tensor(self.scaler.scale_, dtype=torch.get_default_dtype())
        self.min = torch.tensor(self.scaler.min_, dtype=torch.get_default_dtype())

    def _transform(self, features):
        if self.scale is None or self.min is None:
            raise NotFittedError("The scaler has to be fitted first")
        features *= self.scale.to(features.dtype)
        features += self.min.to(features.dtype)

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
        if self.min is not None:
            params["min"] = self.min.tolist()

        config = {
            "scaler_cls": self.__class__.__name__,
            "scaler_args": [self.feature_indices],
            "scaler_kwargs": {
                "feature_range": self.feature_range,
                "take_abs": self.take_abs,
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
        if "min" in params:
            self.min = torch.tensor(params["min"])

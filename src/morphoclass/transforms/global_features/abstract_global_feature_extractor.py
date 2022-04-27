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
"""Implementation of the abstract base class for global feature extractors."""
from __future__ import annotations

import abc

import torch


class AbstractGlobalFeatureExtractor(abc.ABC):
    """A morphology data transform for global feature extraction."""

    @abc.abstractmethod
    def extract_global_feature(self, data):
        """Extract a particular global feature.

        Child classes should implement this function.

        Parameters
        ----------
        data
            A morphology data sample.

        Returns
        -------
        The value of the global feature for the given data sample.
        """

    def __call__(self, data):
        """Call the transform within the data transform pipeline.

        Parameters
        ----------
        data
            A morphology data sample.

        Returns
        -------
        data
            The processed data sample with the extracted global feature
            attached to it.
        """
        # Extract global feature
        feature_value = self.extract_global_feature(data)

        # Convert it into a tensor
        feature_tensor = torch.tensor([[feature_value]], dtype=torch.float32)

        # Insert it into the `Data` object
        if hasattr(data, "u") and data.u is not None:
            data.u = torch.cat([data.u, feature_tensor], dim=1)
        else:
            data.u = feature_tensor
        return data

    def __repr__(self):
        """Compute the repr."""
        return f"{self.__class__.__name__}()"

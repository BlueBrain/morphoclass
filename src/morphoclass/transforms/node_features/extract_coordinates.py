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
"""Implementation of the 3D coordinate node feature extractor."""
from __future__ import annotations

import numpy as np
import torch

from morphoclass.transforms.helper import require_field


class ExtractCoordinates:
    """Extract coordinate features from neurites trees.

    The data should contain the field `tmd_neurites`, see the
    `ExtractTMDNeurites` class.

    For each apical tree in the neuron the coordinates of all node points
    of the morphology are extracted and added to the feature vector.

    Parameters
    ----------
    shift_to_origin : bool (optional)
        If true then all coordinates will be shifted so that the root of the
        given apical tree is at the origin.
    """

    def __init__(self, shift_to_origin=True):
        self.shift_to_origin = shift_to_origin

    @require_field("tmd_neurites")
    def __call__(self, data):
        """Apply the transform to a given data sample to extract node features.

        Parameters
        ----------
        data
            A morphology data sample.

        Returns
        -------
        data
            The processed data sample with the extracted node features
            attached to it.
        """
        features = torch.empty(size=(0,), dtype=torch.float32)
        for apical in data.tmd_neurites:
            distances = np.transpose([apical.x, apical.y, apical.z])
            if self.shift_to_origin:
                if len(np.nonzero(apical.p == -1)[0]) != 1:
                    msg = "Apical must have only one root"
                    raise ValueError(msg)
                distances = distances - distances[apical.p == -1]
            distances = torch.tensor(distances, dtype=torch.float32)
            features = torch.cat([features, distances])

        if hasattr(data, "x") and data.x is not None:
            data.x = torch.cat([data.x, features], dim=1)
        else:
            data.x = features

        return data

    def __repr__(self):
        """Compute the repr."""
        return f"{self.__class__.__name__}(shift_to_origin={self.shift_to_origin})"

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
"""Implementation of the branching angles node feature extractor."""
from __future__ import annotations

import numpy as np
import torch

from morphoclass.transforms.helper import require_field
from morphoclass.utils import print_warning


class ExtractBranchingAngles:
    """Extract branching angles from neurites trees.

    The data should contain the field `tmd_neurites`, see the
    `ExtractTMDNeurites` class.

    For each apical tree in the neuron the branching angles of all node points
    of the morphology are extracted and added to the feature vector. For
    points without a branching (root, leaf, and intermediate nodes) the
    angle is set a fixed value (default: 0.0)

    Parameters
    ----------
    non_branching_angle : float (optional)
        The default angle value for non-branching nodes.
    """

    def __init__(self, non_branching_angle=0.0):
        self.non_branching_angle = non_branching_angle

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
            # Fill in default values
            angles = torch.full(
                fill_value=self.non_branching_angle,
                size=(len(apical.x), 1),
                dtype=torch.float32,
            )
            # Get indices of all branching points
            branching_points_ids = (apical.dA.sum(axis=0) == 2).nonzero()[1]
            # For each branching point compute the branching angle
            for idx in branching_points_ids:
                c1_idx, c2_idx = (apical.dA[:, idx] == 1).nonzero()[0]
                u = np.array([apical.x[idx], apical.y[idx], apical.z[idx]])
                v1 = (
                    np.array([apical.x[c1_idx], apical.y[c1_idx], apical.z[c1_idx]]) - u
                )
                v2 = (
                    np.array([apical.x[c2_idx], apical.y[c2_idx], apical.z[c2_idx]]) - u
                )
                norm = np.linalg.norm(v1) * np.linalg.norm(v2)
                if norm == 0:
                    print_warning(
                        "Zero branching angle found. "
                        "Duplicate points at section start/end?"
                    )
                    angle = 0.0
                else:
                    angle = np.arccos(np.vdot(v1, v2) / norm)
                angles[idx] = torch.tensor(angle)
            features = torch.cat([features, angles])

        if hasattr(data, "x") and data.x is not None:
            data.x = torch.cat([data.x, features], dim=1)
        else:
            data.x = features

        return data

    def __repr__(self):
        """Compute the repr."""
        return (
            f"{self.__class__.__name__}"
            f"(non_branching_angle={self.non_branching_angle})"
        )

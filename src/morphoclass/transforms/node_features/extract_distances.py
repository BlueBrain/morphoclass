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
"""The base class for node distance feature extractors."""
from __future__ import annotations

import torch

from morphoclass.transforms.helper import require_field


class ExtractDistances:
    """Extract some distance features from apical trees.

    The data should contain the field `tmd_neuron`, see the
    `ExtractTMDNeuron` class.

    This is an abstract class. Sub-classes of this class should implement
    `get_distance(apical)` to implement concrete distance extraction from
    the apical tree.

    Parameters
    ----------
    vertical_axis : str
        The name of the axis along which the neurons are vertically oriented.
        In other words, the axis that is perpendicular to the cortex surface.
        This is only relevant for BPCs and is only considered if the parameter
        `negative_bpcs` is true, in which case the projection of the two apicals
        onto this axis decides which one of the two is lower than the other one.
    negative_ipcs : bool
        If true then the distance features of all cells labelled as IPCs will
        have their sign inverted (feature => -feature).
    negative_bpcs : bool
        If true then all cells labelled as BPCs will have their lower apical's
        distance features inverted by flipping their sign. (feature => -feature).
        To determine which of the two apicals is lower the node positions are
        projected onto the axis specified in `vertical_axis`.
    """

    def __init__(self, vertical_axis="y", negative_ipcs=False, negative_bpcs=False):
        if vertical_axis not in ["x", "y", "z"]:
            msg = "vertical_axis must be one of (x, y, z)"
            raise ValueError(msg)
        self.vertical_axis = vertical_axis
        self.negative_ipcs = negative_ipcs
        self.negative_bpcs = negative_bpcs

    def get_distances(self, apical):
        """Extract the node distance feature from an apical tree.

        Parameters
        ----------
        apical : tmd.Tree.Tree
            An apical tree.

        Returns
        -------
        torch.tensor
            A torch tensor of shape (n_nodes,) and of dtype float32 with the
            extracted node distance features.
        """
        raise NotImplementedError

    @require_field("tmd_neurites")
    def __call__(self, data):
        """Apply the transform to a given data sample.

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
        features_list = [self.get_distances(apical) for apical in data.tmd_neurites]

        # Special treatment for IPCs and BPCs
        if self.negative_bpcs and "BPC" in data.y_str:
            assert len(features_list) == 2, "BPC, but not two apicals!"
            # assume upper apical has higher mean vertical coordinate
            # argmin without numpy...
            idx_lower_apical = min(
                range(len(features_list)),
                key=lambda i: float(
                    getattr(data.tmd_neurites[i], self.vertical_axis).mean()
                ),
            )
            features_list[idx_lower_apical] = -features_list[idx_lower_apical]
        elif self.negative_ipcs and "IPC" in data.y_str:
            features_list = [-dist for dist in features_list]

        if len(features_list) == 0:
            features_list = [torch.empty(size=(0,), dtype=torch.float32)]
        features = torch.cat(features_list).unsqueeze(dim=1)

        if hasattr(data, "x") and data.x is not None:
            data.x = torch.cat([data.x, features], dim=1)
        else:
            data.x = features

        return data

    def __repr__(self):
        """Compute the repr."""
        return f"{self.__class__.__name__}(vertical_axis={self.vertical_axis})"

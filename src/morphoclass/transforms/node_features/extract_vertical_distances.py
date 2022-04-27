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
"""Implementation of the vertical distance node feature extractor."""
from __future__ import annotations

import numpy as np
import torch

from morphoclass.transforms.node_features.extract_distances import ExtractDistances


class ExtractVerticalDistances(ExtractDistances):
    """Extract vertical distance features from apical trees.

    The data should contain the field `tmd_neuron`, see the
    `ExtractTMDNeuron` class.

    For each apical tree in the neuron the vertical distance of all node points
    of the morphology are extracted and added to the feature vector.

    The vertical distance of a given node is the distance to the root point of
    the tree that is measured by following the graph edges, and projecting onto
    a given axis. The default axis is 'y'.

    Parameters
    ----------
    vertical_axis : str
        Axis along which the neuron is assumed to be oriented vertically. See
        documentation of the `ExtractDistances` base class for more details.
    negative_ipcs : bool
        Invert sign for features of IPCs. ee documentation of the
        `ExtractDistances` base class for more details.
    negative_bpcs : bool
        Invert sign for features of lower apical trees of BPCs. ee documentation
        of the `ExtractDistances` base class for more details.
    """

    def __init__(self, vertical_axis="y", negative_ipcs=False, negative_bpcs=False):
        super().__init__(
            vertical_axis=vertical_axis,
            negative_ipcs=negative_ipcs,
            negative_bpcs=negative_bpcs,
        )

    def get_distances(self, apical):
        """Extract the node distance feature from an apical tree.

        This is an implementation of the base class's abstract method.

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
        if len(np.nonzero(apical.p == -1)[0]) != 1:
            msg = "Apical must have only one root"
            raise ValueError(msg)
        distances = getattr(apical, self.vertical_axis)
        distances = distances - distances[apical.p == -1]
        distances = torch.tensor(distances, dtype=torch.float32)

        return distances

    def __repr__(self):
        """Compute the repr."""
        return (
            f"{self.__class__.__name__}("
            f"vertical_axis={self.vertical_axis}, "
            f"negative_ipcs={self.negative_ipcs}, "
            f"negative_bpcs={self.negative_bpcs})"
        )

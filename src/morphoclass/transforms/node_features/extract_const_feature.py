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
"""Implementation of the node feature extractor for constant node features."""
from __future__ import annotations

import torch

from morphoclass.transforms.helper import require_field
from morphoclass.transforms.node_features.extract_node_features import (
    ExtractNodeFeatures,
)


class ExtractConstFeature(ExtractNodeFeatures):
    """Extractor that assigns the constant feature value 1.0 to each node."""

    @require_field("tmd_neurites")
    def extract_node_features(self, data):
        """Extract the constant node feature value 1.0 for each node.

        Parameters
        ----------
        data
            A data sample

        Returns
        -------
        features
            Torch tensor of shape (n_nodes,)
        """
        ones = [torch.ones(apical.d.shape) for apical in data.tmd_neurites]
        return torch.cat(ones)

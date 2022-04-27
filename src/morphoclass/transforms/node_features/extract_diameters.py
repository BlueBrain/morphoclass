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
"""Implementation of the branch diameter node feature extractor."""
from __future__ import annotations

import torch

from morphoclass.transforms.helper import require_field
from morphoclass.transforms.node_features.extract_node_features import (
    ExtractNodeFeatures,
)


class ExtractDiameters(ExtractNodeFeatures):
    """Extract the diameter value attached to each node."""

    @require_field("tmd_neurites")
    def extract_node_features(self, data):
        """Extract the diameter node features from data.

        Parameters
        ----------
        data
            A data sample

        Returns
        -------
        features
            Torch tensor of shape (n_nodes,)
        """
        diameters = [torch.from_numpy(apical.d) for apical in data.tmd_neurites]
        return torch.cat(diameters)

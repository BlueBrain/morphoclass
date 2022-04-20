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
"""Implementation of the `ExtractNumberLeaves` global feature extractor."""
from __future__ import annotations

import numpy as np

from morphoclass.transforms.global_features.abstract_global_feature_extractor import (
    AbstractGlobalFeatureExtractor,
)
from morphoclass.transforms.helper import require_field


class ExtractNumberLeaves(AbstractGlobalFeatureExtractor):
    """Extract number of leaf nodes.

    The data should contain the field `tmd_neurites`, see the
    `ExtractTMDNeuritess` class.
    """

    @require_field("tmd_neurites")
    def extract_global_feature(self, data):
        """Extract the number of leaf nodes of the morphology.

        Parameters
        ----------
        data
            A morphology data sample.

        Returns
        -------
        The total number of leaf nodes of the morphology.
        """
        return sum(
            np.sum(apical.dA.sum(axis=0).A1 == 0) for apical in data.tmd_neurites
        )

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
"""Implementation of the `TotalPathLength` global feature extractor."""
from __future__ import annotations

from queue import Queue

import neurom as nm
import numpy as np
from neurom.core.types import NeuriteType

from morphoclass.transforms.global_features.abstract_global_feature_extractor import (
    AbstractGlobalFeatureExtractor,
)


class TotalPathLength(AbstractGlobalFeatureExtractor):
    """Extract total path length of the apical.

    The data should contain the field `tmd_neurites`, see the
    `ExtractTMDNeurites` class.
    """

    def __init__(self, tree_hash_fn=None, from_morphology=False):
        super().__init__()
        self.from_morphology = from_morphology
        self.tree_hash_fn = tree_hash_fn
        self.cache = {}

    def extract_global_feature(self, data):
        """Extract the total path length global feature from data sample.

        Parameters
        ----------
        data
            A morphology data sample.

        Returns
        -------
        The total path length of the morphology.
        """
        if self.from_morphology:
            neuron = data.morphology
            total_path_length = nm.get(
                "total_length", neuron, neurite_type=NeuriteType.apical_dendrite
            ).sum()
        else:
            total_path_length = 0
            for apical in data.tmd_neurites:
                total_path_length += self._get_total_path_length(apical)

        return total_path_length

    def _get_total_path_length(self, tree):
        # Check cache
        if self.tree_hash_fn is not None:
            tree_hash = self.tree_hash_fn(tree)
            if tree_hash in self.cache:
                return self.cache[tree_hash]

        # Compute total path length
        coords = np.transpose([tree.x, tree.y, tree.z])
        result = 0
        q: Queue = Queue()
        for idx in np.where(tree.p == -1)[0]:
            q.put(idx)
        while not q.empty():
            idx = q.get()
            for idx_next in tree.dA[:, idx].nonzero()[0]:
                result += np.linalg.norm(coords[idx] - coords[idx_next])
                q.put(idx_next)

        # Store in cache
        if self.tree_hash_fn is not None:
            tree_hash = self.tree_hash_fn(tree)
            self.cache[tree_hash] = result

        return result

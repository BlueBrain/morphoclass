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
"""Implementation of the `ExtractMaximalApicalPathLength` global feature extractor."""
from __future__ import annotations

import numpy as np

from morphoclass.transforms.global_features.abstract_global_feature_extractor import (
    AbstractGlobalFeatureExtractor,
)
from morphoclass.transforms.helper import require_field


class ExtractMaximalApicalPathLength(AbstractGlobalFeatureExtractor):
    """Extract maximal neurite length.

    Maximal path distance form soma to a tip (leaf node)

    The data should contain the field `tmd_neurites`, see the
    `ExtractTMDNeurites` class.
    """

    def __init__(self, tree_hash_fn=None):
        super().__init__()
        self.tree_hash_fn = tree_hash_fn
        self.tree_hashes = {}

    @require_field("tmd_neurites")
    def extract_global_feature(self, data):
        """Extract the maximal apical length global feature from data.

        Parameters
        ----------
        data
            The input morphology.

        Returns
        -------
        The maximal apical length of the morphology.
        """
        leaf_distances = []
        for apical in data.tmd_neurites:
            leaf_distances += self._get_leaf_path_distances(apical)
        return max(leaf_distances)

    def _get_leaf_path_distances(self, tree):
        if self.tree_hash_fn is not None:
            tree_hash = self.tree_hash_fn(tree)
            if tree_hash in self.tree_hashes:
                return self.tree_hashes[tree_hash]

        leaf_node_mask = tree.dA.sum(axis=0).A1 == 0
        (leaf_node_ids,) = np.where(leaf_node_mask)
        coords = np.transpose([tree.x, tree.y, tree.z])

        results = [0.0 for idx in leaf_node_ids]
        for i, idx in enumerate(leaf_node_ids):
            while tree.p[idx] >= 0:
                distance = np.linalg.norm(coords[idx] - coords[tree.p[idx]])
                results[i] += distance
                idx = tree.p[idx]

        if self.tree_hash_fn is not None:
            tree_hash = self.tree_hash_fn(tree)
            self.tree_hashes[tree_hash] = results

        return results

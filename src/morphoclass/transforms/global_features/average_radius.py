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
"""Implementation of the `AverageRadius` global feature extractor."""
from __future__ import annotations

from queue import Queue

import neurom as nm
import numpy as np
from neurom import COLS
from neurom.core.types import NeuriteType

from morphoclass.transforms.global_features.abstract_global_feature_extractor import (
    AbstractGlobalFeatureExtractor,
)
from morphoclass.transforms.global_features.total_path_length import TotalPathLength


class AverageRadius(AbstractGlobalFeatureExtractor):
    """Extract average radius across all apicals.

    The data should contain the field `tmd_neurites`, see the
    `ExtractTMDNeurites` class.
    """

    def __init__(self, tree_hash_fn=None, from_morphology=False):
        super().__init__()
        self.from_morphology = from_morphology
        self.tree_hash_fn = tree_hash_fn
        self.cache = {}
        self.total_path_length_transform = TotalPathLength(
            self.tree_hash_fn,
            self.from_morphology,
        )

    def extract_global_feature(self, data):
        """Extract the average radius global feature from data.

        Parameters
        ----------
        data
            The input morphology.

        Returns
        -------
        The average radius of the morphology.
        """
        total_path_length = self.total_path_length_transform.extract_global_feature(
            data
        )
        integrated_radii = 0
        if self.from_morphology:
            neuron = data.morphology

            def integrated_radius(p1, p2):
                r = (p1[COLS.R] + p2[COLS.R]) / 2
                d = np.linalg.norm(p1[COLS.XYZ] - p2[COLS.XYZ])
                return r * d

            filter = nm.core.types.tree_type_checker(NeuriteType.apical_dendrite)
            integrated_radii = sum(
                integrated_radius(p1, p2)
                for p1, p2 in nm.iter_segments(neuron, neurite_filter=filter)
            )
        else:
            for apical in data.tmd_neurites:
                integrated_radii += self._get_integrated_diameters(apical) / 2

        return integrated_radii / total_path_length

    def _get_integrated_diameters(self, tree):
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
                length = np.linalg.norm(coords[idx] - coords[idx_next])
                avg_diameter = (tree.d[idx] + tree.d[idx_next]) / 2
                result += length * avg_diameter
                q.put(idx_next)

        # Store in cache
        if self.tree_hash_fn is not None:
            tree_hash = self.tree_hash_fn(tree)
            self.cache[tree_hash] = result

        return result

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
"""Implementation of the `AverageBranchOrder` global feature extractor."""
from __future__ import annotations

import numpy as np

from morphoclass.transforms.global_features.abstract_global_feature_extractor import (
    AbstractGlobalFeatureExtractor,
)
from morphoclass.transforms.helper import require_field


class AverageBranchOrder(AbstractGlobalFeatureExtractor):
    """Extract average branch order.

    Branch order is defined as the number of branching points
    one passes going from the leafs to the root. Thus the root
    point has branching order 0

    The data should contain the field `tmd_neurites`, see the
    `ExtractTMDNeurites` class.
    """

    def __init__(self, tree_hash_fn=None):
        super().__init__()
        self.tree_hash_fn = tree_hash_fn
        self.cache = {}

    @require_field("tmd_neurites")
    def extract_global_feature(self, data):
        """Extract the average branch order global feature from data sample.

        Parameters
        ----------
        data
            A morphology data sample.

        Returns
        -------
        float
            The average branch order of the morpholgy.
        """
        avg_branch_orders = []
        for apical in data.tmd_neurites:
            avg_branch_orders.append(self._get_avg_branch_order(apical))

        return np.mean(avg_branch_orders)

    def _get_avg_branch_order(self, tree):
        # Check cache
        if self.tree_hash_fn is not None:
            tree_hash = self.tree_hash_fn(tree)
            if tree_hash in self.cache:
                return self.cache[tree_hash]

        # Compute average radius - sum up average radii of all edges
        result = []
        n_children = tree.dA.sum(axis=0).A1
        ids_leaf_nodes = np.where(n_children == 0)[0]
        for idx in ids_leaf_nodes:
            branching_order = 0
            while idx >= 0:
                if n_children[idx] > 1:
                    branching_order += 1
                idx = tree.p[idx]
            result.append(branching_order)
        result = np.mean(result)

        # Store in cache
        if self.tree_hash_fn is not None:
            tree_hash = self.tree_hash_fn(tree)
            self.cache[tree_hash] = result

        return result

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
"""Implementation of the `ApplyNodeReductionMasks` transform."""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from morphoclass.transforms.helper import require_field


class ApplyNodeReductionMasks:
    """Transform that applies the apical node masks to the apicals."""

    @staticmethod
    def apply_mask(apical, mask):
        """Apply a node mask to the given apical.

        Parameters
        ----------
        apical : tmd.Tree.Tree
            The apical dendrite tree.
        mask : np.ndarray
            The node mask.
        """
        apical.x = apical.x[mask]
        apical.y = apical.y[mask]
        apical.z = apical.z[mask]
        apical.d = apical.d[mask]
        apical.t = apical.t[mask]

        # Reconnect the tree so that deleted nodes are skipped
        for i, parent in enumerate(apical.p):
            if not mask[i]:
                continue
            while not mask[parent] and parent != -1:
                parent = apical.p[parent]
            apical.p[i] = parent

        # Update the parent indices to the new position of the nodes
        old_ids = mask.nonzero()[0]
        map_idx = {idx: i for i, idx in enumerate(old_ids)}
        map_idx[-1] = -1
        apical.p = np.array([map_idx[idx] for idx in apical.p[mask]])

        # Construct the adjacency matrix
        indices = np.transpose([[i, p] for i, p in enumerate(apical.p) if p != -1])
        values = np.ones(indices.shape[-1])
        size = len(apical.p)
        apical.dA = csr_matrix((values, indices), shape=(size, size))

    @require_field("tmd_neurites")
    @require_field("tmd_neurites_masks")
    def __call__(self, data):
        """Apply the morphology transformation.

        Parameters
        ----------
        data
            The input morphology data sample.

        Returns
        -------
        data
            The modified morphology data sample.
        """
        # Apply mask to apicals
        for apical, mask in zip(data.tmd_neurites, data.tmd_neurites_masks):
            self.apply_mask(apical, mask)

        # Apply mask to features x
        mask = np.concatenate(data.tmd_neurites_masks)
        if hasattr(data, "x") and data.x is not None:
            data.x = data.x[mask]

        return data

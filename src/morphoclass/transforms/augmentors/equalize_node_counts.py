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
"""Implementation of the `EqualizeNodeCounts` transform."""
from __future__ import annotations

import numpy as np
from sklearn.exceptions import NotFittedError

from morphoclass.transforms.helper import raise_no_attribute
from morphoclass.transforms.helper import require_field


class EqualizeNodeCounts:
    """Equalize node counts across the whole dataset.

    If possible, nodes from the original apicals are added to the apicals
    as long as the node count in the sample is smaller than the threshold
    specified by the field `min_total_nodes`.

    The threshold `min_total_nodes` can be either set manually or found
    automatically by fitting the transformer to a dataset, in which case
    the value is set to the minimal number of nodes in the original apicals
    across the whole dataset.

    Parameters
    ----------
    min_total_nodes : int (optional)
        The threshold for the new node count.
    """

    def __init__(self, min_total_nodes=None):
        self.min_total_nodes = min_total_nodes

    def fit(self, dataset):
        """Find a heuristic number of nodes to which to equalize the samples.

        This is simply the minimal number of apical nodes across the whole
        dataset. This way we make sure that each sample can reach this
        number of nodes.

        Parameters
        ----------
        dataset
            The dataset to which to fit.
        """
        self.min_total_nodes = float("inf")
        for data in dataset:
            if not hasattr(data, "tmd_neurites"):
                raise_no_attribute("tmd_neurites")
            n = sum(len(apical.p) for apical in data.tmd_neurites)
            self.min_total_nodes = min(self.min_total_nodes, n)

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
        if self.min_total_nodes is None:
            msg = (
                "Either fit the transformer first or manually "
                "set the min_total_nodes attribute."
            )
            raise NotFittedError(msg)

        still_to_add = self.min_total_nodes
        still_to_add -= sum(mask.sum() for mask in data.tmd_neurites_masks)
        # It might be that a sample has multiple masks and some of the masks
        # have already exhausted all their points. In that case we don't
        # want to consider these apicals and exclude them from random sampling.
        good_apical = np.ones(len(data.tmd_neurites), dtype=bool)

        while still_to_add > 0 and any(good_apical):
            idx_apical = np.random.choice(np.where(good_apical)[0])
            mask = data.tmd_neurites_masks[idx_apical]
            idx_new_point = np.random.choice(np.where(~mask)[0])
            mask[idx_new_point] = True
            still_to_add -= 1
            if all(mask):  # all points in this apical have been added
                good_apical[idx_apical] = False

        return data

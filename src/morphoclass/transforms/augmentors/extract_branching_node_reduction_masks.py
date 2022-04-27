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
"""Implementation of the `ExtractBranchingNodeReductionMasks` transform."""
from __future__ import annotations

from morphoclass.transforms.helper import require_field


class ExtractBranchingNodeReductionMasks:
    """Transform that extracts apical node masks for branching nodes."""

    @staticmethod
    def _get_branching_node_mask(apical):
        # Need to make sure the leaf and root nodes are included as well
        one_child = (apical.dA.sum(axis=0) == 1).A1
        one_parent = (apical.dA.sum(axis=1) == 1).A1
        intermediate = one_child & one_parent
        return ~intermediate

    @require_field("tmd_neurites")
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
        data.tmd_neurites_masks = [
            self._get_branching_node_mask(apical) for apical in data.tmd_neurites
        ]
        return data

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

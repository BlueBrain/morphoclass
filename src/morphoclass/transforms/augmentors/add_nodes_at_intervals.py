"""Implementation of the `AddNodesAtIntervals` transform."""
from __future__ import annotations

from itertools import islice

import numpy as np

from morphoclass.transforms.helper import require_field


class AddNodesAtIntervals:
    """Add points to reduced tree at given interval distance.

    The reduced apical tree specified by the point mask, is enhanced
    by adding additional points from the original apical tree.

    First it is checked that all branching points are set. Then all points
    are added that are further away from those that are already set
    by the given interval distance.

    Parameters
    ----------
    interval : float
        Threshold distance at which a new point is added.
    """

    def __init__(self, interval):
        self.interval = interval

    def add_nodes(self, apical, mask):
        """Add nodes to the mask according to the interval configuration.

        Parameters
        ----------
        apical : tmd.Tree.Tree.Tree
            An apical tree.
        mask : np.ndarray
            The mask to set. The data in this array will be modified in-place.
        """
        # Iterate over all sections in tree
        for beg, end in zip(*apical.get_sections_2()):
            # Make sure the branching points are definitely in
            mask[[beg, end]] = 1

            # Find all points on current section
            pt_ids = [end]
            while pt_ids[-1] != beg:
                pt_ids.append(apical.p[pt_ids[-1]])
            pt_ids = np.array(pt_ids)

            # Find distances between neighbouring points
            pts = np.transpose([apical.x[pt_ids], apical.y[pt_ids], apical.z[pt_ids]])

            pt_dist = np.linalg.norm(pts[:-1] - pts[1:], axis=1)

            # Add points if the distance to the previous point > interval
            current_d = 0.0
            for i, idx in islice(enumerate(pt_ids), 1, None):
                if mask[idx]:
                    # Point already set
                    current_d = 0.0
                    continue
                current_d += pt_dist[i]
                if current_d > self.interval:
                    mask[idx] = True
                    current_d = 0.0

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
        for apical, mask in zip(data.tmd_neurites, data.tmd_neurites_masks):
            self.add_nodes(apical, mask)
        return data

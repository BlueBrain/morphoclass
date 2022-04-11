"""Implementation of the `AddSectionMiddlePoints` transform."""
from __future__ import annotations

import numpy as np

from morphoclass.transforms.helper import require_field


class AddSectionMiddlePoints:
    """Transform that splits apical sections into two by adding a middle point.

    Parameters
    ----------
    n_points_to_add : int
        The number of sections to split into two by adding middle points.
    """

    def __init__(self, n_points_to_add):
        self.n_points_to_add = n_points_to_add

    @staticmethod
    def _add_random_section_middle_point(apical):
        n = len(apical.p)
        idx_parent = -1
        idx_child = -1
        while idx_parent < 0:  # make sure child is not root
            idx_child = np.random.randint(0, n)
            idx_parent = apical.p[idx_child]

        apical.x = np.append(apical.x, apical.x[[idx_child, idx_parent]].mean())
        apical.y = np.append(apical.y, apical.y[[idx_child, idx_parent]].mean())
        apical.z = np.append(apical.z, apical.z[[idx_child, idx_parent]].mean())
        apical.d = np.append(apical.d, apical.d[[idx_child, idx_parent]].mean())
        apical.t = np.append(apical.t, apical.t[idx_child])
        apical.p = np.append(apical.p, idx_parent)
        apical.p[idx_child] = n
        apical.dA.resize((n + 1), (n + 1))
        apical.dA[idx_child] = n
        apical.dA[n] = idx_parent

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
        for _ in range(self.n_points_to_add):
            apical = np.random.choice(data.tmd_neurites)

            # Changing single values from/to zero in a CSR matrix is expensive,
            # Therefore we convert it to the LIL format before adding the points
            # and then back to CSR after.
            apical.dA = apical.dA.tolil()
            self._add_random_section_middle_point(apical)
            apical.dA = apical.dA.tocsr()

        return data

    def __repr__(self):
        """Compute the repr."""
        return f"{self.__class__.__name__}()"

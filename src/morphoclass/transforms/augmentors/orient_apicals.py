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
"""Implementation of the `OrientApicals` transform."""
from __future__ import annotations

import logging
import pathlib

import numpy as np

from morphoclass.orientation import fit_tree_ray
from morphoclass.transforms.helper import require_field

logger = logging.getLogger(__name__)


class OrientApicals:
    """Orient all apicals along the positive y-axis.

    Parameters
    ----------
    special_treatment_ipcs : bool
        If True, then IPCs (inverted pyramidal cells) will be oriented
        along the negative y-axis
    special_treatment_hpcs : bool
        If True, then HPCs (horizontal pyramidal cells) will be oriented
        along the positive x-axis
    """

    def __init__(self, special_treatment_ipcs=False, special_treatment_hpcs=False):
        self.special_treatment_ipcs = special_treatment_ipcs
        self.special_treatment_hpcs = special_treatment_hpcs

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
        if len(data.tmd_neurites) == 0:
            logger.warning("no neurites found, no orienting done.")
            return data
        elif len(data.tmd_neurites) > 1:
            logger.warning(
                "than one neurite found - using only the first one for orientation"
            )

        theta, phi = fit_tree_ray(data.tmd_neurites[0])

        alpha = np.pi / 2 - phi
        beta = -(np.pi / 2 - theta)

        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)

        # This is equal to R_x(beta).R_z(alpha)
        rot_matrix = np.array(
            [[ca, -sa, 0], [cb * sa, ca * cb, -sb], [sa * sb, ca * sb, cb]]
        )

        # Special treatment for IPCs and HPCs. The m-type of a cell can be
        # inferred either from the `file` or from the `y_str` attribute
        if self.special_treatment_ipcs or self.special_treatment_hpcs:
            if hasattr(data, "y_str"):
                m_type = data.y_str
            elif hasattr(data, "path"):
                m_type = pathlib.Path(data.path).parent.stem
            else:
                m_type = None
                logger.warning(
                    "Special treatment for IPCs or HPCs was requested, "
                    "but the m-type cannot be determined. "
                    'Make sure that either the "path" or the "y_str" '
                    "attributes is set."
                )
            if m_type is not None:
                if self.special_treatment_ipcs and "IPC" in m_type.upper():
                    rot_matrix = (
                        np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ rot_matrix
                    )
                if self.special_treatment_hpcs and "HPC" in m_type.upper():
                    rot_matrix = (
                        np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]) @ rot_matrix
                    )

        for tree in data.tmd_neurites:
            points = np.array([tree.x, tree.y, tree.z])
            tree.x, tree.y, tree.z = rot_matrix @ points

        return data

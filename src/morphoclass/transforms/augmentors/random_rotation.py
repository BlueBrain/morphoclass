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
"""Implementation of the `RandomRotation` transform."""
from __future__ import annotations

import random

import numpy as np

from morphoclass.transforms.helper import require_field


class RandomRotation:
    """Randomly rotate the apical in its 3D embedding space.

    The full rotations are parametrized by three Euler angles alpha, beta, and
    gamma in the following way:

        v -> R_y(gamma).R_x(beta).R_y(alpha).v

    With this parametrization the range of the angles is the following:

        alpha: [0, 2 * pi]
        beta:  [0, pi]
        gamma: [0, 2 * pi]

    It is possible to restrict the rotations to the y-rotations only by setting
    beta = gamma = 0. This is useful if the apicals are oriented along the
    y-axis.

    Parameters
    ----------
    only_y_rotation : bool, optional
        If true only rotate along the y-axis.
    """

    def __init__(self, only_y_rotation=False):
        self.only_y_rotation = only_y_rotation

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
        alpha = random.uniform(0, 2 * np.pi)
        beta = 0 if self.only_y_rotation else random.uniform(0, np.pi)
        gamma = 0 if self.only_y_rotation else random.uniform(0, 2 * np.pi)

        # Result of R_y(gamma).R_x(beta).R_y(alpha)
        c, s = np.cos, np.sin
        rot = np.array(
            [
                [
                    c(gamma) * c(alpha) + c(beta) * s(gamma) * s(alpha),
                    -s(gamma) * s(beta),
                    c(alpha) * c(beta) * s(gamma) - c(gamma) * s(alpha),
                ],
                [s(alpha) * s(beta), c(beta), c(alpha) * s(beta)],
                [
                    c(gamma) * c(beta) * s(alpha) - c(alpha) * s(gamma),
                    -c(gamma) * s(beta),
                    c(gamma) * c(alpha) * c(beta) + s(gamma) * s(alpha),
                ],
            ]
        )

        for apical in data.tmd_neurites:
            apical.x, apical.y, apical.z = rot @ np.vstack(
                [apical.x, apical.y, apical.z]
            )

        return data

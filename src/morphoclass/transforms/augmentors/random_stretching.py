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
"""Implementation of the `RandomStretching` transform."""
from __future__ import annotations

import random

from morphoclass.transforms.helper import require_field


class RandomStretching:
    """Randomly scale coordinates along all axes.

    Parameters
    ----------
    d_scale_x : float
        The range for the random scale along the x-axis. The generated random
        scale will be in the range `(1 - d_scale_x, 1 + d_scale_x)`.
    d_scale_y : float
        The range for the random scale along the y-axis. The generated random
        scale will be in the range `(1 - d_scale_y, 1 + d_scale_y)`.
    d_scale_z : float
        The range for the random scale along the z-axis. The generated random
        scale will be in the range `(1 - d_scale_z, 1 + d_scale_z)`.
    """

    def __init__(self, d_scale_x=0.0, d_scale_y=0.0, d_scale_z=0.0):
        self.d_scale_x = d_scale_x
        self.d_scale_y = d_scale_y
        self.d_scale_z = d_scale_z

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
        scale_x = random.uniform(1 - self.d_scale_x, 1 + self.d_scale_x)
        scale_y = random.uniform(1 - self.d_scale_y, 1 + self.d_scale_y)
        scale_z = random.uniform(1 - self.d_scale_z, 1 + self.d_scale_z)

        for apical in data.tmd_neurites:
            apical.x *= scale_x
            apical.y *= scale_y
            apical.z *= scale_z

        return data

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
"""Implementation of the `RandomJitter` transform."""
from __future__ import annotations

import numpy as np

from morphoclass.transforms.helper import require_field


class RandomJitter:
    """Randomly jitter all nodes.

    The jittering is applied in the following way. For each coordinate x
    draw two random numbers, d1 in {-d_add, +d_add} and d2 in
    {-d_scale, +d_scale} and transform x as follows:

        x => (x + d1) * (1 + d2)

    Note that d_scale is the scale variation relative to 1.0.

    Parameters
    ----------
    d_add : float
        The maximal additive factor for jittering.
    d_scale : float
        The maximal multiplicative factor for jittering (relative to 1.0).
    shift_to_origin : bool (optional)
        If true then make sure that the root point is at the origin
        before and after the transformation.
    """

    def __init__(self, d_add=0.0, d_scale=0.0, shift_to_origin=True):
        self.d_add = d_add
        self.d_scale = d_scale
        self.shift_to_origin = shift_to_origin

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
        for apical in data.tmd_neurites:
            # Number of nodes
            n = len(apical.p)

            # Generate random shifts and scales
            random_shift = 2 * self.d_add * np.random.rand(3, n) - self.d_add
            random_scale = 2 * self.d_scale * np.random.rand(3, n) - self.d_scale

            # Extract node coordinates
            coords = np.array([apical.x, apical.y, apical.z])

            # Shift to origin
            if self.shift_to_origin:
                if len(np.nonzero(apical.p == -1)[0]) != 1:
                    msg = "Apical must have only one root"
                    raise ValueError(msg)
                coords = coords - coords[:, apical.p == -1]

            # Apply jittering
            coords += random_shift
            coords *= 1.0 + random_scale

            # Shift to origin again
            if self.shift_to_origin:
                coords = coords - coords[:, apical.p == -1]

            # Update apical
            apical.x, apical.y, apical.z = coords

        return data

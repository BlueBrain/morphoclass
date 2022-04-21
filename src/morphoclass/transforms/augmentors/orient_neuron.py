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
"""Implementation of the `OrientNeuron` transform."""
from __future__ import annotations

from morphoclass.orientation import fit_tree_ray
from morphoclass.orientation import orient_neuron
from morphoclass.transforms.helper import require_field


class OrientNeuron:
    """Orient neuron using ray fitting."""

    @require_field("tmd_neuron")
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
        data.tmd_neuron = orient_neuron(fit_tree_ray, data.tmd_neuron, in_place=True)

        return data

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
"""Implementation of the `BranchingOnlyNeuron` transform."""
from __future__ import annotations

from morphoclass.transforms.helper import require_field


class BranchingOnlyNeuron:
    """Extract simplified structure from TMD neurons.

    All neurites in the original data or reduced to branching points only
    """

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
        neuron = data.tmd_neuron
        neuron.apical = [tree.extract_simplified() for tree in neuron.apical]
        neuron.axon = [tree.extract_simplified() for tree in neuron.axon]
        neuron.basal = [tree.extract_simplified() for tree in neuron.basal]
        neuron.undefined = [tree.extract_simplified() for tree in neuron.undefined]

        return data

    def __repr__(self):
        """Compute the repr."""
        return f"{self.__class__.__name__}()"

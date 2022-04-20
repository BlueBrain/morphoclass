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
"""Branching only using the neurites."""
from __future__ import annotations

from morphoclass.transforms.helper import require_field


class BranchingOnlyNeurites:
    """Extract simplified structure from TMD neurites.

    All neurites in the original data or reduced to branching points only
    """

    @require_field("tmd_neurites")
    def __call__(self, data):
        """Callable for TMD Branching only neurites.

        Parameters
        ----------
        data : torch_geometric.data.data.Data
            Data instance.

        Returns
        -------
        data : torch_geometric.data.data.Data
            Processed data instance.
        """
        data.tmd_neurites = [
            neurite.extract_simplified() for neurite in data.tmd_neurites
        ]

        return data

    def __repr__(self):
        """Representation of the BranchingOnlyNeurites class."""
        return f"{self.__class__.__name__}()"

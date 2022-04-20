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
"""Various constants and definitions."""
from __future__ import annotations

from enum import Enum


class DatasetType(Enum):
    """Dataset types supported."""

    pyramidal = "pyramidal-cells"
    interneurons = "interneurons"

    @classmethod
    def _missing_(cls, dataset_name):
        """Handle dataset with layers as well.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset. E.g., pyramidal_cells, pyramidal_cells/L2, etc.

        Returns
        -------
        DatasetType
            The type of dataset, no matter the layer.

        Raises
        ------
        ValueError
            If dataset is not found.
        """
        name = dataset_name.rpartition("/")[0]
        if name.startswith(DatasetType.pyramidal.value):
            return DatasetType.pyramidal
        elif name.startswith(DatasetType.interneurons.value):
            return DatasetType.interneurons
        else:
            raise ValueError(f"Dataset {name} not found.")

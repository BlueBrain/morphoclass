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
"""Implementation of the MorphologyData class."""
from __future__ import annotations

import os
from typing import TypeVar

import torch
from torch_geometric.data.data import Data

T = TypeVar("T", bound="MorphologyData")


class MorphologyData(Data):
    """An object that hold morphological data and features.

    It extends the `Data` class from torch-geometric to add additional
    features. In particular, the original `Data` class was meant to only
    hold graph data. We use it to also store persistence diagrams,
    persistence images, and other data.
    """

    def to_dict(self) -> dict:
        """Serialise to dictionary.

        Since we don't always store graph data, the `num_nodes` attribute
        cannot always be inferred. Therefore, we explicitly serialise it
        in order to recover it after de-serialisation.

        Returns
        -------
        dict
            The serialised morphology data.
        """
        data_dict: dict = super().to_dict()
        if hasattr(self, "__num_nodes__"):
            data_dict["num_nodes"] = self.num_nodes

        return data_dict

    @classmethod
    def load(cls: type[T], path: str | os.PathLike) -> T:
        """Load a serialised data object from disk."""
        data_dict = torch.load(path)
        data_obj: T = cls.from_dict(data_dict)
        return data_obj

    def save(self, path: str | os.PathLike) -> None:
        """Serialise the data object to disk."""
        data_dict = self.to_dict()
        torch.save(data_dict, path)

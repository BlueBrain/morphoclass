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
"""Implementation of the Cat layer."""
from __future__ import annotations

import torch
from torch.nn import Module


class Cat(Module):
    """Concatenate the outputs of multiple parallel layers."""

    def __init__(self):
        super().__init__()

    def forward(self, x_list):
        """Perform the forward pass.

        Parameters
        ----------
        x_list : iterable of torch.Tensor
            An iterable of outputs of a number of torch layers.

        Returns
        -------
        torch.Tensor
            The concatenated input tensors.
        """
        return torch.cat(x_list, dim=1)

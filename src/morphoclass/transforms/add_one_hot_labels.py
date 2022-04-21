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
"""Implementation of the `AddOneHotLabels` transform."""
from __future__ import annotations

import torch

from morphoclass.transforms.helper import require_field


class AddOneHotLabels:
    """Add one-hot labels to data.

    The prerequisite is that data already contains numerical sparse labels.
    These sparse labels are transformed into one-hot labels using the function
    provided in the constructor.

    Parameters
    ----------
    fn_get_oh_label : callable
        Function that maps sparse labels to one-hot labels.
    """

    def __init__(self, fn_get_oh_label):
        self.fn_get_oh_label = fn_get_oh_label

    @require_field("y")
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
        oh_label = self.fn_get_oh_label(data.y)
        data.y_oh = torch.tensor(oh_label).unsqueeze(0)

        return data

    def __repr__(self):
        """Compute the repr."""
        return f"{self.__class__.__name__}(fn_get_oh_label={self.fn_get_oh_label})"

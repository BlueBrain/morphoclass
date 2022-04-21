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
"""Implementation of the `GlobalFeatureToLabel` transform."""
from __future__ import annotations


class GlobalFeatureToLabel:
    """Convert a global feature to label.

    Parameters
    ----------
    global_feature_index : int
        The index of the global feature that shall be used.
    """

    def __init__(self, global_feature_index):
        self.global_feature_index = global_feature_index

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
        data.y = data.u[0, self.global_feature_index].item()

        return data

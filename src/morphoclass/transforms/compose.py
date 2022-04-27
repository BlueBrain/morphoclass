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
"""Implementation of the `Compose` transform."""
from __future__ import annotations


class Compose:
    """A composition of multiple transforms.

    Parameters
    ----------
    transforms : iterable
        The transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

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
        for transform in self.transforms:
            data = transform(data)
        return data

    def __repr__(self):
        """Compute the repr."""
        args_str = "\n".join(f"\t{transform}," for transform in self.transforms)
        return f"{self.__class__.__name__}([\n{args_str}\n])"

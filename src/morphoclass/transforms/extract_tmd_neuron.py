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
"""Implementation of the `ExtractTMDNeuron` transform."""
from __future__ import annotations

from morphoclass.transforms.helper import require_field
from morphoclass.utils import from_morphio_to_tmd


class ExtractTMDNeuron:
    """Convert the MorphIO morphology to the TMD Neuron class."""

    @require_field("morphology")
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
        data.tmd_neuron = from_morphio_to_tmd(data.morphology, remove_duplicates=True)

        return data

    def __repr__(self):
        """Compute the repr."""
        return f"{self.__class__.__name__}()"

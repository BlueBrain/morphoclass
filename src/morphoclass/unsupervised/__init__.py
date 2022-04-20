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
"""Utilities for unsupervised methods for neuron m-type classification."""
from __future__ import annotations

from morphoclass.unsupervised.plotting import make_silhouette_plot
from morphoclass.unsupervised.plotting import plot_embedding_pca

__all__ = [
    "plot_embedding_pca",
    "make_silhouette_plot",
]

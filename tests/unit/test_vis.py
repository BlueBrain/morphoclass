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
from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure
from tmd.io.io import load_neuron

import morphoclass.vis


# Warning issued because of networkx using matplotlib incorrectly. Remove once
# networkx has fixed this.
# Full warning message:
# Passing *transOffset* without *offsets* has no effect. This behavior is
# deprecated since 3.5 and in 3.6, *transOffset* will begin having an effect
# regardless of *offsets*. In the meantime, if you wish to set *transOffset*,
# call collection.set_offset_transform(transOffset) explicitly.
@pytest.mark.filterwarnings(r"ignore:Passing \*transOffset\*:DeprecationWarning")
def test_plot_tree():
    neuron = load_neuron("tests/data/L5/TPC_A/C050896A-P3.h5")
    tree = neuron.neurites[0]

    fig = Figure()
    ax = fig.subplots()
    morphoclass.vis.plot_tree(tree, ax)


def test_plot_mean_ci():
    values = np.random.random([10, 10])

    fig = Figure()
    ax = fig.subplots()
    morphoclass.vis.plot_mean_ci(ax, values, "values", "r")


def test_plot_learning_curves():
    acc = np.random.random([10, 10])
    loss = np.random.random([10, 10])
    val_acc = np.random.random([10, 10])
    val_loss = np.random.random([10, 10])

    morphoclass.vis.plot_learning_curves(acc, loss, val_acc, val_loss)

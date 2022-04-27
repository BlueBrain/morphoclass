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
from __future__ import annotations

import numpy as np
import pytest

from morphoclass.data import MorphologyDataset
from morphoclass.transforms import AverageRadius
from morphoclass.transforms import ExtractMaximalApicalPathLength
from morphoclass.transforms import ExtractNumberBranchPoints
from morphoclass.transforms import ExtractNumberLeaves
from morphoclass.transforms import ExtractTMDNeurites
from morphoclass.transforms import TotalPathLength


@pytest.fixture
def dataset():
    dataset = MorphologyDataset.from_csv(
        "tests/data/L5_data.csv",
        pre_transform=ExtractTMDNeurites(neurite_type="apical"),
    )
    return dataset


def test_extract_number_leaves(dataset):
    transform = ExtractNumberLeaves()
    assert str(transform) == "ExtractNumberLeaves()"
    dataset.transform = transform

    true_answers = [6, 11, 6, 7]
    global_features = [sample.u.item() for sample in dataset]
    assert all(a == b for a, b in zip(true_answers, global_features))


def test_extract_number_branch_points(dataset):
    transform = ExtractNumberBranchPoints()
    assert str(transform) == "ExtractNumberBranchPoints()"
    dataset.transform = transform

    true_answers = [5, 10, 5, 6]
    global_features = [sample.u.item() for sample in dataset]
    assert all(a == b for a, b in zip(true_answers, global_features))


def test_extract_maximal_apical_path_length(dataset):
    transform = ExtractMaximalApicalPathLength()
    assert str(transform) == "ExtractMaximalApicalPathLength()"
    dataset.transform = transform

    true_answers = np.array([572.03198242, 272.74130249, 366.44064331, 446.8019104])
    global_features = np.array([sample.u.item() for sample in dataset])
    assert np.allclose(true_answers, global_features)


@pytest.mark.parametrize("from_morphology", [True, False])
def test_total_path_length(dataset, from_morphology):
    transform = TotalPathLength()
    assert str(transform) == "TotalPathLength()"

    true_answers = np.array([919.37536621, 953.71313477, 978.45343018, 913.63848877])

    dataset.transform = TotalPathLength(from_morphology=from_morphology)
    global_features = np.array([sample.u.item() for sample in dataset])
    assert np.allclose(true_answers, global_features)


@pytest.mark.parametrize("from_morphology", [True, False])
def test_average_radius(dataset, from_morphology):
    transform = AverageRadius()
    assert str(transform) == "AverageRadius()"

    # Why do both methods give slightly different results?
    true_answers = {
        True: np.array([0.20658356, 0.20734136, 0.19838345, 0.19535339]),
        False: np.array([0.20658356, 0.20734136, 0.19838344, 0.19535339]),
    }

    dataset.transform = AverageRadius(from_morphology=from_morphology)
    global_features = np.array([sample.u.item() for sample in dataset])
    assert np.allclose(true_answers[from_morphology], global_features)

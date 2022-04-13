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

    true_answers = [47, 121, 13, 7]
    global_features = [sample.u.item() for sample in dataset]
    assert all(a == b for a, b in zip(true_answers, global_features))


def test_extract_number_branch_points(dataset):
    transform = ExtractNumberBranchPoints()
    assert str(transform) == "ExtractNumberBranchPoints()"
    dataset.transform = transform

    true_answers = [46, 120, 12, 6]
    global_features = [sample.u.item() for sample in dataset]
    assert all(a == b for a, b in zip(true_answers, global_features))


def test_extract_maximal_apical_path_length(dataset):
    transform = ExtractMaximalApicalPathLength()
    assert str(transform) == "ExtractMaximalApicalPathLength()"
    dataset.transform = transform

    true_answers = np.array([1264.56384277, 959.083313, 739.150696, 985.957581])
    global_features = np.array([sample.u.item() for sample in dataset])
    assert np.allclose(true_answers, global_features)


@pytest.mark.parametrize("from_morphology", [True, False])
def test_total_path_length(dataset, from_morphology):
    transform = TotalPathLength()
    assert str(transform) == "TotalPathLength()"

    true_answers = np.array(
        [5985.36474609, 5213.89306641, 1407.46386719, 1458.79882812]
    )

    dataset.transform = TotalPathLength(from_morphology=from_morphology)
    global_features = np.array([sample.u.item() for sample in dataset])
    assert np.allclose(true_answers, global_features)


@pytest.mark.parametrize("from_morphology", [True, False])
def test_average_radius(dataset, from_morphology):
    transform = AverageRadius()
    assert str(transform) == "AverageRadius()"

    # Why do both methods give slightly different results?
    true_answers = {
        True: np.array(
            [
                0.5068979263305664,
                0.5532674789428711,
                0.5768128633499146,
                0.6376450061798096,
            ]
        ),
        False: np.array(
            [0.523948609828949, 0.567867875099182, 0.588098704814910, 0.641212165355682]
        ),
    }

    dataset.transform = AverageRadius(from_morphology=from_morphology)
    global_features = np.array([sample.u.item() for sample in dataset])
    assert np.allclose(true_answers[from_morphology], global_features)

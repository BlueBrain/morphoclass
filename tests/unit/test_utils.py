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

import os
import sys

import neurom as nm
import numpy as np
import pytest
from tmd.io.io import load_neuron
from torch_geometric.data import Data

import morphoclass.utils


@pytest.fixture(scope="session")
def random_arr():
    return np.random.rand(2, 3)


@pytest.fixture(scope="session")
def tmp_file(tmpdir_factory):
    tmpdir = tmpdir_factory.mktemp("data").join("tmp_var.npy")
    return str(tmpdir)


def fake_function():
    print("This is some fake output")
    print("This is some fake output to err", file=sys.stderr)


def test_no_print():
    func = morphoclass.utils.no_print(fake_function)
    func()


@pytest.mark.parametrize("suppress_err", [True, False])
def test_suppress_print(suppress_err):
    with morphoclass.utils.suppress_print(suppress_err=suppress_err):
        fake_function()


def test_save_var(random_arr, tmp_file):
    print(tmp_file)
    morphoclass.utils.save_var(random_arr, tmp_file)


def test_load_var(random_arr, tmp_file):
    arr = morphoclass.utils.load_var(tmp_file)
    assert np.allclose(arr, random_arr)


def test_tmd_to_morphio():
    """Testing the conversion from MorphIO to TMD and the inverse
    NB: Pay attention to the points duplicated by the MorphIO method"""
    morphio_neuron = nm.load_morphology("tests/data/L5/UPC/rp101228_L5-2_idD.h5")
    tmd_new_neuron = morphoclass.utils.from_morphio_to_tmd(morphio_neuron)
    morphio_new_neuron = morphoclass.utils.from_tmd_to_morphio(tmd_new_neuron)
    assert len(morphio_neuron.root_sections) == len(morphio_new_neuron.root_sections)
    assert len(morphio_neuron.sections) == len(morphio_new_neuron.sections)

    test_tmd_neuron = load_neuron("tests/data/L5/TPC_C/C231296A-P4A2.h5")
    test_morphio_neuron = morphoclass.utils.from_tmd_to_morphio(test_tmd_neuron)
    test_morpho = nm.load_morphology("tests/data/L5/TPC_C/C231296A-P4A2.h5")
    sum1 = np.sum(
        [
            test_morphio_neuron.sections[i].points.shape[0]
            for i in range(len(test_morphio_neuron.sections))
        ]
    )
    sum2 = np.sum(
        [
            test_morpho.sections[i].points.shape[0]
            for i in range(len(test_morpho.sections))
        ]
    )
    assert sum1 == sum2

    # Test removing duplicates
    tmd_new_neuron_no_dup = morphoclass.utils.from_morphio_to_tmd(
        morphio_neuron, remove_duplicates=True
    )

    # Check that after remove the duplicates some trees contain fewer points
    comparisons = []
    for n1, n2 in zip(tmd_new_neuron.neurites, tmd_new_neuron_no_dup.neurites):
        ratio = len(n2.x) / len(n1.x)
        assert (
            ratio > 0.9
        )  # Heuristic to make sure we're comparing the right pair of neurites
        comparisons.append(ratio < 1.0)
    assert any(comparisons)


@pytest.mark.parametrize(
    "data_path, mtype, neuron_file, n_apicals",
    [
        ("tests/data/", "L5/UPC", "rp101228_L5-2_idD.h5", 0),
        ("tests/data/", "L5/UPC", "rp101228_L5-2_idD.h5", 1),
        ("tests/data/", "L5/UPC", "rp101228_L5-2_idD.h5", 2),
        ("tests/data/", "L5/UPC", "rp101228_L5-2_idD.h5", 3),
        ("tests/data/", "L5/UPC", "empty.txt", 0),
        ("tests/data/pickled/", "L5/UPC", "rp101228_L5-2_idD.pickle", 1),
    ],
)
def test_read_apical_from_file(data_path, mtype, neuron_file, n_apicals, monkeypatch):
    def load_neuron_new(path):
        neuron = load_neuron(path)
        tree = neuron.apical[0]
        neuron.apical = []
        for _ in range(n_apicals):
            new_tree = tree.copy_tree()
            # Invert the y coordinate in order to test inverted pyramidal cells
            new_tree.y = -new_tree.y
            neuron.apical.append(new_tree)
        return neuron

    monkeypatch.setattr("morphoclass.utils.load_neuron", load_neuron_new)

    full_path = os.path.join(data_path, mtype, neuron_file)

    ext = os.path.splitext(full_path)[-1]
    if ext not in [".swc", ".h5", ".pickle"]:
        ret = morphoclass.utils.read_apical_from_file(full_path, mtype)
        assert ret is None
        return

    if n_apicals == 0:
        with pytest.raises(ValueError):
            morphoclass.utils.read_apical_from_file(full_path, mtype)
    elif n_apicals < 3:
        morphoclass.utils.read_apical_from_file(full_path, mtype)
    else:
        with pytest.raises(NotImplementedError):
            morphoclass.utils.read_apical_from_file(full_path, mtype)


def test_read_layer_neurons_from_dir():
    morphoclass.utils.read_layer_neurons_from_dir("tests/data", 5)


def test_normalize_features():
    samples, labels, paths = morphoclass.utils.read_layer_neurons_from_dir(
        "tests/data", 5
    )
    samples = [sample for sample in samples if sample is not None]
    morphoclass.utils.normalize_features(samples)


def test_get_loader():
    samples, labels, paths = morphoclass.utils.read_layer_neurons_from_dir(
        "tests/data", 5
    )
    samples = [sample for sample in samples if sample is not None]
    morphoclass.utils.get_loader(samples, [0, 1])


def test_nodes_features():
    path = "tests/data/L5/TPC_A/C050896A-P3.h5"
    label = 0
    nodes_features = [
        "radial_dist",
        "path_dist",
        "coordinates",
        "vertical_dist",
        "angles",
    ]
    sample = morphoclass.utils.read_apical_from_file(
        path=path, label=label, nodes_features=nodes_features
    )
    assert isinstance(sample, Data)
    assert sample.x.shape[1] == 7
    sample1 = morphoclass.utils.read_apical_from_file(path=path, label=label)
    assert isinstance(sample1, Data)
    assert sample1.x.shape[1] == 1


def test_str_time_delta():
    test_cases_short = [
        (5.5, "5s"),
        (25 * 24 * 3600 + 15 * 3600 + 11 * 60 + 33.2, "25d 15h 11min 33s"),
        (13 * 3600 + 57.8, "13h 0min 57s"),
        (25 * 60 + 3.8, "25min 3s"),
    ]

    test_cases_long = [
        (5.5, "5 seconds"),
        (
            25 * 24 * 3600 + 15 * 3600 + 11 * 60 + 33.2,
            "25 days, 15 hours, 11 minutes, 33 seconds",
        ),
        (13 * 3600 + 57.8, "13 hours, 0 minutes, 57 seconds"),
        (25 * 60 + 3.8, "25 minutes, 3 seconds"),
    ]

    for dt, dt_str in test_cases_short:
        assert dt_str == morphoclass.utils.str_time_delta(dt)

    for dt, dt_str in test_cases_long:
        assert dt_str == morphoclass.utils.str_time_delta(dt, short=False)

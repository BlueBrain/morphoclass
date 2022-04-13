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

import logging
import pathlib

import neurom as nm
import pytest
from torch_geometric.data import Data

from morphoclass.data.filters import combined_filter
from morphoclass.data.filters import exclusion_filter
from morphoclass.data.filters import has_apicals_filter
from morphoclass.data.filters import inclusion_filter


@pytest.fixture(scope="session")
def samples():
    samples = [
        Data(path=pathlib.Path("TPC_A/a.h5")),
        Data(path=pathlib.Path("TPC_B/b.h5")),
        Data(path=pathlib.Path("IPC/c.cwc")),
        Data(path=pathlib.Path("IPC/d.asc")),
    ]

    morphology1 = nm.load_morphology("tests/data/L5/TPC_A/C050896A-P3.h5")
    morphology2 = nm.load_morphology("tests/data/L5/TPC_B/C030397A-P2.h5")
    for section in morphology2.root_sections:
        if section.type == nm.NeuriteType.apical_dendrite:
            morphology2.delete_section(section)

    samples[0].morphology = morphology1
    samples[1].morphology = morphology2

    return samples


def test_filename_exclusion_filter(samples):
    excluded_filenames = ["b", "c"]
    my_filter = exclusion_filter(excluded_filenames)
    expected_results = [True, False, False, True]
    results = map(my_filter, samples)

    assert all(r1 == r2 for r1, r2 in zip(results, expected_results))


def test_mtype_exclusion_filter(samples):
    my_filter = exclusion_filter("TPC")
    expected_results = [False, False, True, True]
    results = map(my_filter, samples)
    assert all(r1 == r2 for r1, r2 in zip(results, expected_results))


def test_combined_filter(samples):
    filter1 = exclusion_filter(["c", "d"])
    filter2 = exclusion_filter("TPC_A")
    my_filter = combined_filter(filter1, filter2)

    expected_results = [False, True, False, False]
    results = map(my_filter, samples)
    assert all(r1 == r2 for r1, r2 in zip(results, expected_results))


def test_filename_inclusion_filter(samples):
    included_filenames = ["b", "c"]
    my_filter = inclusion_filter(included_filenames)
    expected_results = [False, True, True, False]
    results = map(my_filter, samples)

    assert all(r1 == r2 for r1, r2 in zip(results, expected_results))


def test_has_apicals_filter(samples, caplog):
    expected_results = [True, False, False, False]
    results = []

    with caplog.at_level(logging.WARNING):
        for sample in samples:
            result = has_apicals_filter(sample)
            results.append(result)
    assert "data has no 'morphology' attribute" in caplog.text
    assert all(r1 == r2 for r1, r2 in zip(results, expected_results))

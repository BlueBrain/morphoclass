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
# from tns import extract_input
from __future__ import annotations

import pytest
from torch_geometric.transforms import Compose

from morphoclass.data.morphology_data import MorphologyData
from morphoclass.data.morphology_data_loader import MorphologyDataLoader
from morphoclass.data.morphology_dataset import MorphologyDataset
from morphoclass.transforms import ExtractRadialDistances
from morphoclass.transforms import ExtractTMDNeurites

# from morphoclass.data import TNSDataset, generate_tns_distributions


class TestMorphologyData:
    def test_num_nodes_serialisation(self):
        data = MorphologyData()

        # If data.num_nodes was not set, then it shouldn't be serialised
        assert not hasattr(data, "__num_nodes__")
        assert "num_nodes" not in data.to_dict()

        # After explicitly setting num_nodes it should be properly serialised
        data.num_nodes = 35
        assert hasattr(data, "__num_nodes__")
        loaded_data = MorphologyData.from_dict(data.to_dict())
        assert hasattr(loaded_data, "__num_nodes__")


class TestMorphologyDataset:
    def test_from_structured_dir(self):
        # Invalid directory
        with pytest.raises(ValueError):
            MorphologyDataset.from_structured_dir(
                data_path="tests/non_existent", layer="L5"
            )

        # No m-types found for given layer
        dataset = MorphologyDataset.from_structured_dir(
            data_path="tests/data", layer="L0"
        )
        assert len(dataset) == 0

        # Pre-filter and pre-transform
        MorphologyDataset.from_structured_dir(
            data_path="tests/data",
            layer="L5",
            pre_filter=lambda data: True,
            pre_transform=lambda data: data,
        )

        dataset = MorphologyDataset.from_structured_dir(
            data_path="tests/data", layer="L5"
        )
        assert dataset is not None
        assert len(dataset) == 8

    def test_from_csv_file(self):
        dataset = MorphologyDataset.from_csv("tests/data/L5_data.csv")
        assert len(dataset) == 4
        assert len(dataset.y_to_label) == 4
        for sample in dataset:
            assert sample.y_str == dataset.y_to_label[sample.y]
            assert sample.y == dataset.label_to_y[sample.y_str]

        dataset = MorphologyDataset.from_csv("tests/data/L5_data_no_labels.csv")
        assert len(dataset) == 4
        assert len(dataset.y_to_label) == 0
        for sample in dataset:
            assert sample.y is None
            assert not hasattr(sample, "y_str") or sample.y_str is None


def test_morphology_loader():
    transform = Compose(
        [
            ExtractTMDNeurites(neurite_type="apical"),
            ExtractRadialDistances(),
        ]
    )
    dataset = MorphologyDataset.from_structured_dir(
        data_path="tests/data", layer="L5", pre_transform=transform
    )
    assert dataset is not None

    loader = MorphologyDataLoader(dataset, batch_size=4)
    batch = next(iter(loader))
    assert len(batch.y) == 4


# @pytest.mark.xfail
# def test_tns_dataset():
#     dataset = MorphologyDataset(data_path="tests/data", layer="L5")
#     assert dataset is not None
#     assert len(dataset) == 4
#
#     distributions = generate_tns_distributions(dataset)
#     parameters = extract_input.parameters(
#         neurite_types=['apical'],
#         method='tmd')
#     parameters_all = {key: parameters for key in distributions}
#     tns_dataset = TNSDataset(distributions, parameters_all, 1)
#
#     # Overridden functions that aren't needed
#     assert tns_dataset.raw_file_names == []
#     assert tns_dataset.processed_file_names == []
#     assert tns_dataset.download() is None

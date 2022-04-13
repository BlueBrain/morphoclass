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

import pytest
from torch_geometric.transforms import Compose

from morphoclass.data import MorphologyDataset
from morphoclass.transforms import BranchingOnlyNeurites  # Other
from morphoclass.transforms import ExtractDistanceWeights
from morphoclass.transforms import ExtractEdgeIndex
from morphoclass.transforms import ExtractRadialDistances
from morphoclass.transforms import ExtractTMDNeurites
from morphoclass.transforms import FeatureRobustScaler
from morphoclass.transforms import MakeCopy


@pytest.fixture()
def dataset():
    print("dataset fixture called")
    pre_transform = Compose(
        [
            ExtractTMDNeurites(neurite_type="apical"),
        ]
    )

    dataset = MorphologyDataset.from_structured_dir(
        data_path="tests/data", layer="L5", pre_transform=pre_transform
    )

    return dataset


def test_extract_edge_index():
    transform = ExtractEdgeIndex()
    assert str(transform) == "ExtractEdgeIndex(make_undirected=False)"

    # Edge extraction
    pre_transform = Compose(
        [
            ExtractTMDNeurites(neurite_type="apical"),
            ExtractEdgeIndex(),
        ]
    )

    dataset = MorphologyDataset.from_structured_dir(
        data_path="tests/data", layer="L5", pre_transform=pre_transform
    )
    assert hasattr(dataset[0], "edge_index")

    # Apicals missing
    with pytest.raises(ValueError):
        MorphologyDataset.from_structured_dir(
            data_path="tests/data", layer="L5", pre_transform=ExtractEdgeIndex()
        )

    # Making undirected
    transform = ExtractEdgeIndex(make_undirected=True)
    assert str(transform) == "ExtractEdgeIndex(make_undirected=True)"
    dataset.transform = transform
    assert hasattr(dataset[0], "edge_index")


def test_extract_distance_weights(dataset):
    # Get an appropriate scale through Robust Scaler
    dataset.transform = Compose(
        [
            MakeCopy(),
            ExtractRadialDistances(),
        ]
    )

    scaler = FeatureRobustScaler(feature_indices=0, with_centering=False)
    scaler.fit(dataset)
    assert scaler.scale is not None

    transform = Compose(
        [
            MakeCopy(),
            BranchingOnlyNeurites(),
            ExtractEdgeIndex(),
            ExtractDistanceWeights(scale=scaler.scale.item() / 10),
        ]
    )

    dataset.transform = transform
    for sample in dataset:
        assert hasattr(sample, "edge_attr")
        assert hasattr(sample, "edge_index")
        n_edges = sample.edge_index.size()[-1]
        assert sample.edge_attr.size() == (n_edges, 1)
        assert sample.edge_attr.min() >= 0
        assert sample.edge_attr.max() <= 1
        assert not all(sample.edge_attr == 0)

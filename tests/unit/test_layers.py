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

from random import randint

import pytest
import torch
from torch_geometric.transforms import Compose

from morphoclass import data
from morphoclass import layers
from morphoclass import transforms


@pytest.fixture(scope="session")
def dataset():
    pre_transform = Compose(
        [
            transforms.ExtractTMDNeurites(neurite_type="apical"),
            transforms.BranchingOnlyNeurites(),
            transforms.ExtractEdgeIndex(),
        ]
    )

    dataset = data.MorphologyDataset.from_structured_dir(
        data_path="tests/data", layer="L5", pre_transform=pre_transform
    )

    return dataset


def test_attention_global_pool():
    # Fake data
    x = torch.rand(5, 3)
    batch = torch.tensor([0, 0, 0, 1, 1])

    # Simple attention pool
    att = layers.AttentionGlobalPool(3)
    out = att(x, batch)
    assert out.shape == (2, 3)

    # Attention pool + attention_per_feature + save_attention
    att = layers.AttentionGlobalPool(
        n_features=3, attention_per_feature=True, save_attention=True
    )
    out = att(x, batch)
    assert out.shape == (2, 3)
    assert att.last_a_j is not None
    assert att.last_a_j.shape == (5, 3)


def test_cheb_conv_separable(dataset):
    dataset.transform = Compose(
        [
            transforms.MakeCopy(keep_fields=["edge_index", "tmd_neurites"]),
            transforms.ExtractCoordinates(),
        ]
    )

    for sample in dataset:
        n_nodes, in_channels = sample.x.size()
        for orders in [5, [0, 2, 4], [3, 8], [6, 2, 2, 5]]:
            out_channels = randint(1, 15)
            conv = layers.ChebConvSeparable(
                in_channels=in_channels, out_channels=out_channels, orders=orders
            )
            x, edge_index = sample.x, sample.edge_index
            out = conv(x, edge_index)
            assert out.shape == (n_nodes, out_channels)


def test_tree_lstm_pool(dataset):
    x_size = 16
    h_size = 32

    x = torch.rand(10, x_size)
    edge_index = torch.tensor(
        [
            [0, 1],
            [1, 2],
            [1, 3],
            [2, 4],
            [2, 5],
            [3, 6],
            [3, 7],
            [6, 8],
            [6, 9],
        ]
    ).t()

    x_batch = torch.cat([x, x])
    edge_index_batch = torch.cat([edge_index, edge_index + edge_index.max() + 1], dim=1)

    pool = layers.TreeLSTMPool(x_size, h_size)
    result = pool(x_batch, edge_index_batch)
    assert result.shape == (2, h_size)

    # TODO: test with real data


def test_running_std():
    n_batch = 32
    shape = (4, 8)
    t1 = torch.rand(n_batch, *shape)
    t2 = torch.rand(n_batch, *shape)

    layer = layers.RunningStd(*shape)
    layer.train()
    have = layer(t1).std(dim=0, unbiased=False)
    want = torch.ones(shape)
    assert torch.allclose(have, want, atol=1e-3)

    layer = layers.RunningStd(*shape)
    layer.train()
    layer(t1)
    layer.eval()
    have = layer(t2)
    want = t2 / t1.std(dim=0, unbiased=False)
    assert torch.allclose(have, want, atol=1e-3)

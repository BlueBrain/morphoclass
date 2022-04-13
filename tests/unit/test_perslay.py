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
import torch

from morphoclass.layers.perslay import GaussianPointTransformer
from morphoclass.layers.perslay import PersLay
from morphoclass.layers.perslay import PointwisePointTransformer


@pytest.fixture(scope="session")
def inputs():
    input = torch.tensor(
        [
            [0.2, 0.3],
            [0.1, 0.4],
            [0.6, 0.3],
            [0.2, 0.1],
            [0.1, 0.3],
            [0.6, 0.1],
            [0.4, 0.7],
            [0.8, 0.1],
            [0.4, 0.2],
        ]
    )
    point_index = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 2])
    return input, point_index


def test_gaussian_point_transformer(inputs):
    out_features = 16
    m = GaussianPointTransformer(out_features=out_features)
    input, point_idx = inputs
    for attr in (m.sample_points, m.sample_inverse_sigmas):
        assert np.all(attr.shape == np.array([2, out_features]))
    assert np.all(((m.sample_points >= 0) & (m.sample_points <= 1)).tolist())
    assert np.all(
        ((m.sample_inverse_sigmas >= 0.8) & (m.sample_inverse_sigmas <= 1.2)).tolist()
    )

    output = m(input, point_idx)
    assert np.all(output.shape == np.array([len(input), out_features]))

    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            out_expected = torch.exp(
                -(
                    (
                        (input[i, :] - m.sample_points[:, j])
                        * m.sample_inverse_sigmas[:, j]
                    )
                    ** 2
                ).sum()
            )
            assert np.allclose(
                output[i, j].detach().numpy(), out_expected.detach().numpy()
            )


def test_pointwise_point_transformer(inputs):
    out_features = 16
    m = PointwisePointTransformer(out_features=out_features)
    input, point_idx = inputs
    output = m(input, point_idx)
    assert np.all(output.shape == np.array([len(input), out_features]))


@pytest.mark.parametrize(
    "transformation",
    [
        "gaussian",
        "pointwise",
        PointwisePointTransformer(out_features=16, hidden_features=128),
    ],
)
@pytest.mark.parametrize("operation", ["sum", "mean", "max"])
@pytest.mark.parametrize("weights", ["attention", "uniform", "grid"])
def test_perslay(inputs, transformation, operation, weights):
    out_features = 16
    m = PersLay(
        out_features=out_features,
        transformation=transformation,
        operation=operation,
        weights=weights,
    )
    input, point_idx = inputs
    n_diagrams_in_batch = len(np.unique(point_idx))
    output = m(input, point_idx)
    assert np.all(output.shape == np.array([n_diagrams_in_batch, out_features]))

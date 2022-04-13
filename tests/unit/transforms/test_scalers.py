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

import torch
from torch_geometric.transforms import Compose

from morphoclass.data import MorphologyDataLoader
from morphoclass.data import MorphologyDataset
from morphoclass.transforms import ExtractCoordinates
from morphoclass.transforms import ExtractRadialDistances
from morphoclass.transforms import ExtractTMDNeurites
from morphoclass.transforms import FeatureManualScaler
from morphoclass.transforms import FeatureMinMaxScaler
from morphoclass.transforms import FeatureRobustScaler
from morphoclass.transforms import FeatureStandardScaler
from morphoclass.transforms import scaler_from_config


def test_min_max_scaler():
    transform = Compose(
        [
            ExtractTMDNeurites(neurite_type="apical"),
            ExtractRadialDistances(),
            ExtractCoordinates(shift_to_origin=True),
        ]
    )
    dataset = MorphologyDataset.from_structured_dir(
        data_path="tests/data", layer="L5", pre_transform=transform
    )

    # Scaler on one feature
    scaler = FeatureMinMaxScaler(feature_indices=0)
    scaler.fit(dataset)
    dataset.transform = scaler
    loader = MorphologyDataLoader(dataset, batch_size=len(dataset))
    batch = next(iter(loader))
    radial_distances = batch.x[:, 0]
    assert radial_distances.min() >= 0
    assert radial_distances.max() <= 1

    # Scaler on multiple features
    scaler = FeatureMinMaxScaler(feature_indices=[0, 2])
    scaler.fit(dataset)
    dataset.transform = scaler
    loader = MorphologyDataLoader(dataset, batch_size=len(dataset))
    batch = next(iter(loader))
    radial_distances = batch.x[:, [0, 2]]
    assert radial_distances.min() >= 0
    assert radial_distances.max() <= 1

    # Scaler fit on a subset of samples
    scaler = FeatureMinMaxScaler(feature_indices=[0, 2])
    scaler.fit(dataset, idx=[0, 1])
    dataset.transform = scaler
    loader = MorphologyDataLoader(dataset, batch_size=len(dataset))
    batch = next(iter(loader))
    radial_distances = batch.x[[0, 1], [0, 2]]
    assert radial_distances.min() >= 0
    assert radial_distances.max() <= 1

    # Reconstruction
    config = scaler.get_config()
    scaler_reconstructed = scaler_from_config(config)
    assert all(scaler.scale == scaler_reconstructed.scale)
    assert all(scaler.min == scaler_reconstructed.min)


def test_standard_scaler():
    pre_transform = Compose(
        [
            ExtractTMDNeurites(neurite_type="apical"),
            ExtractRadialDistances(),
            ExtractCoordinates(shift_to_origin=True),
        ]
    )
    dataset = MorphologyDataset.from_structured_dir(
        data_path="tests/data", layer="L5", pre_transform=pre_transform
    )

    scaler_radial_distances = FeatureMinMaxScaler(feature_indices=0)
    scaler_coordinates = FeatureStandardScaler(feature_indices=[1, 2, 3])
    scaler_radial_distances.fit(dataset)
    scaler_coordinates.fit(dataset)

    dataset.transform = Compose(
        [
            scaler_radial_distances,
            scaler_coordinates,
        ]
    )
    loader = MorphologyDataLoader(dataset, batch_size=len(dataset))
    batch = next(iter(loader))
    features = batch.x

    assert features[:, 0].min() >= 0
    assert features[:, 0].max() <= 1
    assert (
        features[:, 1:].mean(dim=0).allclose(torch.tensor([0.0, 0.0, 0.0]), atol=1e-3)
    )
    assert features[:, 1:].std(dim=0).allclose(torch.tensor([1.0, 1.0, 1.0]), atol=1e-3)

    # Reconstruction
    config = scaler_coordinates.get_config()
    scaler_reconstructed = scaler_from_config(config)
    assert all(scaler_coordinates.scale == scaler_reconstructed.scale)
    assert all(scaler_coordinates.mean == scaler_reconstructed.mean)


def test_manual_scaler():
    pre_transform = Compose(
        [
            ExtractTMDNeurites(neurite_type="apical"),
            ExtractRadialDistances(),
            ExtractCoordinates(shift_to_origin=True),
        ]
    )
    dataset = MorphologyDataset.from_structured_dir(
        data_path="tests/data", layer="L5", pre_transform=pre_transform
    )

    feature_index = 1
    shift, scale = 2.5, 3.3
    scaler = FeatureManualScaler(
        feature_indices=feature_index, shift=shift, scale=scale
    )
    assert scaler._fit(None) is None

    loader = MorphologyDataLoader(dataset, batch_size=len(dataset))
    batch = next(iter(loader))
    features_before = batch.x[:, feature_index]

    dataset.transform = scaler

    loader = MorphologyDataLoader(dataset, batch_size=len(dataset))
    batch = next(iter(loader))
    features_after = batch.x[:, feature_index]

    assert torch.allclose(features_after, (features_before - shift) / scale)

    # Reconstruction
    config = scaler.get_config()
    scaler_reconstructed = scaler_from_config(config)
    assert scaler.scale == scaler_reconstructed.scale
    assert scaler.shift == scaler_reconstructed.shift


def test_robust_scaler():
    pre_transform = Compose(
        [
            ExtractTMDNeurites(neurite_type="apical"),
            ExtractRadialDistances(),
            ExtractCoordinates(shift_to_origin=True),
        ]
    )
    dataset = MorphologyDataset.from_structured_dir(
        data_path="tests/data", layer="L5", pre_transform=pre_transform
    )

    scaler_radial_distances = FeatureRobustScaler(
        feature_indices=0, with_centering=False
    )
    scaler_coordinates = FeatureRobustScaler(
        feature_indices=[1, 2, 3], with_centering=False
    )
    scaler_radial_distances.fit(dataset)
    scaler_coordinates.fit(dataset)

    loader = MorphologyDataLoader(dataset, batch_size=len(dataset))
    batch = next(iter(loader))
    features_before_scaling = batch.x.clone()

    dataset.transform = Compose(
        [
            scaler_radial_distances,
            scaler_coordinates,
        ]
    )
    loader = MorphologyDataLoader(dataset, batch_size=len(dataset))
    batch = next(iter(loader))
    features_after_scaling = batch.x

    nonzero = features_before_scaling != 0
    before = features_after_scaling[nonzero].abs()
    after = features_before_scaling[nonzero].abs()
    assert torch.all(before < after)

    # Reconstruction
    config = scaler_coordinates.get_config()
    scaler_reconstructed = scaler_from_config(config)
    assert all(scaler_coordinates.scale == scaler_reconstructed.scale)
    assert scaler_coordinates.center == scaler_reconstructed.center

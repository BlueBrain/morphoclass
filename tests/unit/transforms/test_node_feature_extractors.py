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

import morphio
import numpy as np
import pytest
import torch
from torch_geometric.transforms import Compose

from morphoclass.data import MorphologyDataset
from morphoclass.transforms import BranchingOnlyNeurites
from morphoclass.transforms import ExtractBranchingAngles
from morphoclass.transforms import ExtractConstFeature
from morphoclass.transforms import ExtractCoordinates
from morphoclass.transforms import ExtractDiameters
from morphoclass.transforms import ExtractIsBranching
from morphoclass.transforms import ExtractIsIntermediate
from morphoclass.transforms import ExtractIsLeaf
from morphoclass.transforms import ExtractIsRoot
from morphoclass.transforms import ExtractPathDistances
from morphoclass.transforms import ExtractRadialDistances
from morphoclass.transforms import ExtractTMDNeurites
from morphoclass.transforms import ExtractVerticalDistances
from morphoclass.transforms import MakeCopy
from morphoclass.transforms import OrientApicals
from morphoclass.utils import morphio_root_section_to_tmd_tree


def test_extract_const_feature():
    transform = ExtractConstFeature()
    assert str(transform) == "ExtractConstFeature()"
    pre_transform = Compose(
        [
            ExtractTMDNeurites(neurite_type="apical"),
            ExtractConstFeature(),
        ]
    )
    dataset = MorphologyDataset.from_structured_dir(
        data_path="tests/data", layer="L5", pre_transform=pre_transform
    )
    data = dataset[0]
    assert hasattr(data, "x")
    assert all(data.x == 1)


def test_extract_radial_distances():
    transform = ExtractRadialDistances()
    assert str(transform) == (
        "ExtractRadialDistances(negative_ipcs=False, " "negative_bpcs=False)"
    )
    pre_transform = Compose(
        [
            ExtractTMDNeurites(neurite_type="apical"),
            ExtractRadialDistances(),
            ExtractRadialDistances(),
        ]
    )
    dataset = MorphologyDataset.from_structured_dir(
        data_path="tests/data", layer="L5", pre_transform=pre_transform
    )
    data = dataset[0]
    assert hasattr(data, "x")
    assert data.x.size()[1] == 2

    # Apicals missing
    with pytest.raises(ValueError):
        MorphologyDataset.from_structured_dir(
            data_path="tests/data", layer="L5", pre_transform=transform
        )

    # Test IPC and BPC special treatment
    dataset = MorphologyDataset.from_structured_dir(
        data_path="tests/data",
        layer="L5",
        pre_transform=ExtractTMDNeurites(neurite_type="apical"),
        transform=ExtractRadialDistances(negative_ipcs=True, negative_bpcs=True),
    )
    sample = dataset.data[0]
    sample.y_str = "xx_IPC_xx"
    assert all(dataset[0].x <= 0)
    delattr(dataset.data[0], "x")

    sample.y_str = "xx_BPC_xx"
    tree = sample.tmd_neurites[0].copy_tree()
    tree.y *= 0.5
    sample.tmd_neurites.append(tree)
    features = dataset[0].x
    assert any(features < 0)
    assert not all(features <= 0)


def test_extract_path_distances():
    transform = ExtractPathDistances()
    assert str(transform) == (
        "ExtractPathDistances(negative_ipcs=False, " "negative_bpcs=False)"
    )
    pre_transform = Compose(
        [
            ExtractTMDNeurites(neurite_type="apical"),
            ExtractPathDistances(),
            ExtractPathDistances(),
        ]
    )
    dataset = MorphologyDataset.from_structured_dir(
        data_path="tests/data", layer="L5", pre_transform=pre_transform
    )
    data = dataset[0]
    assert hasattr(data, "x")
    assert data.x.size()[1] == 2

    # Apicals missing
    with pytest.raises(ValueError):
        MorphologyDataset.from_structured_dir(
            data_path="tests/data", layer="L5", pre_transform=transform
        )

    # Test IPC and BPC special treatment
    dataset = MorphologyDataset.from_structured_dir(
        data_path="tests/data",
        layer="L5",
        pre_transform=ExtractTMDNeurites(neurite_type="apical"),
        transform=ExtractRadialDistances(negative_ipcs=True, negative_bpcs=True),
    )
    sample = dataset.data[0]
    sample.y_str = "xx_IPC_xx"
    assert all(dataset[0].x <= 0)
    delattr(dataset.data[0], "x")

    sample.y_str = "xx_BPC_xx"
    tree = sample.tmd_neurites[0].copy_tree()
    tree.y *= 0.5
    sample.tmd_neurites.append(tree)
    features = dataset[0].x
    assert any(features < 0)
    assert not all(features <= 0)


def test_extract_vertical_distances():
    with pytest.raises(ValueError):
        ExtractVerticalDistances(vertical_axis="invalid")

    transform = ExtractVerticalDistances()
    assert str(transform) == (
        "ExtractVerticalDistances(vertical_axis=y, "
        "negative_ipcs=False, negative_bpcs=False)"
    )
    pre_transform = Compose(
        [
            ExtractTMDNeurites(neurite_type="apical"),
            ExtractVerticalDistances(),
            ExtractVerticalDistances(vertical_axis="x"),
        ]
    )
    dataset = MorphologyDataset.from_structured_dir(
        data_path="tests/data", layer="L5", pre_transform=pre_transform
    )
    data = dataset[0]
    assert hasattr(data, "x")
    assert data.x.size()[1] == 2

    dataset.transform = transform

    # Apicals missing
    with pytest.raises(ValueError):
        MorphologyDataset.from_structured_dir(
            data_path="tests/data", layer="L5", pre_transform=transform
        )

    # More than one root node
    dataset = MorphologyDataset.from_structured_dir(
        data_path="tests/data",
        layer="L5",
        pre_transform=pre_transform,
        transform=transform,
    )
    with pytest.raises(ValueError):
        for data in dataset:
            data.tmd_neurites[0].p[:5] = -1
        dataset.__getitem__(0)

    # Test IPC and BPC special treatment
    dataset = MorphologyDataset.from_structured_dir(
        data_path="tests/data",
        layer="L5",
        pre_transform=Compose(
            [ExtractTMDNeurites(neurite_type="apical"), OrientApicals()]
        ),
        transform=ExtractVerticalDistances(negative_ipcs=True, negative_bpcs=True),
    )
    sample = dataset.data[0]
    sample.y_str = "xx_IPC_xx"
    assert (dataset[0].x <= 0).sum() > (dataset[0].x > 0).sum()
    delattr(dataset.data[0], "x")

    sample.y_str = "xx_BPC_xx"
    tree = sample.tmd_neurites[0].copy_tree()
    tree.y *= 0.5
    sample.tmd_neurites.append(tree)
    features = dataset[0].x
    assert any(features < 0)
    assert not all(features <= 0)


def test_extract_coordinates():
    transform = ExtractCoordinates()
    assert str(transform) == "ExtractCoordinates(shift_to_origin=True)"

    pre_transform = Compose(
        [
            ExtractTMDNeurites(neurite_type="apical"),
            ExtractCoordinates(),
            ExtractCoordinates(shift_to_origin=False),
        ]
    )
    dataset = MorphologyDataset.from_structured_dir(
        data_path="tests/data", layer="L5", pre_transform=pre_transform
    )
    data = dataset[0]
    assert hasattr(data, "x")
    assert data.x.size()[1] == 6

    # Apicals missing
    with pytest.raises(ValueError):
        MorphologyDataset.from_structured_dir(
            data_path="tests/data", layer="L5", pre_transform=transform
        )

    # More than one root node
    dataset = MorphologyDataset.from_structured_dir(
        data_path="tests/data",
        layer="L5",
        pre_transform=pre_transform,
        transform=transform,
    )
    with pytest.raises(ValueError):
        for data in dataset:
            data.tmd_neurites[0].p[:5] = -1
        dataset.__getitem__(0)


def test_extract_diameters():
    transform = ExtractDiameters()
    assert str(transform) == "ExtractDiameters()"

    pre_transform = Compose(
        [
            ExtractTMDNeurites(neurite_type="apical"),
            ExtractDiameters(),
        ]
    )
    dataset = MorphologyDataset.from_structured_dir(
        data_path="tests/data", layer="L5", pre_transform=pre_transform
    )

    for sample in dataset:
        assert hasattr(sample, "x") and sample.x.shape[-1] == 1


def test_extract_branching_angles(capsys):
    # Test __str__
    transform = ExtractBranchingAngles(non_branching_angle=-1)
    assert str(transform) == "ExtractBranchingAngles(non_branching_angle=-1)"
    transform = ExtractBranchingAngles()
    assert str(transform) == "ExtractBranchingAngles(non_branching_angle=0.0)"

    # Basic feature extraction
    transform = Compose(
        [MakeCopy(keep_fields="tmd_neurites"), ExtractBranchingAngles()]
    )
    dataset = MorphologyDataset.from_structured_dir(
        data_path="tests/data",
        layer="L5",
        pre_transform=ExtractTMDNeurites(neurite_type="apical"),
        transform=transform,
    )

    sample = dataset[0]

    # Zero branching angle
    transform = Compose(
        [MakeCopy(keep_fields=["morphology", "tmd_neurites"]), ExtractBranchingAngles()]
    )
    dataset = MorphologyDataset.from_structured_dir(
        data_path="tests/data",
        layer="L5",
        pre_transform=ExtractTMDNeurites(neurite_type="apical"),
        transform=transform,
    )
    data = dataset[0]
    for root in data.morphology.root_sections:
        if root.type == morphio.SectionType.apical_dendrite:
            data.tmd_neurites[0] = morphio_root_section_to_tmd_tree(
                root, remove_duplicates=False
            )
    transform(data)
    _, err = capsys.readouterr()
    assert "Zero branching angle found" in err

    # Multiple features extraction
    transform = Compose(
        [
            MakeCopy(keep_fields="tmd_neurites"),
            ExtractCoordinates(),
            ExtractBranchingAngles(),
        ]
    )
    dataset = MorphologyDataset.from_structured_dir(
        data_path="tests/data",
        layer="L5",
        pre_transform=ExtractTMDNeurites(neurite_type="apical"),
        transform=transform,
    )

    sample = dataset[0]

    # Extract angles from simplified tree
    transform = Compose(
        [
            MakeCopy(keep_fields="tmd_neurites"),
            BranchingOnlyNeurites(),
            ExtractBranchingAngles(),
        ]
    )
    dataset = MorphologyDataset.from_structured_dir(
        data_path="tests/data",
        layer="L5",
        pre_transform=ExtractTMDNeurites(neurite_type="apical"),
        transform=transform,
    )

    sample = dataset[0]

    del sample


def test_extract_is_root():
    # Test __str__
    transform = ExtractIsRoot()
    assert str(transform) == "ExtractIsRoot()"

    # Basic feature extraction
    transform = Compose([MakeCopy(keep_fields="tmd_neurites"), ExtractIsRoot()])
    dataset = MorphologyDataset.from_structured_dir(
        data_path="tests/data",
        layer="L5",
        pre_transform=ExtractTMDNeurites(neurite_type="apical"),
        transform=transform,
    )

    sample = dataset[0]
    apical = sample.tmd_neurites[0]
    root_idx = torch.tensor(apical.p == -1)
    not_root_idx = torch.tensor(apical.p != -1)
    assert all(sample.x[root_idx] == 1.0)
    assert all(sample.x[not_root_idx] == 0.0)

    del sample


def test_extract_is_leaf():
    # Test __str__
    transform = ExtractIsLeaf()
    assert str(transform) == "ExtractIsLeaf()"

    # Basic feature extraction
    transform = Compose([MakeCopy(keep_fields="tmd_neurites"), ExtractIsLeaf()])
    dataset = MorphologyDataset.from_structured_dir(
        data_path="tests/data",
        layer="L5",
        pre_transform=ExtractTMDNeurites(neurite_type="apical"),
        transform=transform,
    )

    sample = dataset[0]
    apical = sample.tmd_neurites[0]
    non_leaf_idx = torch.tensor(
        [1.0 if x in apical.p else 0.0 for x in range(len(apical.x))]
    )
    assert all(sample.x[non_leaf_idx.to(torch.bool)] == 0.0)
    assert all(sample.x[(-non_leaf_idx + 1.0).to(torch.bool)] == 1.0)

    del sample


def test_extract_is_intermediate():
    # Test __str__
    transform = ExtractIsIntermediate()
    assert str(transform) == "ExtractIsIntermediate()"

    # Basic feature extraction
    transform = Compose([MakeCopy(keep_fields="tmd_neurites"), ExtractIsIntermediate()])
    dataset = MorphologyDataset.from_structured_dir(
        data_path="tests/data",
        layer="L5",
        pre_transform=ExtractTMDNeurites(neurite_type="apical"),
        transform=transform,
    )

    sample = dataset[0]
    apical = sample.tmd_neurites[0]
    intermediate_idx = torch.tensor(
        [
            1.0
            if np.count_nonzero(apical.p == idx) == 1 and apical.p[idx] != -1
            else 0.0
            for idx in range(len(apical.x))
        ]
    )
    assert all(sample.x[intermediate_idx.to(torch.bool)] == 1.0)
    assert all(sample.x[(-intermediate_idx + 1.0).to(torch.bool)] == 0.0)

    del sample


def test_extract_is_branching():
    # Test __str__
    transform = ExtractIsBranching()
    assert str(transform) == "ExtractIsBranching()"

    # Basic feature extraction
    transform = Compose([MakeCopy(keep_fields="tmd_neurites"), ExtractIsBranching()])
    dataset = MorphologyDataset.from_structured_dir(
        data_path="tests/data",
        layer="L5",
        pre_transform=ExtractTMDNeurites(neurite_type="apical"),
        transform=transform,
    )

    sample = dataset[0]
    apical = sample.tmd_neurites[0]
    branching_idx = torch.tensor(
        [
            1.0 if np.count_nonzero(apical.p == idx) > 1 else 0.0
            for idx in range(len(apical.x))
        ]
    )
    assert all(sample.x[branching_idx.to(torch.bool)] == 1.0)
    assert all(sample.x[(-branching_idx + 1.0).to(torch.bool)] == 0.0)

    del sample


def test_extract_is_any():
    # Basic feature extraction
    transform = Compose(
        [
            MakeCopy(keep_fields="tmd_neurites"),
            ExtractIsRoot(),
            ExtractIsIntermediate(),
            ExtractIsBranching(),
            ExtractIsLeaf(),
        ]
    )
    dataset = MorphologyDataset.from_structured_dir(
        data_path="tests/data",
        layer="L5",
        pre_transform=ExtractTMDNeurites(neurite_type="apical"),
        transform=transform,
    )

    sample = dataset[0]
    assert all(sample.x.sum(dim=1) == 1.0)

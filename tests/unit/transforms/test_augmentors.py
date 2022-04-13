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

import pathlib

import numpy as np
import pytest
from torch_geometric.transforms import Compose

from morphoclass.data import MorphologyDataset
from morphoclass.transforms import AddNodesAtIntervals  # OrientNeuron,
from morphoclass.transforms import AddRandomPointsToReductionMask
from morphoclass.transforms import AddSectionMiddlePoints
from morphoclass.transforms import ApplyNodeReductionMasks
from morphoclass.transforms import BranchingOnlyNeurites
from morphoclass.transforms import BranchingOnlyNeuron
from morphoclass.transforms import EqualizeNodeCounts
from morphoclass.transforms import ExtractBranchingNodeReductionMasks
from morphoclass.transforms import ExtractIsIntermediate
from morphoclass.transforms import ExtractTMDNeurites
from morphoclass.transforms import ExtractTMDNeuron
from morphoclass.transforms import MakeCopy
from morphoclass.transforms import OrientApicals
from morphoclass.transforms import RandomJitter
from morphoclass.transforms import RandomRotation
from morphoclass.transforms import RandomStretching


def test_branching_only_trees():
    transform = BranchingOnlyNeuron()
    assert str(transform) == "BranchingOnlyNeuron()"

    pre_transform = Compose(
        [
            ExtractTMDNeuron(),
            BranchingOnlyNeuron(),
        ]
    )

    dataset = MorphologyDataset.from_structured_dir(
        data_path="tests/data", layer="L5", pre_transform=pre_transform
    )

    for sample in dataset:
        for tree in sample.tmd_neuron.neurites:
            # Locations of points with 1 child. For simplified trees
            # only the root should have one child
            rows, cols = np.nonzero(tree.dA.sum(axis=0) == 1)
            assert len(rows) == len(cols) == 1

    # TMD Neuron missing
    with pytest.raises(ValueError):
        MorphologyDataset.from_structured_dir(
            data_path="tests/data", layer="L5", pre_transform=transform
        )


def test_branching_only_apicals():
    transform = BranchingOnlyNeurites()
    assert str(transform) == "BranchingOnlyNeurites()"

    pre_transform = Compose(
        [
            ExtractTMDNeurites(neurite_type="apical"),
            BranchingOnlyNeurites(),
        ]
    )

    dataset = MorphologyDataset.from_structured_dir(
        data_path="tests/data", layer="L5", pre_transform=pre_transform
    )

    for sample in dataset:
        for tree in sample.tmd_neurites:
            # Locations of points with 1 child. For simplified trees
            # only the root should have one child
            rows, cols = np.nonzero(tree.dA.sum(axis=0) == 1)
            assert len(rows) == len(cols) == 1

    # TMD Neuron missing
    with pytest.raises(ValueError):
        MorphologyDataset.from_structured_dir(
            data_path="tests/data", layer="L5", pre_transform=transform
        )


def test_add_section_middle_points():
    n_points = 20

    pre_transform = ExtractTMDNeurites(neurite_type="apical")
    transform = Compose(
        [
            MakeCopy(keep_fields=["tmd_neurites"]),
            AddSectionMiddlePoints(n_points_to_add=n_points),
        ]
    )
    dataset = MorphologyDataset.from_structured_dir(
        data_path="tests/data",
        layer="L5",
        pre_transform=pre_transform,
        transform=transform,
    )

    new_apical = dataset[0].tmd_neurites[0]
    n_old = len(dataset.data[0].tmd_neurites[0].p)
    n_new = n_old + n_points
    assert len(new_apical.x) == n_new
    assert len(new_apical.y) == n_new
    assert len(new_apical.z) == n_new
    assert len(new_apical.d) == n_new
    assert len(new_apical.t) == n_new
    assert len(new_apical.p) == n_new
    dh_new, dw_new = new_apical.dA.shape
    assert dh_new == n_new
    assert dw_new == n_new


def test_extract_node_reduction_masks():
    pre_transform = Compose(
        [
            ExtractTMDNeurites(neurite_type="apical"),
            ExtractBranchingNodeReductionMasks(),
            ExtractIsIntermediate(),
        ]
    )
    dataset = MorphologyDataset.from_structured_dir(
        data_path="tests/data", layer="L5", pre_transform=pre_transform
    )

    for sample in dataset:
        intermediate = sample.x.detach().cpu().numpy().squeeze().astype(int)
        branching = sample.tmd_neurites_masks[0]
        assert np.all(intermediate == ~branching)


def test_add_random_points_to_reduction_mask():
    n_points = 20

    pre_transform = Compose(
        [
            ExtractTMDNeurites(neurite_type="apical"),
            ExtractBranchingNodeReductionMasks(),
        ]
    )
    transform = AddRandomPointsToReductionMask(n_points=n_points)

    dataset = MorphologyDataset.from_structured_dir(
        data_path="tests/data",
        layer="L5",
        pre_transform=pre_transform,
        transform=transform,
    )

    for i in range(len(dataset)):
        mask_before = dataset.data[i].tmd_neurites_masks[0].copy()
        mask_after = dataset[i].tmd_neurites_masks[0]
        assert mask_after.sum() == mask_before.sum() + n_points


def test_add_nodes_at_intervals():
    pre_transform = Compose(
        [
            ExtractTMDNeurites(neurite_type="apical"),
            ExtractBranchingNodeReductionMasks(),
        ]
    )
    transform = Compose(
        [
            MakeCopy(keep_fields=["tmd_neurites", "tmd_neurites_masks"]),
            AddNodesAtIntervals(interval=10),
            ApplyNodeReductionMasks(),
        ]
    )

    dataset = MorphologyDataset.from_structured_dir(
        data_path="tests/data",
        layer="L5",
        pre_transform=pre_transform,
        transform=transform,
    )

    original_tree = dataset.data[0].tmd_neurites[0]
    tree = dataset[0].tmd_neurites[0]

    n_original = len(original_tree.p)
    n_simplified = len(original_tree.extract_simplified().p)
    n_transformed = len(tree.p)

    assert n_simplified < n_transformed < n_original


def test_equalize_node_counts():
    pre_transform = Compose(
        [
            ExtractTMDNeurites(neurite_type="apical"),
            ExtractBranchingNodeReductionMasks(),
        ]
    )

    dataset = MorphologyDataset.from_structured_dir(
        data_path="tests/data", layer="L5", pre_transform=pre_transform
    )

    node_equalizer = EqualizeNodeCounts()
    node_equalizer.fit(dataset)

    dataset.transform = Compose(
        [
            MakeCopy(keep_fields=["tmd_neurites", "tmd_neurites_masks"]),
            node_equalizer,
            ApplyNodeReductionMasks(),
        ]
    )

    # Check that we fitted correctly, compared to the original tree
    # at least one sample will have the node count unchanged,
    # the other samples should have reduced node counts
    count_diffs_list = []
    for orig_sample, sample in zip(dataset.data, dataset):
        n_orig = sum(len(apical.p) for apical in orig_sample.tmd_neurites)
        n_trans = sum(len(apical.p) for apical in sample.tmd_neurites)
        count_diffs_list.append(n_trans - n_orig)
    count_diffs = np.array(count_diffs_list)

    assert any(count_diffs == 0)
    assert any(count_diffs < 0)
    assert not any(count_diffs > 0)

    # Test that if `min_node_count = 0` then the output is
    # branching only trees
    node_equalizer.min_total_nodes = 0
    count_diffs = []
    for orig_sample, sample in zip(dataset.data, dataset):
        n_orig = sum(
            len(apical.extract_simplified().p) for apical in orig_sample.tmd_neurites
        )
        n_trans = sum(len(apical.p) for apical in sample.tmd_neurites)
        count_diffs.append(n_trans - n_orig)
    count_diffs = np.array(count_diffs)

    assert all(count_diffs == 0)

    # Test that if `min_node_count` is bigger than all node counts in the
    # dataset, then the output is the original trees
    node_equalizer.min_total_nodes = float("inf")
    count_diffs = []
    for orig_sample, sample in zip(dataset.data, dataset):
        n_orig = sum(len(apical.p) for apical in orig_sample.tmd_neurites)
        n_trans = sum(len(apical.p) for apical in sample.tmd_neurites)
        count_diffs.append(n_trans - n_orig)
    count_diffs = np.array(count_diffs)

    assert all(count_diffs == 0)


def test_apply_node_reduction_mask():
    pre_transform = Compose(
        [ExtractTMDNeurites(neurite_type="apical"), ExtractIsIntermediate()]
    )
    dataset = MorphologyDataset.from_structured_dir(
        data_path="tests/data", layer="L5", pre_transform=pre_transform
    )

    # Add reduction masks to branching points
    for data in dataset:
        intermediate_pts = data.x.detach().cpu().squeeze().numpy()
        branching_points = 1 - intermediate_pts
        data.tmd_neurites_masks = [branching_points.astype(bool)]

    dataset.transform = Compose(
        [
            MakeCopy(keep_fields=["tmd_neurites", "tmd_neurites_masks"]),
            ApplyNodeReductionMasks(),
        ]
    )

    for i, sample in enumerate(dataset):
        old_apical = dataset.data[i].tmd_neurites[0]
        new_apical = sample.tmd_neurites[0]
        assert np.all(old_apical.extract_simplified().p == new_apical.p)


def test_random_stretching():
    # TODO
    dataset = MorphologyDataset.from_structured_dir(
        data_path="tests/data",
        layer="L5",
        pre_transform=ExtractTMDNeurites(neurite_type="apical"),
    )
    transform = RandomStretching(d_scale_x=0.2, d_scale_y=0.2, d_scale_z=0.2)
    transform(dataset[0])


def test_random_rotation():
    # TODO
    dataset = MorphologyDataset.from_structured_dir(
        data_path="tests/data",
        layer="L5",
        pre_transform=ExtractTMDNeurites(neurite_type="apical"),
    )
    transform = RandomRotation(only_y_rotation=False)
    transform(dataset[0])


def test_random_jitter():
    transform = RandomJitter(d_add=10, d_scale=0.5)

    # Test no apicals
    dataset = MorphologyDataset.from_structured_dir(
        data_path="tests/data", layer="L5", transform=transform
    )
    with pytest.raises(ValueError):
        dataset.__getitem__(0)

    # Test normal usage
    dataset = MorphologyDataset.from_structured_dir(
        data_path="tests/data",
        layer="L5",
        pre_transform=ExtractTMDNeurites(neurite_type="apical"),
        transform=transform,
    )
    dataset.__getitem__(0)

    # Test more than one root
    with pytest.raises(ValueError):
        dataset.data[0].tmd_neurites[0].p[1] = -1
        dataset.__getitem__(0)

    # Test no shift to origin
    dataset.transform.shift_to_origin = False
    dataset.__getitem__(0)


# Warning issued because of networkx using matplotlib incorrectly. Remove once
# networkx has fixed this.
# Full warning message:
# Passing *transOffset* without *offsets* has no effect. This behavior is
# deprecated since 3.5 and in 3.6, *transOffset* will begin having an effect
# regardless of *offsets*. In the meantime, if you wish to set *transOffset*,
# call collection.set_offset_transform(transOffset) explicitly.
@pytest.mark.filterwarnings(r"ignore:Passing \*transOffset\*:DeprecationWarning")
def test_orient_tree():
    # Test no apicals
    dataset = MorphologyDataset.from_structured_dir(
        data_path="tests/data",
        layer="L5",
        pre_transform=ExtractTMDNeurites(neurite_type="apical"),
    )
    assert dataset is not None
    # TODO, below is just manual testing
    dataset[0].path = (
        pathlib.Path(dataset[0].path).parent.parent / "HPC" / "whatever.h5"
    )
    del dataset[0].y_str
    print(pathlib.Path(dataset[0].path).parent.stem)
    dataset.transform = Compose(
        [
            MakeCopy(),
            OrientApicals(special_treatment_hpcs=True, special_treatment_ipcs=True),
        ]
    )
    from matplotlib.figure import Figure

    from morphoclass import vis

    fig = Figure()
    ax = fig.subplots()
    vis.plot_tree(dataset[0].tmd_neurites[0], ax=ax)
    # fig.show()

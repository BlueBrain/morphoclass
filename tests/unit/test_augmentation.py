# Copyright © 2022-2022 Blue Brain Project/EPFL
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

import warnings
from collections import defaultdict

import numpy as np
import pytest
import tmd.Tree.Tree

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from morphoclass.augmentation import add_node_specific_radial_dist
    from morphoclass.augmentation import add_nodes
    from morphoclass.augmentation import all_same_number_of_nodes
    from morphoclass.augmentation import coordinates_noise_augmentation
    from morphoclass.augmentation import coordinates_stretching
    from morphoclass.augmentation import corresponding_parents_list
    from morphoclass.augmentation import create_middle_points
    from morphoclass.augmentation import create_neuron_from_mask
    from morphoclass.augmentation import finding_point_on_branch
    from morphoclass.augmentation import get_simplified_nodes_masks
    from morphoclass.augmentation import nodes_specific_interval_dist
    from morphoclass.augmentation import radial_dist_hist_multiple_classes
    from morphoclass.augmentation import radial_dist_histograms_equalization
    from morphoclass.augmentation import tree_rotation


def test_coordinates_noise_augmentation(simple_apical):
    augmented_tree = coordinates_noise_augmentation(simple_apical)
    assert isinstance(augmented_tree, tmd.Tree.Tree)
    assert len(simple_apical.x) == len(augmented_tree.x)
    assert np.all(simple_apical.p == augmented_tree.p)
    assert np.all(simple_apical.d == augmented_tree.d)


def test_coordinates_stretching_augmentation(simple_apical):
    augmented_tree = coordinates_stretching(simple_apical)
    assert isinstance(augmented_tree, tmd.Tree.Tree)
    assert len(simple_apical.x) == len(augmented_tree.x)
    assert np.all(simple_apical.p == augmented_tree.p)
    assert np.all(simple_apical.d == augmented_tree.d)


def test_rotation_augmentation(simple_apical):
    augmented_tree = tree_rotation(simple_apical)
    assert isinstance(augmented_tree, tmd.Tree.Tree)
    assert len(simple_apical.x) == len(augmented_tree.x)
    assert np.all(simple_apical.p == augmented_tree.p)
    assert np.all(simple_apical.d == augmented_tree.d)


def test_create_middle_points(simple_apical):
    simplified_apical = simple_apical.extract_simplified()
    augmented_apical = create_middle_points(simplified_apical)
    assert len(augmented_apical.x) == 5
    with pytest.raises(ValueError):
        augmented_apical = create_middle_points(
            simple_apical, number_of_points_to_add=10.1
        )


def test_simplified_node_mask(simple_apical):
    mask = get_simplified_nodes_masks(simple_apical)
    assert np.all(mask == [1, 1, 0, 1, 1])


def test_corresponding_parent_list(simple_apical):
    mask = get_simplified_nodes_masks(simple_apical)
    parent_list = corresponding_parents_list(simple_apical, mask)
    mask_2 = np.ones(len(simple_apical.x))
    parent_list_2 = corresponding_parents_list(simple_apical, mask_2)
    assert np.all(parent_list == [-1, 0, 1, 1])
    assert np.all(parent_list_2 == [-1, 0, 1, 2, 1])


def test_create_neuron_from_mask(simple_apical):
    mask = [1, 0, 0, 1, 1]
    augmented_tree = create_neuron_from_mask(simple_apical, mask)
    mask_2 = [1, 1, 0, 1, 1]
    augmented_tree_2 = create_neuron_from_mask(simple_apical, mask_2)
    assert len(augmented_tree.x) == 3
    assert np.all(augmented_tree.p == [-1, 0, 0])
    assert len(augmented_tree_2.x) == 4
    assert np.all(augmented_tree_2.p == [-1, 0, 1, 1])


def test_add_nodes(simple_apical):
    augmented_tree, add_point = add_nodes(simple_apical, number_points_to_add=1)
    if add_point == 1:
        assert len(augmented_tree.x) == 5
    else:
        assert len(augmented_tree.x) == 4
    augmented_tree_2, add_point_2 = add_nodes(simple_apical, number_points_to_add=100)
    print(add_point_2)
    assert (
        len(augmented_tree_2.x)
        == len(simple_apical.extract_simplified().x) + add_point_2
    )


def test_add_node_specific_radial_dist(simple_apical_2):
    simplified_mask = get_simplified_nodes_masks(simple_apical_2)
    new_simplified_mask, add_point = add_node_specific_radial_dist(
        simple_apical_2,
        simplified_mask,
        desired_value_min=10,
        desired_value_max=np.sqrt(500),
    )
    new_simplified_mask_2, add_point_2 = add_node_specific_radial_dist(
        simple_apical_2, simplified_mask, desired_value_min=500, desired_value_max=600
    )
    new_simplified_mask_3, add_point_3 = add_node_specific_radial_dist(
        simple_apical_2, desired_value_min=10, desired_value_max=np.sqrt(500)
    )
    if add_point:
        assert np.sum(new_simplified_mask) == 7
    if add_point_3:
        assert np.sum(new_simplified_mask_2) == 7
    assert add_point_2 is False


def test_all_the_same_number_of_nodes(simple_apical_2, simple_apical_3):
    apicals_list = [simple_apical_2, simple_apical_3]
    new_apicals_list, add_point_list = all_same_number_of_nodes(
        apicals_list, number_of_nodes=4
    )
    print(add_point_list)
    assert (
        len(new_apicals_list[0].x)
        == (len(apicals_list[0].extract_simplified().x)) + add_point_list[0]
    )
    assert (
        len(new_apicals_list[1].x)
        == (len(apicals_list[1].extract_simplified().x)) + add_point_list[1]
    )
    new_apicals_list_2, add_point_list_2 = all_same_number_of_nodes(
        apicals_list, number_of_nodes=10
    )
    assert (
        len(new_apicals_list_2[0].x)
        == (len(apicals_list[0].extract_simplified().x)) + add_point_list_2[0]
    )
    assert (
        len(new_apicals_list_2[1].x)
        == (len(apicals_list[1].extract_simplified().x)) + add_point_list_2[1]
    )


def test_radial_dist_histograms_equalization(simple_apical_2, simple_apical_3):
    tree_list_1 = [simple_apical_2]
    tree_list_2 = [simple_apical_3]
    (
        new_tree_list_1,
        new_tree_list_2,
        masks_list_1,
        masks_list_2,
    ) = radial_dist_histograms_equalization(tree_list_1, tree_list_2, bins=[0, 10, 30])
    assert isinstance(new_tree_list_1[0], tmd.Tree.Tree)
    assert isinstance(new_tree_list_2[0], tmd.Tree.Tree)
    assert len(new_tree_list_1[0].x) == len(new_tree_list_2[0].x)
    (
        new_tree_list_1,
        new_tree_list_2,
        masks_list_1,
        masks_list_2,
    ) = radial_dist_histograms_equalization(tree_list_2, tree_list_1, bins=[0, 10, 30])
    assert isinstance(new_tree_list_1[0], tmd.Tree.Tree)
    assert isinstance(new_tree_list_2[0], tmd.Tree.Tree)
    assert len(new_tree_list_1[0].x) == len(new_tree_list_2[0].x)


def test_nodes_specific_distances(simple_apical):
    new_tree = nodes_specific_interval_dist(simple_apical, threshold=0)
    assert isinstance(new_tree, tmd.Tree.Tree)
    assert len(new_tree.x) == 5


def test_finding_branches_points(simple_apical):
    list_branch_nodes = finding_point_on_branch(simple_apical, start=1, end=3)
    assert set(list_branch_nodes) == {1, 2, 3}


def test_radial_dist_multi_class():
    new_neurons, mask_list = radial_dist_hist_multiple_classes("tests/data/", layer=5)
    assert isinstance(new_neurons, defaultdict)
    assert isinstance(mask_list, defaultdict)

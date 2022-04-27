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
"""Utilities for morphology data augmentation (deprecated).

The functionality in this module has been superseded by the transforms in
the `morphoclass.transforms` module. Please used them.
"""
from __future__ import annotations

import os
import pathlib
import random
import warnings
from collections import defaultdict

import numpy as np
import tmd.Tree.Tree
from tmd.io.io import load_neuron

warnings.warn(
    "The morphoclass.augmentation module is deprecated, "
    "use morphoclass.transforms instead.",
    DeprecationWarning,
    stacklevel=2,
)


# Augmentation focused on the coordinates
def coordinates_stretching(
    tree, scale_x_interval=0.2, scale_y_interval=0.2, scale_z_interval=0.2
):
    """Create a new tree with augmentation on the coordinates (stretching).

    Parameters
    ----------
    tree: tmd.Tree.Tree
        Original Tree for the augmentation purpose
    scale_x_interval: float
        Maximum stretching value regarding x axis
    scale_y_interval: float
        Maximum stretching value regarding y axis
    scale_z_interval: float
        Maximum stretching value regarding z axis

    Returns
    -------
    augmented_tree: tmd.Tree.Tree
        Augmented Tree
    """
    scale_x = random.uniform(1 - scale_x_interval, 1 + scale_x_interval)
    scale_y = random.uniform(1 - scale_y_interval, 1 + scale_y_interval)
    scale_z = random.uniform(1 - scale_z_interval, 1 + scale_z_interval)

    new_tree = tree.copy_tree()

    new_x = new_tree.x * scale_x
    new_y = new_tree.y * scale_y
    new_z = new_tree.z * scale_z

    augmented_tree = tmd.Tree.Tree(
        x=new_x, y=new_y, z=new_z, d=tree.d, t=tree.t, p=tree.p
    )

    return augmented_tree


def tree_rotation(tree):
    """Create a new tree with augmentation on the coordinates (rotation).

    Parameters
    ----------
    tree: tmd.Tree.Tree
        Original Tree for the augmentation purpose

    Returns
    -------
    augmented_tree: tmd.Tree.Tree
        Augmented Tree
    """
    new_tree = tree.copy_tree()

    theta_y = random.uniform(0, 2 * np.pi)
    new_tree.rotate_xy(theta_y)

    return new_tree


def coordinates_noise_augmentation(tree, noise=0.05):
    """Create a new tree with augmentation on the coordinates (noise).

    Parameters
    ----------
    tree: tmd.Tree.Tree
        Original Tree for the augmentation purpose
    noise: float
        Percentage of noise added to the coordinates

    Returns
    -------
    augmented_tree: tmd.Tree.Tree
        Augmented Tree
    """
    new_tree = tree.copy_tree()

    new_x = new_tree.x + new_tree.x * np.random.random(len(new_tree.x)) * noise
    new_y = new_tree.y + new_tree.y * np.random.random(len(new_tree.y)) * noise
    new_z = new_tree.z + new_tree.z * np.random.random(len(new_tree.z)) * noise

    augmented_tree = tmd.Tree.Tree(
        x=new_x, y=new_y, z=new_z, d=tree.d, t=tree.t, p=tree.p
    )

    return augmented_tree


# Augmentation focused on the number of nodes
# Adding Synthetic nodes
def create_middle_points(tree, number_of_points_to_add=1):
    """Augment the tree by inserting a given number of middle points.

    This creates an augmented tree by creating the number of points specified
    in the middle of segments between two existing points.

    NB: Those points are artificially created (in the middle of segments
    between two existing points).

    Parameters
    ----------
    tree: tmd.Tree.Tree
        Original tree for the augmentation
    number_of_points_to_add: int
        Number of points to add in the augmented tree

    Returns
    -------
    augmented_tree: tmd.Tree.Tree
        Augmented Tree
    """
    if isinstance(number_of_points_to_add, int) is False:
        raise ValueError("The number of points to add has to be an int")

    beg, end = tree.get_sections_2()

    new_x = tree.x
    new_y = tree.y
    new_z = tree.z
    new_d = tree.d
    new_t = tree.t
    new_p = tree.p

    for _ in range(number_of_points_to_add):
        section_number = np.random.randint(1, len(end) - 1)
        begin_section = beg[section_number]
        end_section = end[section_number]

        new_x = np.append(
            new_x,
            (tree.x[begin_section] + (tree.x[end_section] - tree.x[begin_section]) / 2),
        )
        new_y = np.append(
            new_y,
            (tree.y[begin_section] + (tree.y[end_section] - tree.y[begin_section]) / 2),
        )
        new_z = np.append(
            new_z,
            (tree.z[begin_section] + (tree.z[end_section] - tree.z[begin_section]) / 2),
        )
        new_d = np.append(
            new_d,
            (tree.d[begin_section] + (tree.d[end_section] - tree.d[begin_section]) / 2),
        )
        new_t = np.append(new_t, tree.t[begin_section])
        new_p = np.append(new_p, tree.p[section_number + 1])
        new_p[section_number + 1] = len(new_p) - 1

    augmented_tree = tmd.Tree.Tree(x=new_x, y=new_y, z=new_z, d=new_d, t=new_t, p=new_p)
    return augmented_tree


# From the non simplified tree
def get_simplified_nodes_masks(tree):
    """Compute the node mask for the simplified tree.

    Extracts from the non-simplified tree, the simplified nodes and creates
    a mask.

    Parameters
    ----------
    tree: tmd.Tree.Tree
        Original tree

    Returns
    -------
    mask: list
        List identifying the simplified nodes among all the nodes
    """
    simplified_nodes = []

    beg, end = tree.get_sections_2()
    simplified_nodes.append(beg[0])
    simplified_nodes.extend(end)
    mask = np.zeros(len(tree.x))
    mask[simplified_nodes] = 1

    return mask


def corresponding_parents_list(tree, mask):
    """Create the list of the parents given the mask list.

    Parameters
    ----------
    tree: tmd.Tree.Tree
        Original tree
    mask: list
        Corresponding list of the nodes kept for the simplified tree

    Return
    ------
    parent_list: list
        Corresponding list of the nodes from the mask list
    """
    parent_list = []
    indexes = [ind for ind, i in enumerate(mask) if i == 1.0]
    for ind, i in enumerate(mask):
        if i == 1:
            if tree.p[ind] == -1:
                parent_list.append(-1)
            else:
                point_number = ind
                while True:
                    parent = tree.p[point_number]
                    if parent in indexes:
                        parent_list.append(np.where(indexes == parent)[0][0])
                        break
                    else:
                        point_number = parent
                        if point_number == 0:
                            break

    return parent_list


def create_neuron_from_mask(tree, simplified_nodes_mask):
    """Create a new neuron from the original tree and the mask of the new nodes.

    Parameters
    ----------
    tree: tmd.Tree.Tree
        Original tree (non-simplified)
    simplified_nodes_mask: list
        List of the nodes that have to be kept for the new simplified tree

    Returns
    -------
    augmented_tree: tmd.Tree.Tree
        Augmented tree with the new list of nodes
    """
    indexes = [ind for ind, i in enumerate(simplified_nodes_mask) if i == 1.0]
    parents_list = corresponding_parents_list(tree, mask=simplified_nodes_mask)

    x_tree = np.zeros(len(parents_list))
    y_tree = np.zeros(len(parents_list))
    z_tree = np.zeros(len(parents_list))
    d_tree = np.zeros(len(parents_list))
    t_tree = np.zeros(len(parents_list))

    x_tree[0] = tree.x[0]
    y_tree[0] = tree.y[0]
    z_tree[0] = tree.z[0]
    d_tree[0] = tree.d[0]
    t_tree[0] = tree.t[0]

    x_tree[1:] = tree.x[indexes[1:]]
    y_tree[1:] = tree.y[indexes[1:]]
    z_tree[1:] = tree.z[indexes[1:]]
    d_tree[1:] = tree.d[indexes[1:]]
    t_tree[1:] = tree.t[indexes[1:]]

    augmented_tree = tmd.Tree.Tree(
        x=x_tree, y=y_tree, z=z_tree, d=d_tree, t=t_tree, p=parents_list
    )

    return augmented_tree


def add_nodes(tree, number_points_to_add=100):
    """Create a new mask of nodes by adding the number of points specified.

    Parameters
    ----------
    tree: tmd.Tree.Tree
        Original Tree (non-simplified one)
    number_points_to_add: int
        Number of points to add from the simplified tree

    Returns
    -------
    augmented_tree: tmd.Tree.Tree
        Augmented tree obtained from the simplified tree + nodes added
    """
    mask_list = get_simplified_nodes_masks(tree)
    add_point = 0
    for _ in range(number_points_to_add):
        test = 0
        while True:
            i = random.randint(1, len(mask_list) - 1)
            test = test + 1
            if test == len(mask_list):
                break
            if mask_list[i] == 0:
                mask_list[i] = 1
                add_point = add_point + 1
                break

    augmented_tree = create_neuron_from_mask(tree, simplified_nodes_mask=mask_list)

    return augmented_tree, add_point


def add_node_specific_radial_dist(
    tree, simplified_nodes_mask=None, desired_value_min=500, desired_value_max=600
):
    """Create a new node mask by adding a point with specific radial distance.

    Parameters
    ----------
    tree: tmd.Tree.Tree
        Original tree (non-simplified one)
    simplified_nodes_mask: list of int
        List of all the nodes that have to be kept
    desired_value_min: int
        Minimum radial distances for the new point
    desired_value_max: int
        Maximum radial distances for the new point

    Returns
    -------
    simplified_nodes_mask: list of int
        List of all the nodes (the input one + the new point)
    add_point: bool
        Boolean indicating if it was possible to add a new point
        in the chosen range or not.
    """
    if simplified_nodes_mask is None:
        simplified_nodes_mask = get_simplified_nodes_masks(tree)
    radial_distances = tree.get_point_radial_distances()
    idx = np.argwhere(
        (desired_value_min < radial_distances) & (radial_distances < desired_value_max)
    )
    j = 0
    add_points = False
    if len(idx) > 1:
        while True:
            i = np.random.randint(0, len(idx) - 1)
            j = j + 1
            if j == len(idx):
                break
            if simplified_nodes_mask[idx[i]] == 0:
                simplified_nodes_mask[idx[i]] = 1
                add_points = True
                break

    return simplified_nodes_mask, add_points


def all_same_number_of_nodes(apicals_list, number_of_nodes=150):
    """Create a new population of trees with a given node count lower bound.

    Parameters
    ----------
    apicals_list: list of tuples (label, tmd.Neuron)
        List of original neurons
    number_of_nodes: int
        Number of nodes that every new trees (apical of the neurons)
        has to have (at least).

    Returns
    -------
    tree_list: list of tmd.Tree.Tree
        List of all the augmented apicals (from the original neurons)
    """
    tree_list = []
    add_point_list = []

    for api in apicals_list:
        neurons_to_add = number_of_nodes - len(api.extract_simplified().x)
        if neurons_to_add > 0:
            api_simplified, add_point = add_nodes(api, neurons_to_add)
        else:
            api_simplified = api.extract_simplified()
            add_point = 0

        tree_list.append(api_simplified)
        add_point_list.append(add_point)

    return tree_list, add_point_list


def radial_dist_histograms_equalization(tree_list_1, tree_list_2, bins):
    """Create a new population with the same radial distance distribution.

    From two populations of trees, create new populations with the same
    radial distances histograms.

    Parameters
    ----------
    tree_list_1: list of tmd.Tree.Tree
        List of the first population
    tree_list_2: list of tmd.Tree.Tree
        List of the second population
    bins: list of int
        Bins used to compare the histograms of the two populations

    Returns
    -------
    new_tree_list_1: list of tmd.Tree.Tree
        List of the apicals of the first population
        (AFTER equalization of histograms).
    new_tree_list_2: list of tmd.Tree.Tree
        List of the apicals of the second population
        (AFTER equalization of histograms).
    mask_list_1: list of int
        List of the nodes kept from the non simplified tree for
        the first list of apicals.
    mask_list_2: list of int
        List of the nodes kept from the non simplified tree for
        the second list of apicals.
    """
    masks_list_1 = []
    masks_list_2 = []

    radial_distances_1 = []
    radial_distances_2 = []

    for api in tree_list_1:
        radial_dist = api.extract_simplified().get_point_radial_distances()
        radial_distances_1.extend(radial_dist)
        masks_list_1.append(get_simplified_nodes_masks(api))

    for api in tree_list_2:
        radial_dist = api.extract_simplified().get_point_radial_distances()
        radial_distances_2.extend(radial_dist)
        masks_list_2.append(get_simplified_nodes_masks(api))

    number1, bins_values1 = np.histogram(radial_distances_1, bins=bins)
    number2, bins_values2 = np.histogram(radial_distances_2, bins=bins)

    for i in range(len(number1)):
        diff = number1[i] - number2[i]
        if diff > 0:
            while True:
                if diff > 0:
                    neuron_nb = random.randint(0, len(tree_list_2) - 1)
                    (
                        masks_list_2[neuron_nb],
                        added_point,
                    ) = add_node_specific_radial_dist(
                        tree_list_2[neuron_nb],
                        simplified_nodes_mask=masks_list_2[neuron_nb],
                        desired_value_min=bins_values2[i],
                        desired_value_max=bins_values2[i + 1],
                    )
                    if added_point:
                        diff = diff - 1
                else:
                    break
        else:
            while True:
                if diff < 0:
                    neuron_nb = random.randint(0, len(tree_list_1) - 1)
                    (
                        masks_list_1[neuron_nb],
                        added_point,
                    ) = add_node_specific_radial_dist(
                        tree_list_1[neuron_nb],
                        simplified_nodes_mask=masks_list_1[neuron_nb],
                        desired_value_min=bins_values1[i],
                        desired_value_max=bins_values1[i + 1],
                    )
                    if added_point:
                        diff = diff + 1

                else:
                    break

    new_tree_list_1 = []
    new_tree_list_2 = []

    for j, api in enumerate(tree_list_2):
        new_tree_list_2.append(create_neuron_from_mask(api, masks_list_2[j]))
    for j, api in enumerate(tree_list_1):
        new_tree_list_1.append(create_neuron_from_mask(api, masks_list_1[j]))

    return new_tree_list_1, new_tree_list_2, masks_list_1, masks_list_2


def radial_dist_hist_multiple_classes(data_path, layer, bins=None):
    """Generate a new population with the same radial distance distribution.

    From multiple populations of neurons, create new populations with the
    same radial distances histograms.

    Parameters
    ----------
    data_path: str
        Path directory with all the population
    layer: int
        Layer to take into account
    bins: list of int, optional
        Bins used to compare the histograms of the different populations.
        The default value is range(0, 1400, 100)

    Returns
    -------
    new_neurons: dict
        Dictionaries containing the augmented apicals after radial
        distance histograms equalization.
    mask_list: dict
        Dictionaries containing the new mask of the augmented apicals
    """
    if bins is None:
        bins = np.arange(0, 1400, 100)
    # Collect all m-types corresponding to the given layer
    m_types_layer = []
    layer_str = f"L{layer}"
    for layer_dir in pathlib.Path(data_path).iterdir():
        if layer_dir.is_dir() and layer_dir.name.startswith(layer_str):
            m_types_layer.append(layer_dir.name)

    neurons: dict[str, list[tmd.Tree.Tree]] = defaultdict()
    radial_distances: dict[str, list[float]] = defaultdict()
    mask_list: dict[str, list[np.ndarray]] = defaultdict()
    bins_values: dict[str, list[np.ndarray]] = defaultdict()
    new_neurons: dict[str, list[tmd.Tree.Tree]] = defaultdict()

    for layer_str in m_types_layer:
        neurons_path = os.listdir(f"{data_path}/{layer_str}")
        neurons[layer_str] = []
        for path in neurons_path:
            tmp_path = f"{data_path}/{layer_str}/{path}"
            if tmp_path.endswith(".h5") or tmp_path.endswith(".swc"):
                tmp_neuron = load_neuron(tmp_path)
                neurons[layer_str].extend(tmp_neuron.apical)

    for type_ in m_types_layer:
        radial_distances[type_] = []
        mask_list[type_] = []
        bins_values[type_] = []
        for api in neurons[type_]:
            radial_dist = api.extract_simplified().get_point_radial_distances()
            radial_distances[type_].extend(radial_dist)
            mask_list[type_].append(get_simplified_nodes_masks(api))
        bins_values[type_], _ = np.histogram(radial_distances[type_], bins=bins)

    max_values = []
    for i in range(len(bins) - 1):
        values = []
        for type_ in m_types_layer:
            values.append(bins_values[type_][i])
        max_values.append(np.max(values))

    for type_ in m_types_layer:
        for i in range(len(max_values)):
            diff = max_values[i] - bins_values[type_][i]
            no_points = 0
            if diff > 0:
                while True:
                    if diff > 0 and no_points < (len(neurons[type_])):
                        neuron_nb = random.randint(0, len(neurons[type_]) - 1)
                        (
                            mask_list[type_][neuron_nb],
                            added_point,
                        ) = add_node_specific_radial_dist(
                            neurons[type_][neuron_nb],
                            simplified_nodes_mask=mask_list[type_][neuron_nb],
                            desired_value_min=bins[i],
                            desired_value_max=bins[i + 1],
                        )
                        if added_point:
                            diff = diff - 1
                            no_points = 0
                        else:
                            no_points += 1
                    else:
                        break
        new_neurons[type_] = []

        for j, api in enumerate(neurons[type_]):
            new_neurons[type_].append(create_neuron_from_mask(api, mask_list[type_][j]))

    return new_neurons, mask_list


def finding_point_on_branch(tree, start, end):
    """Find all the points in a branch specified by its start and end points.

    Parameters
    ----------
    tree: tmd.Tree.Tree
        Tree
    start: int
        Index of the starting point of the branch
    end: int
        Index of the ending point of the branch

    Returns
    -------
    points_on_branch: list
       List on points (specified by their indexes in the non simplified tree)
       on the specified branch.
    """
    points_on_branch = [end]
    point = end
    while True:
        parent = tree.p[point]
        if parent == start:
            points_on_branch.append(start)
            break
        points_on_branch.append(parent)
        point = parent
    return points_on_branch


def nodes_specific_interval_dist(tree, threshold=10):
    """Create a new tree with points respecting a specified interval.

    Parameters
    ----------
    tree: tmd.Tree.Tree
        Tree
    threshold: float
        Max distances between two points

    Returns
    -------
    new_tree: tmd.Tree.Tree
        New tree with nodes spaced by the corresponding threshold
    """
    beg, end = tree.get_sections_2()
    mask = get_simplified_nodes_masks(tree)

    branching_dist = np.sqrt(
        (tree.x[end] - tree.x[beg]) ** 2
        + (tree.y[end] - tree.y[beg]) ** 2
        + (tree.z[end] - tree.z[beg]) ** 2
    )
    for j, dist in enumerate(branching_dist):
        if dist > threshold:
            points_on_branch = finding_point_on_branch(tree, start=beg[j], end=end[j])
            distances = np.sqrt(
                (tree.x[points_on_branch[0:-2]] - tree.x[points_on_branch[1:-1]]) ** 2
                + (tree.y[points_on_branch[0:-2]] - tree.y[points_on_branch[1:-1]]) ** 2
                + (tree.z[points_on_branch[0:-2]] - tree.z[points_on_branch[1:-1]]) ** 2
            )
            dist = 0
            for i in range(len(distances)):
                dist += distances[i]
                if dist > threshold:
                    mask[points_on_branch[i + 1]] = 1
                    dist = 0

    new_tree = create_neuron_from_mask(tree, mask)
    return new_tree

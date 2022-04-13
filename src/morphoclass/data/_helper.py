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
import os
import random
import shutil

import dill
import numpy as np
import scipy.sparse
from tmd.io.io import load_neuron
from tmd.io.io import load_population
from tmd.Topology.analysis import get_persistence_image_data
from tmd.Topology.methods import get_persistence_diagram
from tmd.Tree.Tree import Tree

logger = logging.getLogger(__name__)


def load_apical_persistence_diagrams(folder, mtype):
    """Load persistence diagrams for apicals.

    Loads all neurons from a given directory and extracts persistence
    diagrams from their apical trees.

    Parameters
    ----------
    folder
        Folder containing m-type folders.
    mtype
        M-type name, must be a subfolder of 'folder' and contain neuron files.


    Returns
    -------
    List of apical persistence diagrams of all neurons in the m-type folder.
    """
    path = os.path.join(folder, mtype)
    population = load_population(path)
    persistence_diagrams = []

    for neuron in population.neurons:
        if len(neuron.apical) != 1:
            raise ValueError(
                f"Was expecting exactly one apical tree, found: {len(neuron.apical)}"
            )

        persistence_diagrams.append(get_persistence_diagram(neuron.apical[0]))

    return persistence_diagrams


def augment_persistence_diagrams(
    persistence_diagrams, xlims, ylims, factor=10, maxd=2, p=0.5
):
    """Augment persistence diagrams.

    Try the naive approach: for each persistence diagram randomly
    select a subset of nodes and shift them slightly. Choose the shifts
    randomly so that in the resulting persistence image the points shift
    by at most `maxd`.

    :param persistence_diagrams:
    :return:
    """
    augmented_diagrams = []
    dx = (xlims[1] - xlims[0]) / 100.0 * (maxd + 0.5)
    dy = (ylims[1] - ylims[0]) / 100.0 * (maxd + 0.5)

    for diagram in persistence_diagrams:
        augmented_diagrams.append(diagram)
        for _ in range(factor - 1):
            new_diagram = []
            for x, y in diagram:
                if random.random() < p:
                    new_diagram.append(
                        [
                            x + (random.random() * 2 - 1) * dx,
                            y + (random.random() * 2 - 1) * dy,
                        ]
                    )
                else:
                    new_diagram.append([x, y])
            augmented_diagrams.append(new_diagram)

    return augmented_diagrams


def augment_persistence_diagrams_v2(
    persistence_diagrams, labels, xlims, ylims, factor=10, maxd=2, p=0.5
):
    """Augment persistence diagrams.

    Try the naive approach: for each persistence diagram randomly
    select a subset of nodes and shift them slightly. Choose the shifts
    randomly so that in the resulting persistence image the points shift
    by at most `maxd`.

    :param persistence_diagrams:
    :return:
    """
    if len(persistence_diagrams) != len(labels):
        raise ValueError(
            "The length of the data and the labels must match. "
            f"Found len(data)={len(persistence_diagrams)}, "
            f"len(labels)={len(labels)}"
        )

    augmented_diagrams = []
    augmented_labels = []
    dx = (xlims[1] - xlims[0]) / 100.0 * (maxd + 0.5)
    dy = (ylims[1] - ylims[0]) / 100.0 * (maxd + 0.5)

    for diagram, label in zip(persistence_diagrams, labels):
        # Append the original diagram and label
        augmented_diagrams.append(diagram)
        augmented_labels.append(label)

        # Create and append augmented diagram and label
        for _ in range(factor - 1):
            new_diagram = []
            for x, y in diagram:
                if random.random() < p:
                    new_diagram.append(
                        [
                            x + (random.random() * 2 - 1) * dx,
                            y + (random.random() * 2 - 1) * dy,
                        ]
                    )
                else:
                    new_diagram.append([x, y])
            augmented_diagrams.append(new_diagram)
            augmented_labels.append(label)

    return augmented_diagrams, augmented_labels


def persistence_diagrams_to_persistence_images(
    persistence_diagrams, xlims=None, ylims=None
):
    """
    Convert a persistence diagrams to persistence images.

    :param persistence_diagrams: persistence diagram to convert
    :param xlims: the x-dimension of the persistence images to create
    :param ylims: the y-dimension of the persistence images to create
    :return: an numpy array with the create persistence images
    """
    return np.array(
        [
            get_persistence_image_data(diagram, xlims=xlims, ylims=ylims)
            for diagram in persistence_diagrams
        ]
    )


def reduce_tree_to_branching(tree):
    """Simplifies a neurite tree to the braning points only.

    An analogous methods already exists and is `Tree.extract_simplified()`.
    One should use that method instead of using this one.

    Parameters
    ----------
    tree
        Input neurite tree.

    Returns
    -------
    Simplified neurite tree.
    """
    adj = tree.dA
    assert adj[0].sum() == 0, "Was expecting node 0 to be the root, but it has parents"

    mask = tree.get_bif_term() != 1  # bif_term = number of children
    mask[0] = True  # include the root as well...
    idx = np.argwhere(mask).squeeze()
    adj_new = scipy.sparse.lil_matrix(adj.shape, dtype=adj.dtype)

    # For each node in idx find the next branching ancestor and set it as the new parent
    parents = adj.argmax(axis=1).getA1()
    for i in idx[1:]:  # skip the root
        p = parents[i]
        while (
            p not in idx
        ):  # this could end up in an infinite loop for bad data, avoid?
            p = parents[p]
        adj_new[i, p] = 1

    adj_new = adj_new[np.ix_(mask, mask)].tocsr()
    parents_new = adj_new.argmax(axis=1).getA1()
    parents_new[0] = -1

    return Tree(
        x=tree.x[mask],
        y=tree.y[mask],
        z=tree.z[mask],
        d=tree.d[mask],
        t=tree.t[mask],
        p=np.array(parents_new),
    )


def pickle_data(data_dir, rewrite=False):
    """Cache neuron data-set by pickling every neuron.

    This is helpful because loading each neuron each time can take
    a considerable amount of time. The given neuron files are loaded
    as `tmd.Neuron` instances and pickled into files on disk.

    Note that the `Dataset` class is written in such a way that it is able
    to both process `.h5` and `.swc` files, as well as `.pickle` data
    created by this function.

    Parameters
    ----------
    data_dir
        Directory from which to load the data-set. The structure should be:
            data_dir/
                mtype1/
                    neuron1
                    neuron2
                    ...
                mtype2/
                    neuron1
                    neuron2
                    ...
    rewrite
        * If False and the target directory already exists then an error is raised.
        * If True the existing target directory is deleted and new pickled data
          is created.

    Returns
    -------
    The target directory to which the data was written.

    Raises
    ------
    FileExistsError
        File with output dataset exists.
    """
    # Were we given a valid directory?
    if not os.path.isdir(data_dir):
        raise ValueError(f'"{data_dir}" is not a directory')

    # Does the target directory already exist and should we overwrite it?
    pickle_dir = data_dir + "_pickled"
    if os.path.isdir(pickle_dir):
        if not rewrite:
            raise FileExistsError(
                f'"{pickle_dir}" already exists, if you want to rewrite the '
                "data then delete this directory manually first or call this "
                "function with argument rewrite=True"
            )
        else:
            shutil.rmtree(pickle_dir)
    os.mkdir(pickle_dir)

    valid_ext = [".swc", ".h5"]
    for mtype in sorted(os.listdir(data_dir)):
        subdir = os.path.join(data_dir, mtype)
        apicals_subdir = os.path.join(pickle_dir, mtype)
        if not os.path.isdir(subdir):
            continue
        else:
            os.mkdir(apicals_subdir)

        print("Loading data in {subdir}")

        for sample in sorted(os.listdir(subdir)):
            sample_path = os.path.join(subdir, sample)
            if os.path.isdir(sample_path):
                continue
            file_name, ext = os.path.splitext(sample)
            if ext not in valid_ext:
                continue

            try:
                neuron = load_neuron(sample_path)
            except Exception as e:
                print(
                    f'Loading neuron "{sample_path}" failed with the '
                    "following exception:"
                )
                print(e)
                continue

            out_file_name = file_name + ".pickle"
            with open(os.path.join(apicals_subdir, out_file_name), "wb") as file_handle:
                dill.dump(neuron, file_handle)

    return pickle_dir

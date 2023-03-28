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
"""Miscellaneous utilities."""
from __future__ import annotations

import contextlib
import datetime
import logging
import os
import pathlib
import platform
import sys
from contextlib import contextmanager

import dill
import networkx as nx
import numpy as np
import pkg_resources
import scipy.sparse
import torch
from morphio import PointLevel
from morphio import SectionType
from morphio.mut import Morphology
from neurom import COLS
from sklearn import model_selection
from tmd.io.io import load_neuron
from tmd.Neuron import Neuron
from tmd.Soma import Soma
from tmd.Tree import Tree
from tmd.utils import TREE_TYPE_DICT
from torch_geometric.data import Data
from torch_geometric.data import DataLoader

import morphoclass as mc

logger = logging.getLogger(__name__)

TYPE_DCT = {v: k for k, v in TREE_TYPE_DICT.items()}
morphio_type_to_tmd_type = {
    SectionType.soma: TYPE_DCT["soma"],
    SectionType.axon: TYPE_DCT["axon"],
    SectionType.basal_dendrite: TYPE_DCT["basal_dendrite"],
    SectionType.apical_dendrite: TYPE_DCT["apical_dendrite"],
    SectionType.undefined: -1,
}


def print_message_to_stderr(header, msg):
    """Print a message to stderr.

    Parameters
    ----------
    header : str
        A header to prepend to the message.
    msg : str
        The message.
    """
    print(f"{header}: {msg}", file=sys.stderr)


def print_warning(msg):
    """Print a warning message."""
    print_message_to_stderr("WARNING", msg)


def print_error(msg):
    """Print an error message."""
    print_message_to_stderr("ERROR", msg)


def no_print(function):
    """Decorate a function to suppress all output to stdout."""

    def wrapper(*args, **kwargs):
        old_stdout = sys.stdout
        with open(os.devnull, "w") as fh_null:
            sys.stdout = fh_null
            ret = function(*args, **kwargs)
            sys.stdout = old_stdout
        return ret

    return wrapper


@contextmanager
def suppress_print(suppress_err=False):
    """Get a context manager for suppressing output to stdout."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    with open(os.devnull, "w") as fh_null:
        sys.stdout = fh_null
        if suppress_err:
            sys.stderr = fh_null
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def save_var(var, file_name, exist_ok=False):
    """Serialize a given variable into a file.

    Parameters
    ----------
    var : obj
        The variable to be pickled.
    file_name : str or Path
        The filename where to write the variable.
    exist_ok : bool
        If set to False existing files won't be overwritten and an exception
        will be raised instead.

    Raises
    ------
    FileExistsError
        File already exists.
    """
    if not exist_ok and os.path.exists(file_name):
        raise FileExistsError(
            f'File "{file_name}" already exits, set parameter '
            '"exist_ok" to True to overwrite it'
        )

    with open(file_name, "wb") as handle:
        dill.dump(var, handle, protocol=dill.HIGHEST_PROTOCOL)


def load_var(file_name):
    """Load a serialized variable from the given file.

    Parameters
    ----------
    file_name : str
        The file from which to load the variable.
    """
    with open(file_name, "rb") as handle:
        var = dill.load(handle)
    return var


def from_morphio(root_section):
    """Convert a MorphIO root section to a TMD tree.

    Parameters
    ----------
    root_section
        A MorphIO root section.

    Returns
    -------
    tree : tmd.Tree.Tree.Tree
        A TMD tree.
    """
    sections = list(root_section.iter())
    points = np.vstack([section.points for section in sections])
    diameters = np.hstack([section.diameters for section in sections])
    types = np.hstack([[section.type] * len(section.points) for section in sections])

    counter = -1
    parents = []
    section_last_point_id: dict[int, int] = {}
    for section in sections:
        for i, _ in enumerate(section.points):
            parent_id = counter
            counter += 1
            if i == 0:
                parent_id = (
                    -1 if section.is_root else section_last_point_id[section.parent.id]
                )
            parents.append(parent_id)
        section_last_point_id[section.id] = counter
    return Tree.Tree(
        points[:, 0], points[:, 1], points[:, 2], diameters, types, parents
    )


def from_morphio_to_tmd(morphio_neuron, remove_duplicates=False):
    """Change a neuron in MorphIO format to TMD neuron format.

    Parameters
    ----------
    morphio_neuron
        A MorphIO neuron.
    remove_duplicates : bool
        Whether or not to remove duplicate points at section joints.

    Returns
    -------
    tmd_neuron
        A TMD neuron.
    """
    tmd_neuron = Neuron.Neuron()

    tmd_neuron.set_soma(
        Soma.Soma(
            x=morphio_neuron.soma.points[:, COLS.X],
            y=morphio_neuron.soma.points[:, COLS.Y],
            z=morphio_neuron.soma.points[:, COLS.Z],
            d=2 * morphio_neuron.soma.points[:, COLS.R],
        )
    )

    for root in morphio_neuron.root_sections:
        # tmd_neuron.append_tree(from_morphio(root), td)  # previous version
        tree = morphio_root_section_to_tmd_tree(
            root, remove_duplicates=remove_duplicates
        )
        tmd_neuron.append_tree(tree, TREE_TYPE_DICT)
    return tmd_neuron


def find_point_on_branch(tree: Tree, start: int, end: int) -> list[int]:
    """Find all the points in a branch specified by its start and end points.

    Parameters
    ----------
    tree
        A TMD tree.
    start
        Index of the starting point of the branch
    end
        Index of the ending point of the branch

    Returns
    -------
    list
        A list of indices of points on the given branch. The indices refer
        to the non-simplified tree.
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


def from_tmd_to_morphio(tmd_neuron):
    """Change a neuron in TMD format to MorphIO neuron format.

    Parameters
    ----------
    tmd_neuron
        A TMD neuron.

    Returns
    -------
    A MorphIO neuron.
    """
    morpho_neuron = Morphology()
    morpho_neuron.soma.points = np.transpose(
        [tmd_neuron.soma.x, tmd_neuron.soma.y, tmd_neuron.soma.z]
    )
    morpho_neuron.soma.diameters = tmd_neuron.soma.d

    branches = []
    branches_type = []
    branches.extend(tmd_neuron.axon)
    branches_type.extend([SectionType.axon for _ in range(len(tmd_neuron.axon))])
    branches.extend(tmd_neuron.basal)
    branches_type.extend(
        [SectionType.basal_dendrite for _ in range(len(tmd_neuron.basal))]
    )
    branches.extend(tmd_neuron.apical)
    branches_type.extend(
        [SectionType.apical_dendrite for _ in range(len(tmd_neuron.apical))]
    )

    for nb_branch, branch in enumerate(branches):
        beg, end = branch.get_sections_2()
        section = []
        for nb, (i, j) in enumerate(zip(beg, end)):
            if nb == 0:
                points = np.flip(find_point_on_branch(branch, i, j))
                section.append(
                    morpho_neuron.append_root_section(
                        PointLevel(
                            np.transpose(
                                [branch.x[points], branch.y[points], branch.z[points]]
                            ).tolist(),
                            branch.d[points],
                        ),
                        branches_type[nb_branch],
                    )
                )
            if nb > 0:
                ind = np.where(end == i)
                points = np.flip(find_point_on_branch(branch, i, j))
                section.append(
                    section[ind[0][0]].append_section(
                        PointLevel(
                            np.transpose(
                                [branch.x[points], branch.y[points], branch.z[points]]
                            ).tolist(),
                            branch.d[points],
                        )
                    )
                )

    return morpho_neuron


def morphio_root_section_to_tmd_tree(root_section, remove_duplicates=True):
    """Create a TMD tree class from a MorphIO root section.

    Parameters
    ----------
    root_section
        A MorphIO root section
    remove_duplicates : bool
        If true then duplicate points at section joins are removed.

    Returns
    -------
    tree : tmd.Tree.Tree.Tree
        A TMD tree.
    """

    def accumulate_sections(section):
        """Accumulate all data from a section and its children.

        Recursively traverse through all children of the given section
        and collect the node coordinates, diameters, node types, and
        parent indices.

        Given section's data is combined with the children's data
        and returned

        Parameters
        ----------
        section
            A MorphIO section.

        Returns
        -------
        tree : tmd.Tree.Tree.Tree
            A tree object with the accumulated data.
        """
        # Extract data from the given section
        # To avoid duplicate points at section joins we remove the first
        # points of all sections except the root section itself
        if remove_duplicates and not section.is_root:
            points = section.points[1:]
            diameters = section.diameters[1:]
        else:
            points = section.points
            diameters = section.diameters
        n_pts = len(points)
        types = np.array([morphio_type_to_tmd_type[section.type]] * n_pts)
        parents = np.arange(n_pts) - 1

        # Go through all children of the given section, recursively extract
        # their data and append it to the existing data
        for child in section.children:
            c_points, c_diameters, c_types, c_parents = accumulate_sections(child)

            # Parent indices need to be shifted according to the current
            # position
            c_parents += len(parents)

            # Make the child section properly attach to the parent section
            c_parents[0] = n_pts - 1

            # Concatenated child's data with parent's data
            points = np.concatenate([points, c_points])
            diameters = np.concatenate([diameters, c_diameters])
            types = np.concatenate([types, c_types])
            parents = np.concatenate([parents, c_parents])

        return points, diameters, types, parents

    pts, d, t, p = accumulate_sections(root_section)
    x, y, z = pts.T
    tree = Tree.Tree(x=x, y=y, z=z, d=d, t=t, p=p)

    return tree


def tree_to_graph(tree):
    """Convert a TMD tree to a networkx graph.

    Parameters
    ----------
    tree : tmd.Tree.Tree.Tree

    Returns
    -------
    networkx.Graph
        The graph representation of the tree.
    """
    coordinates = np.transpose([tree.x, tree.y, tree.z])
    assert np.sum(tree.p == -1) == 1
    coordinates -= coordinates[tree.p == -1]

    graph = nx.Graph()
    for idx in range(len(tree.p)):
        graph.add_node(
            idx, pos=coordinates[idx], pos2d=coordinates[idx, :2], d=tree.d[idx]
        )

    for child, parent in enumerate(tree.p):
        if parent != -1:
            graph.add_edge(child, parent)

    return graph


def are_you_sure():
    """Interactive prompt asking "Are you sure? (y/N)".

    Returns
    -------
    bool
        True if the user answered with "y", otherwise False.
    """
    return input("Are you sure? (y/N)").lower() == "y"


def _fuse_simplified_bitufted_apicals(apical_1, apical_2):
    """Fuse two apical trees into one.

    Fuses two apical trees into one and extracts radial distances
    and the adjacency matrix.

    First the apical trees are simplified to branching nodes only.

    The adjacency matrices of both apical trees are combined into a
    big block matrix resulting in a disjoint graph. Since radial distances
    don't contain the orientation of the original tree, the distances of the
    lower-lying tree get a negative sign to distinguish them from the
    higher-lying tree.

    Parameters
    ----------
    apical_1:
        First apical tree
    apical_2:
        Second apical tree

    Returns
    -------
    Radial distances and the adjacency matrix of the combined tree

    """
    simp_1 = apical_1.extract_simplified()
    simp_2 = apical_2.extract_simplified()

    # Compute mean projections wrt the y-axis, then order the apicals
    # So that the first one is pointing more upwards than the second one
    proj_1 = simp_1.get_point_projection().mean()
    proj_2 = simp_2.get_point_projection().mean()
    if proj_2 > proj_1:
        simp_1, simp_2 = simp_2, simp_1

    # Extract radial distances and adjacency matrices
    rad_1 = simp_1.get_point_radial_distances()
    rad_2 = simp_2.get_point_radial_distances()

    adj_1 = simp_1.dA
    adj_2 = simp_2.dA

    radial_distances = np.concatenate([rad_1, -rad_2])
    adj = scipy.sparse.bmat(blocks=[[adj_1, None], [None, adj_2]], format="coo")

    return radial_distances, adj


def get_stratified_split(dataset, val_size=None, random_state=None):
    """Get indices for a stratified train/val split.

    Parameters
    ----------
    dataset : morphoclass.data.MorphologyDataset
        The dataset to split.
    val_size : flaot
        The relative size of the validation set.
    random_state
        A random state for the `StratifiedShuffleSplit` class.

    Returns
    -------
    train_idx : list
        The indices of the training set.
    val_idx
        The indices of the validation set.
    """
    all_labels = [data.y for data in dataset]

    sss = model_selection.StratifiedShuffleSplit(
        test_size=val_size, random_state=random_state
    )
    split_gen = sss.split(X=all_labels, y=all_labels)
    train_idx, val_idx = next(split_gen)

    train_idx = [int(x) for x in train_idx]
    val_idx = [int(x) for x in val_idx]

    return train_idx, val_idx


# TODO: the function names don't reflect the fact that
#       we're extracting radial_distances from the neurons
def read_apical_from_file(path, label, nodes_features=None, inverted_apical=True):
    """Read the neuron data from a file and constructs a torch Data object.

    We extract the simplified apical trees, and use the apical distances
    as features.

    Parameters
    ----------
    path
        Path to the neuron file. Supported file types are
        .h5, .swc, and .pickle.
    label
        The label to be attached to the Data class.
    nodes_features
        List of the nodes features.
    inverted_apical : boolean
        If True, radial distances are inverted if the mean of the
        projections is negative.

    Returns
    -------
    sample
        A sample of the type Data.
    """
    # Check if the given path is a file
    assert os.path.isfile(path), f"The given {path} is not pointing to a valid file"

    # Check if the given file is a neuron file or a pickled binary
    if path.endswith(".h5") or path.endswith(".swc"):
        neuron = load_neuron(path)
    elif path.endswith(".pickle"):
        with open(path, "rb") as file_handle:
            neuron = dill.load(file_handle)
        # not a neuron file
        if not hasattr(neuron, "name"):
            return None
    else:
        return None

    # Check how many apicals the neuron has, and handle them accordingly
    # Extract radial distances and the adjacency matrix.
    if not hasattr(neuron, "apical") or len(neuron.apical) == 0:
        raise ValueError(f"Neuron {neuron.name} has no apical tree data")
    elif len(neuron.apical) == 2:
        # TODO: Implement feature extraction for bitufted cells as well,
        #       at the moment the is not handled at all! Basically one needs
        #       to do the same as in the `else` block below.
        #       Actually, the whole feature extraction should be implemented
        #       in a more streamlined way as a pipe/chain of transformers
        #       and should be able to treat the general case.
        radial_distances, adj = _fuse_simplified_bitufted_apicals(*neuron.apical)
        all_features = torch.from_numpy(radial_distances)
        all_features = all_features.to(torch.float32).unsqueeze(dim=1)
    elif len(neuron.apical) > 2:
        raise NotImplementedError(
            "Handling of more than two apical trees is not supported"
        )
    else:
        tree = neuron.apical[0].extract_simplified()
        if isinstance(nodes_features, list):
            if len(nodes_features) > 0:
                for num_node, node in enumerate(nodes_features):
                    if node == "radial_dist":
                        radial_distances = tree.get_point_radial_distances()
                        # If the cell is inverted, then make the radial
                        # distances negative to indicate that
                        if tree.get_point_projection().mean() < 0 and inverted_apical:
                            radial_distances = -radial_distances

                        features = torch.tensor(
                            radial_distances, dtype=torch.float32
                        ).unsqueeze(dim=1)

                    elif node == "coordinates":
                        coordinates = [
                            (tree.x - tree.x[0]),
                            (tree.y - tree.y[0]),
                            (tree.z - tree.z[0]),
                        ]
                        # coordinates = [tree.x, tree.y, tree.z]
                        features = torch.tensor(coordinates, dtype=torch.float32).t()

                    elif node == "vertical_dist":
                        vertical_dist = [(tree.y - tree.y[0])]
                        features = torch.tensor(vertical_dist, dtype=torch.float32).t()

                    elif node == "path_dist":
                        distances = neuron.apical[0].get_point_path_distances()
                        starts_points, ends_points = neuron.apical[0].get_sections_2()
                        path_lengths = np.zeros([len(ends_points) + 1])
                        path_lengths[0] = distances[0]
                        path_lengths[1:] = distances[ends_points]
                        features = torch.tensor(
                            path_lengths, dtype=torch.float32
                        ).unsqueeze(dim=1)

                    elif node == "angles":
                        beg, end = tree.get_sections_2()
                        angles = []
                        angles.extend([0, 0])
                        for i in range(len(beg) - 1):
                            u = tree.get_direction_between(beg[i], end[i])
                            v = tree.get_direction_between(beg[(i + 1)], end[(i + 1)])
                            c = np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)
                            angles.append(np.arccos(c))
                        features = torch.tensor(angles, dtype=torch.float32).unsqueeze(
                            dim=1
                        )

                    if num_node == 0:
                        all_features = features
                    else:
                        all_features = torch.cat((all_features, features), 1)

        else:
            radial_distances = tree.get_point_radial_distances()

            # If the cell is inverted, then make the radial distances
            # negative to indicate that
            if tree.get_point_projection().mean() < 0 and inverted_apical:
                radial_distances = -radial_distances
            all_features = torch.tensor(
                radial_distances, dtype=torch.float32
            ).unsqueeze(dim=1)

        adj = tree.dA

    # Construct the sample

    adj = adj.tocoo()
    # Directed graph with direction away from soma
    # This is because in adjacency matrix rows = children, cols = parents
    sources = torch.tensor(adj.col, dtype=torch.int64)
    targets = torch.tensor(adj.row, dtype=torch.int64)

    # Make undirected
    # sources, targets = torch.cat([sources, targets]), torch.cat([targets, sources])

    edge_index = torch.stack([sources, targets])

    # `tree` isn't always initialized...
    # x = torch.tensor(tree.x)
    # y = torch.tensor(tree.y)
    # z = torch.tensor(tree.z)

    sample = Data(
        x=all_features,
        y=label,
        # pos=torch.stack([x, y, z], dim=1),
        edge_index=edge_index,
    )

    return sample


def read_layer_neurons_from_dir(
    data_dir, layer, labels_dic=None, nodes_features=None, inverted_apical=True
):
    """Read all neurons corresponding to a given layer.

    Parameters
    ----------
    data_dir
        Directory containing subdirectories L[layer]_... with neuron files.
    layer
        The layer for which to load the samples.
    inverted_apical : boolean
        If True, radial distances are inverted if the mean of the
        projections is negative.
    labels_dic : dict
        A mapping from integers to layer names.
    nodes_features
        A list of the nodes features.

    Returns
    -------
    samples
        A list of samples, each of type Data.
    labels
        A dictionary for assigning integers to layer names.
    paths
        A list of the corresponding paths to the neuron stored in samples.
    """
    data_dir = pathlib.Path(data_dir)
    assert data_dir.is_dir(), "The given directory doesn't exist"
    if labels_dic is None:
        layers = [
            x.name for x in data_dir.iterdir() if x.name == f"L{layer}" and x.is_dir()
        ]
        labels = dict(
            enumerate(
                sorted(
                    f"{layer}/{x.name}"
                    for layer in layers
                    for x in (data_dir / layer).iterdir()
                )
            )
        )

    else:
        labels = labels_dic.copy()

    samples = []
    paths = []
    for label, layer in labels.items():
        layer_dir = data_dir / layer
        if layer_dir.exists():
            print(f"Reading files in {layer}")
            for file in sorted(layer_dir.iterdir()):
                sample = read_apical_from_file(
                    path=str(file),
                    label=label,
                    nodes_features=nodes_features,
                    inverted_apical=inverted_apical,
                )
                if sample is not None:
                    samples.append(sample)
                    paths.append(str(file))
                else:
                    print_warning(f"Could not read morphology: {file}")

    return samples, labels, paths


def normalize_features(samples, max_values=None):
    """Normalize all sample features by their maximum.

    Parameters
    ----------
    samples : iterable
        A list of samples to be normalized.
    max_values : float or torch.Tensor
        A list of the max values features_wise.

    Returns
    -------
    samples
        The normalized samples.
    max_values
        The max values that were used for normalization.
    """
    all_x = [torch.abs(sample.x) for sample in samples]

    if max_values is None:
        max_values = torch.cat(all_x, dim=0).max(dim=0).values

    for sample in samples:
        sample.x /= max_values

    return samples, max_values


def get_loader(samples, idx, batch_size=None):
    """Create a DataLoader from a list of samples and a given index.

    Parameters
    ----------
    samples
        A list of samples.
    idx
        A list of indices that specifies which samples should be included.
    batch_size : int, optional
        Number of training samples per batch. By default a full batch will be
        used.

    Returns
    -------
    torch_geometric.data.DataLoader
        A DataLoader object.
    """
    sel = [samples[i] for i in idx]
    if batch_size is None:
        batch_size = len(idx)

    return DataLoader(sel, batch_size=batch_size)


def make_torch_deterministic(make_torch_traceback_nicer=True):
    """Make GPU computations using torch deterministic.

    Note that in some cases this may have a performance impact. For
    further information see
    https://pytorch.org/docs/stable/notes/randomness.html

    Parameters
    ----------
    make_torch_traceback_nicer : bool, optional
        make CUDA generated error tracebacks ("device-side assert triggered")
        more explicit. Useful for debugging.

    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if make_torch_traceback_nicer:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def warn_if_nondeterministic(device):
    """Create a warning if PyTorch is not in deterministic mode.

    Parameters
    ----------
    device : str or torch.device
        The currently used device for which to check the determinism.
    """
    if isinstance(device, torch.device):
        device = device.type

    if isinstance(device, str) and device.lower() == "cuda":
        if (
            torch.backends.cudnn.deterministic is False
            or torch.backends.cudnn.benchmark is True
        ):
            logger.warning("Torch is in a nondeterministic CUDA mode.")


def str_time_delta(time_delta, short=True):
    """Create a nice string representation of a time delta.

    The time delta is rounded down to whole seconds.

    Parameters
    ----------
    time_delta : float
        The time delta. Can be computed as the difference of
        results of two calls of `time.time_perf()`.
    short : bool
        Switch between a short and a long string representation.

    Returns
    -------
    time_str : str
        The string representation of the time delta.
    """
    time_delta = int(time_delta)
    n_days = time_delta // (24 * 60 * 60)
    time_delta %= 24 * 60 * 60
    n_hours = time_delta // (60 * 60)
    time_delta %= 60 * 60
    n_minutes = time_delta // 60
    n_seconds = time_delta % 60

    if short:
        time_tokens = [f"{n_days}d", f"{n_hours}h", f"{n_minutes}min", f"{n_seconds}s"]
        separator = " "
    else:
        time_tokens = [
            f"{n_days} days",
            f"{n_hours} hours",
            f"{n_minutes} minutes",
            f"{n_seconds} seconds",
        ]
        separator = ", "

    if n_days + n_hours + n_minutes == 0:
        time_tokens = time_tokens[3:]
    elif n_days + n_hours == 0:
        time_tokens = time_tokens[2:]
    elif n_days == 0:
        time_tokens = time_tokens[1:]

    time_str = separator.join(time_tokens)
    return time_str


@contextlib.contextmanager
def np_temp_seed(seed):
    """Contextmanager for setting a temporary numpy seed.

    Parameters
    ----------
    seed : int
        The temporary numpy seed.
    """
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def convert_morphologies(input_dir, old_ext, new_ext, notebook=False):
    """Convert a collection of morphologies to a new format.

    All files matching the old extension will be read recursively
    from the input directory and its subdirectories. A new output
    directory will then be created by appending and underscore
    and the new extension to the input directory name, e.g.
    the directory "my_files" will become "my_files_swc" if the
    new extension is "swc".

    Note that there are no checks if the output directory
    already exists and files might be overwritten in the case
    that it does!

    Parameters
    ----------
    input_dir : str or pathlib.Path
        The directory containing the input morphologies. All
        subdirectories will be processed recursively.
    old_ext : str
        The file extension to convert from. Only files with this
        extension will be considered.
    new_ext : str
        The new extension to convert to. Must be a file extension
        supported by MorphIO.
    notebook : bool
        If true then a widget progress bar will be displayed
        instead of an ASCII one.
    """
    if new_ext.startswith("."):
        new_ext = new_ext[1:]

    input_dir = pathlib.Path(input_dir)
    if not input_dir.exists():
        logger.warning("Input path does not exist. Nothing to do.")
        return

    output_dir = input_dir / ".." / (input_dir.name + "_" + new_ext)
    output_dir = output_dir.resolve()
    print(f"Input directory  : {input_dir.resolve()}")
    print(f"Output directory : {output_dir.resolve()}")
    print(f"Old extension    : {old_ext}")
    print(f"New extension    : {new_ext}")

    if notebook:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
    pbar = tqdm(sorted(input_dir.rglob("*" + old_ext)))

    for neuron_file in pbar:
        neuron_name = neuron_file.stem
        neuron_folder = neuron_file.relative_to(input_dir).parent
        pbar.set_postfix(file=f"{neuron_folder} : {neuron_name}")
        neuron_file_new = output_dir / neuron_folder / f"{neuron_name}.{new_ext}"
        neuron_file_new.parent.mkdir(parents=True, exist_ok=True)
        neuron = Morphology(str(neuron_file))
        neuron.write(str(neuron_file_new))


def add_metadata(checkpoint):
    """Add metadata information to the checkpoint.

    The metadata information includes
    - A timestamp in ISO format
    - The name of the package (should be "morphoclass")
    - The version of the package
    - Installed packages

    Parameters
    ----------
    checkpoint : dict
        The checkpoint dictionary.
    """
    this_package, *_ = __name__.partition(".")
    installed_packages = []
    for item in pkg_resources.working_set:
        if item.key != this_package:
            installed_packages.append(f"{item.project_name}=={item.version}")
    checkpoint["metadata"] = {
        "timestamp": datetime.datetime.today().isoformat(timespec="seconds"),
        "package": this_package,
        "version": mc.__version__,
        "installed_packages": sorted(installed_packages),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }


def dict2kwargs(d):
    """Convert method parameters dict to keyword string for the HTML report.

    Parameters
    ----------
    d : dict, str
        Dictionary to convert or empty string if no parameters are available.

    Returns
    -------
    str
        String of dictionary shown as keywords.
    """
    if isinstance(d, dict):
        return ", ".join(f"{k}={v}" for k, v in d.items())
    else:
        return str(d)

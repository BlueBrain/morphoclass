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
"""Utilities for GNN models. Currently, only the hierarchical labels."""
from __future__ import annotations

import warnings

import numpy as np
import torch

from morphoclass.utils import print_error


class HierarchicalLabels:
    """Class representing a set of hierarchical labels.

    It holds the structural information of a set of hierarchical labels
    and is meant to be used conjointly with an appropriate hierarchical
    model.

    The constructor constructs class instance from a given label adjacency
    matrix.

    It takes labels for all hierarchical nodes and their
    hierarchical relationship in terms of the adjacency matrix.
    Additionally a dictionary mapping flat labels (=leaf nodes)
    to the hierarchical one-hot labels must be provided.

    Parameters
    ----------
    labels
        List of labels for all hierarchical nodes.
    adj
        Adjacency matrix representing the label hierarchy.
    flat_to_hierarchical_oh
        dictionary mapping flat labels to hierarchical one-hot labels.

    Returns
    -------
    instance
        An instance of the `HierarchicalLabels` class.
    """

    def __init__(self, labels, adj, flat_to_hierarchical_oh):
        self._labels = np.array(labels)
        self._flat_to_hierarchical_oh = flat_to_hierarchical_oh
        self._adj = adj
        self._roots: np.ndarray | None = None

    @staticmethod
    def _print_assignment_error():
        print_error(
            "manually changing class attributes is not allowed. "
            "Instead, create a new instance of class."
        )

    @property
    def roots(self):
        """Get the label roots.

        Returns
        -------
        The label roots.
        """
        if self._roots is None:
            self._roots = self._adj.sum(axis=1) == 0
            # self._roots = self._roots.astype(np.int)
        return self._roots

    @property
    def adj(self):
        """Get the adjacency matrix for hierarchical labels.

        Returns
        -------
        The adjacency matrix representing hierarchical labels.
        """
        return self._adj

    @adj.setter
    def adj(self, value):
        """Set the adjacency matrix for hierarchical labels (not allowed).

        Parameters
        ----------
        value
            The new value.
        """
        self._print_assignment_error()

    @property
    def labels(self):
        """Get the labels.

        Returns
        -------
        The labels.
        """
        return self._labels

    @labels.setter
    def labels(self, value):
        """Set the labels (not allowed).

        Parameters
        ----------
        value
            The new labels.
        """
        self._print_assignment_error()

    @classmethod
    def from_adjacency_matrix(cls, labels, adj, flat_to_hierarchical_oh):
        """Construct class instance from a given label adjacency matrix.

        This factory takes labels for all hierarchical nodes and their
        hierarchical relationship in terms of the adjacency matrix.
        Additionally a dictionary mapping flat labels (=leaf nodes)
        to the hierarchical one-hot labels must be provided.

        Parameters
        ----------
        labels
            List of labels for all hierarchical nodes.
        adj
            Adjacency matrix representing the label hierarchy.
        flat_to_hierarchical_oh
            dictionary mapping flat labels to hierarchical one-hot labels.

        Returns
        -------
        instance
            An instance of the `HierarchicalLabels` class.
        """
        warnings.warn(
            "This factory method is deprecated, use the class constructor "
            "directly with the same arguments instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls(labels, flat_to_hierarchical_oh, adj)

    @classmethod
    def from_flat_labels(cls, label_dict):
        """Construct a class instance from a label dictionary.

        This factory expects a dictionary mapping numerical flat labels,
        i.e. leaf nodes of the hierarchical tree, to the list of
        hierarchical nodes representing the path from the root of the
        hierarchy to the given leaf node.

        For example, a valid representation for the following hierarchy

        .. code-block:: text

              A      B
             / |   /  |
            a  b  c   d
                 /|\
                x y z

        would be given by

        .. code-block:: python

            label_dict = {
                0: ['A', 'a'],
                1: ['A', 'b'],
                2: ['B, 'c', 'x'],
                3: ['B, 'c', 'y'],
                4: ['B, 'c', 'z'],
                5: ['B', 'd'],
            }

        Parameters
        ----------
        label_dict : dict
            Dictionary mapping dense numerical flat labels to a list of
            hierarchical string labels.
        """
        # Construct list of nodes
        nodes_inv = {}
        n = 0
        for idx in label_dict:
            for node in label_dict[idx]:
                if node not in nodes_inv:
                    nodes_inv[node] = n
                    n += 1
        nodes = {v: k for k, v in nodes_inv.items()}

        # Construct adjacency matrix
        n = len(nodes)
        adj = np.zeros((n, n), dtype=int)
        for hie in label_dict.values():
            for node, parent in zip(hie[1:], hie):
                adj[nodes_inv[node], nodes_inv[parent]] = 1

        # Construct the flat-to-hierarchical dictionary
        flat_to_hierarchical_oh = {
            n: np.zeros(len(nodes), dtype=int) for n in label_dict
        }
        for n, ls in label_dict.items():
            which_nodes = [nodes_inv[label] for label in ls]
            flat_to_hierarchical_oh[n][which_nodes] = 1

        labels = [nodes[i] for i in range(len(nodes))]

        return cls(labels, adj, flat_to_hierarchical_oh)

    # Not keeping this as we would not be able to infer the dictionary
    # between the flat and hierarchical labels
    # @classmethod
    # def from_parent_list(cls, labels, parents):
    #     """
    #     Construct hierarchical labels from parents list.
    #     Obviously works only for trees, but not for DAGs, but we don't want
    #     to handle DAGs anyway
    #
    #     Args:
    #         labels: any hashable representation of labels (eg string)
    #         parents: indices of parents
    #     Returns:
    #
    #     """
    #     n = len(labels)
    #     assert n == len(parents)
    #
    #     # Construct adjacency matrix
    #     adj = np.zeros((n, n), dtype=np.int)
    #     for child, parent in enumerate(parents):
    #         if parent >= 0:
    #             adj[child, parent] = 1
    #
    #     # Construct the flat-to-hierarchical dictionary
    #     # TODO
    #
    #     return cls.from_adjacency_matrix(labels, adj)
    #
    # @classmethod
    # def from_flat_labels_alt(cls, label_dict):
    #     """
    #     Initialise attributes from given label_dict
    #
    #     The argument `label_dict` is a dictionary mapping numerical labels
    #     to a list of strings representing the hierarchical labels. For
    #     example `['animal', 'mammal', 'dog', 'Corgi']` would represent
    #     the hierarchical label for 'Corgi', which is a subclass of 'dog',
    #     which is a subclass of 'mammal', which is a subclass of 'animal'.
    #
    #     Args:
    #         label_dict (dict): dictionary mapping dense numerical labels
    #                            to a list of hierarchical string labels.
    #     """
    #     parents = []
    #     labels = []
    #     one_hot_dict = dict()
    #
    #     # Construct `parents` and `labels`
    #     seen_at = dict()
    #     for _, item_labels in sorted(label_dict.items()):
    #         parent = -1
    #         for label in item_labels:
    #             if label not in seen_at:
    #                 # = index where this label is going to be
    #                 seen_at[label] = len(parents)
    #                 parents.append(parent)
    #                 labels.append(label)
    #             parent = seen_at[label]
    #
    #     # Create one_hot_labels
    #     for n in label_dict:
    #         oh_label = np.zeros(len(parents)).byte()
    #         oh_label[[seen_at[l] for l in label_dict[n]]] = 1
    #         one_hot_dict[n] = oh_label
    #
    #     return cls.from_parent_list(labels, parents)

    def gen_class_masks(self, parent_mask=None):
        """Generate masks for labels probabilities of which should sum to 1.

        Parameters
        ----------
        parent_mask
            The labels to start with, defaults to root nodes.

        Yields
        ------
        mask
            All possible masks for nodes with sums of probabilities equal to 1.
        """
        if parent_mask is None:
            parent_mask = self.roots

        parent_idx = parent_mask.nonzero()[0]
        if len(parent_idx) == 0:
            return
        else:
            yield parent_mask
            for idx in parent_idx:
                child_mask = self.adj[:, idx] == 1
                yield from self.gen_class_masks(child_mask)

    def get_class_segmentation_mask(self):
        """Create a segmentation mask for labels.

        The mast is such that each softmax block is marked by a different
        integer.

        Returns
        -------
        segmentation_mask
            The class segmentation mask.
        """
        segmentation_mask = np.zeros(len(self), dtype=int)
        for n, mask in enumerate(self.gen_class_masks()):
            segmentation_mask += n * mask

        return segmentation_mask

    def flat_to_hierarchical_oh(self, n):
        """Map flat numerical labels to hierarchical one-hot labels.

        Parameters
        ----------
        n
            Flat numerical label.

        Returns
        -------
        oh_label
            One-hot hierarchical label.
        """
        return self._flat_to_hierarchical_oh[n]

    def n_leafs(self):
        """Get the number of leaf nodes.

        This is the same as the number of flat labels.

        Returns
        -------
        The number of leaf nodes.
        """
        return len(self._flat_to_hierarchical_oh)

    def flat_labels(self):
        """Flat labels in numerical form.

        Returns
        -------
        list
            The numerical values of the flat labels.
        """
        return list(self._flat_to_hierarchical_oh.keys())

    def __repr__(self):
        """Compute the repr."""
        return f"{self.__class__.__name__}(roots={self.labels[self.roots]})"

    def __len__(self):
        """Get the number of labels."""
        return len(self.labels)


class HierarchicalLabelsDeprecated:
    """Class representing a set of hierarchical labels.

    The attributes are initialized from given label_dict

    The argument `label_dict` is a dictionary mapping numerical labels
    to a list of strings representing the hierarchical labels. For
    example `['animal', 'mammal', 'dog', 'Corgi']` would represent
    the hierarchical label for 'Corgi', which is a subclass of 'dog',
    which is a subclass of 'mammal', which is a subclass of 'animal'.

    Parameters
    ----------
    label_dict : dict
        Dictionary mapping dense numerical labels to a list of hierarchical
        string labels.
    """

    def __init__(self, label_dict):
        self.labels = []
        self.one_hot_dict = {}

        # Construct `tree_idx` and `labels`
        tree_idx: list[int] = []
        seen_at = {}
        for _, item_labels in sorted(label_dict.items()):
            parent = -1
            for label in item_labels:
                if label not in seen_at:
                    seen_at[label] = len(
                        tree_idx
                    )  # = index where this label is going to be
                    tree_idx.append(parent)
                    self.labels.append(label)
                parent = seen_at[label]

        # Create one_hot_labels
        for n in label_dict:
            oh_label = torch.zeros(len(tree_idx)).byte()
            oh_label[[seen_at[label] for label in label_dict[n]]] = 1
            self.one_hot_dict[n] = oh_label

        self.tree_idx = torch.tensor(tree_idx)

    def to_one_hot(self, dense_label):
        """Convert dense labels to one-hot labels.

        Parameters
        ----------
        dense_label
            Given dense labels.

        Returns
        -------
        One-hot labels.
        """
        return self.one_hot_dict[dense_label]

    @staticmethod
    def is_tree(tree_idx):
        """Check if the given `parents` tensor represents a valid tree.

        Start with the parents and iteratively visit the children, and
        mark the visited nodes. Continue until all nodes are visited.
        If at one iteration step the number of visited nodes does not
        increase (so there's a disconnected loop) or a child is found
        to have been seen before, then `parents` does not represent a
        tree.

        Parameters
        ----------
        tree_idx
            The tree index.

        Returns
        -------
        bool
            Whether or not the given tree index represents a valid tree.
        """
        # sanity check: parent indices must be valid or (-1);
        # actually we don't need to check this explicitly since
        # this is automatically check by what follows. So this
        # might only be useful for performance reasons, if at all.
        valid_idx = torch.arange(-1, tree_idx.size(0))
        match = tree_idx.unsqueeze(1) == valid_idx.unsqueeze(0)
        if not torch.all(torch.any(match, dim=1)).item():
            return False

        # start with the roots
        seen = parent_mask = tree_idx < 0

        # loop until have seen all nodes
        while not torch.all(seen):
            # extract indices of the parents
            parent_index = torch.nonzero(parent_mask, as_tuple=False)
            # children are those that point to parents
            child_mask = torch.any(tree_idx == parent_index, dim=0)
            # add children to seen
            new_seen = seen | child_mask
            # if the no new nodes were seen then it's not a tree
            if torch.equal(new_seen, seen):
                return False
            # otherwise continue
            seen = new_seen
            # and the children become the new parents
            parent_mask = child_mask

        return True

    def get_mask(self, order=0):
        """Get a mask for the nodes corresponding to a given hierarchy order.

        The order 0 in the tree hierarchy is the root.

        Actually all of the non-trivial masks are already
        computed in `is_tree` so if performance starts to matter
        one could extend `is_tree` to cache the `child_mask` and
        return.
        """
        if order == 0:
            return self.tree_idx < 0
        else:
            mask_prv = self.get_mask(order - 1)
            idx_prv = torch.nonzero(mask_prv, as_tuple=False)
            return torch.sum(self.tree_idx == idx_prv, dim=0).type(mask_prv.dtype)

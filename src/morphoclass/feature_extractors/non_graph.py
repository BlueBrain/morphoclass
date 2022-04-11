"""Non-graph feature extraction such as persistence diagrams and images."""
from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Literal

import numpy as np
from tmd.Topology.methods import get_persistence_diagram
from tmd.Tree.Tree import Tree

from morphoclass import deepwalk

logger = logging.getLogger(__name__)


def get_tmd_diagrams(
    neurite_collection: Iterable[Iterable[Tree]],
    feature: Literal["projection", "radial_distances"],
) -> list[np.ndarray]:
    """Convert neurites to TMD diagrams.

    Parameters
    ----------
    neurite_collection
        A collection of neurites extracted from a population of neurons. The
        outer iterable is over the neurons, the inner iterables are over
        the neurites extracted from the given neuron.
    feature
        The kind of feature to extract. For more details on the features
        please see `tmd.Topology.methods.get_persistence_diagram`.

    Returns
    -------
    list[np.ndarray]
        A list of extracted persistence diagrams, one for each neuron.
    """
    diagrams = []
    for neurites in neurite_collection:
        diagram = []
        for tree in neurites:
            diagram.extend(get_persistence_diagram(tree, feature))
        diagrams.append(np.array(diagram))

    return diagrams


def get_deepwalk_diagrams(
    neurite_collection: Iterable[Iterable[Tree]],
) -> list[np.ndarray]:
    """Convert neurites to deepwalk diagrams.

    Parameters
    ----------
    neurite_collection
        A collection of neurites extracted from a population of neurons. The
        outer iterable is over the neurons, the inner iterables are over
        the neurites extracted from the given neuron.

    Returns
    -------
    list[np.ndarray]
        A list of extracted persistence diagrams, one for each neuron.
    """
    deepwalk.warn_if_not_installed()

    kwargs = {
        "representation_size": 2,
        "max_memory_data_size": 1000000000,
        "walk_length": 20,
        "number_walks": 10,
        "seed": 0,
        "undirected": True,
        "window_size": 5,
        "workers": 1,
    }

    diagrams = []
    for neurites in neurite_collection:
        diagram = []
        for tree in neurites:
            diagram.append(deepwalk.get_embedding(tree, **kwargs))
        diagrams.append(np.concatenate(diagram))

    return diagrams

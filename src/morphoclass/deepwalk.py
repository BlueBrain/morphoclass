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
"""Portal to DeepWalk methods from morphoclass."""
from __future__ import annotations

import logging
import random
import warnings

logger = logging.getLogger(__name__)


def check_installed() -> bool:
    """Check whether the DeepWalk package is installed.

    Returns
    -------
    bool
        Whether the DeepWalk package is installed
    """
    try:
        import deepwalk  # noqa: F401
    except ImportError:
        return False
    else:
        return True


def how_to_install_msg() -> str:
    """Get installation instructions for DeepWalk.

    Returns
    -------
    str
        The instructions on how to install DeepWalk.
    """
    return (
        "To install the version <version> of DeepWalk run "
        '"pip install deepwalk==<version>". morphoclass was tested using '
        'DeepWalk version "1.0.3". To install the latest version of DeepWalk '
        'run "pip install deepwalk". Some versions of DeepWalk will '
        'erroneously install the package "futures". This package is not '
        "needed and its presence might cause errors. We recommend "
        'uninstalling it via "pip uninstall futures".'
    )


def warn_if_not_installed() -> None:
    """Issue a UserWarning if deepwalk is not installed."""
    if not check_installed():
        warnings.warn(
            f"DeepWalk is not installed. {how_to_install_msg()}",
            stacklevel=3,
        )


def _tree2graph(tree, undirected=True):
    """Convert neuron tree into graph.

    Parameters
    ----------
    tree : tmd.Tree.Tree.Tree
        TMD tree of the neuron.
    undirected : bool
        Whether the graph is directed or not. Default is undirected.

    Returns
    -------
    G : deepwalk.graph.Graph
        The graph of neuronal tree.
    """
    from deepwalk import graph

    size = len(tree.x)
    deepwalk_graph = graph.Graph()
    for seg_id in range(1, size):
        par_id = tree.p[seg_id]
        seg_id = int(seg_id)
        par_id = int(par_id)
        deepwalk_graph[seg_id].append(par_id)
        if undirected:
            deepwalk_graph[par_id].append(seg_id)

    deepwalk_graph.make_consistent()
    return deepwalk_graph


def get_embedding(
    tree,
    undirected,
    number_walks,
    walk_length,
    max_memory_data_size,
    seed,
    representation_size,
    window_size,
    workers,
):
    """Learn representations for tree vertices in graphs using DeepWalk.

    Parameters
    ----------
    tree : tmd.Tree.Tree.Tree
        TMD tree of the neuron.
    undirected : bool
        Treat graph as undirected.
    number_walks : int
        Number of random walks to start at each node.
    walk_length : int
        Length of the random walk started at each node.
    max_memory_data_size : int
        Size to start dumping walks to disk, instead of keeping them in memory.
    seed : int
        Seed for random walk generator.
    representation_size : int
        Number of latent dimensions to learn for each node.
    window_size : int
        Window size of skipgram model.
    workers : int
        Number of parallel processes.

    Returns
    -------
    numpy.ndarray
        Embedded vectors.

    References
    ----------
    See file:
    - https://github.com/phanein/deepwalk/blob/\
        6e6dff245e4692e9bea47e9017c1034e51afbf29/deepwalk/__main__.py
    in case max_memory_data_size is ever crossed.

    Raises
    ------
    NotImplementedError
        The part was not implemented since we don't have big graphs size
        that are not able to fit into the memory.
    """
    from deepwalk import graph
    from gensim.models import Word2Vec

    deepwalk_graph = _tree2graph(tree, undirected=undirected)
    logger.debug(f"Number of nodes: {len(deepwalk_graph.nodes())}")
    num_walks = len(deepwalk_graph.nodes()) * number_walks
    logger.debug(f"Number of walks: {num_walks}")
    data_size = num_walks * walk_length
    logger.debug(f"Data size (walks*length): {data_size}")

    if data_size < max_memory_data_size:
        logger.debug("Walking...")
        walks = graph.build_deepwalk_corpus(
            deepwalk_graph,
            num_paths=number_walks,
            path_length=walk_length,
            rand=random.Random(seed),
        )
        logger.debug("Training...")
        model = Word2Vec(
            walks,
            vector_size=representation_size,
            window=window_size,
            min_count=0,
            sg=1,
            hs=1,
            workers=workers,
        )
    else:
        raise NotImplementedError(
            "Please check the reference link to see how to support the"
            "data_size > max_memory_data_size"
        )

    return model.wv.vectors

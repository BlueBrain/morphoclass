"""Implementation of the `plot_node_saliency` function."""
from __future__ import annotations

import networkx as nx
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


def plot_node_saliency(
    tree,
    grads,
    ax=None,
    rot=0,
    scale=1.0,
    edge_color="orange",
    width=1,
    center_nodes=True,
    show_axes=False,
    show_legend=True,
    name="grad",
):
    """Plot the node salience for a neuronal tree.

    Parameters
    ----------
    tree : tmd.Tree.Tree.Tree
        The neuronal tree to plot.
    grads : array
        The node saliency is the absolute value of the gradients. Therefore
        the length of the grads sequence should be equal to the number of nodes.
    ax : matplotlib.axes.Axes (optional)
        Plotting axes.
    rot : float (optional)
        Rotate the neuronal tree by the given angle around the y-axis. The angle
        is in degrees.
    scale : float (optional)
        Change the thickness of the plotted graph edges.
    edge_color : str (optional)
        The color of the edges. Is passed through to NetworkX
    width : float (optional)
        Change the thickness of the plotted graph edges.
    center_nodes : bool (optional)
        If true then the tree will be shifted so that the first node (usually
        the root of the tree) is at the coordinate origin.
    show_axes : bool  (optional)
        If true then show the coordinate axes.
    show_legend : bool (optional)
        If true then show the plot legend.
    name : str (optional)
        The name of the saliency that will appear in the plot legend.
    """
    if len(tree.x) != len(grads):
        raise ValueError(
            "The number of nodes and gradients differ. "
            f"({len(tree.x)} != {len(grads)})"
        )

    if ax is None:
        fig = Figure()
        ax = fig.subplots()
    # Construct graph
    g = nx.Graph()
    g.add_nodes_from(range(len(tree.p)))
    for parent, child in enumerate(tree.p[1:], 1):
        g.add_edge(parent, child)

    # Construct rotation matrix
    alpha = rot * np.pi / 180
    rot_mat = np.array(
        [
            [np.cos(alpha), 0, -np.sin(alpha)],
            [0, 1, 0],
            [np.sin(alpha), 0, np.cos(alpha)],
        ]
    ).T

    # Collect node coordinates, set the tree root to the origin, and rotate by rot_mat
    coordinates = np.stack([tree.x, tree.y, tree.z])
    if center_nodes:
        coordinates -= coordinates[:, 0:1]
    coordinates = rot_mat @ coordinates
    node_positions = {i: tuple(coord) for i, coord in enumerate(coordinates[:2].T)}
    node_sizes = np.abs(grads)
    node_sizes = node_sizes * 200
    node_colors = ["blue" if x > 0 else "red" for x in grads]

    nx.draw(
        g,
        ax=ax,
        node_color=node_colors,
        edge_color=edge_color,
        node_size=node_sizes,
        pos=node_positions,
        width=width * scale,
        with_labels=False,
    )

    if show_axes:
        # this is set by `nx.draw`, undo it
        ax.set_axis_on()
        ax.tick_params(
            axis="both",
            which="both",
            bottom=True,
            left=True,
            labelbottom=True,
            labelleft=True,
        )
    if show_legend:
        label_1 = f"{name} > 0"
        label_2 = f"{name} < 0"
        color_1 = "blue"
        color_2 = "red"
        handle_1 = Line2D(
            [0], [0], marker="o", color="w", label=label_1, markerfacecolor=color_1
        )
        handle_2 = Line2D(
            [0], [0], marker="o", color="w", label=label_2, markerfacecolor=color_2
        )
        ax.legend(handles=[handle_1, handle_2])

"""Functions for visualizing data."""
from __future__ import annotations

import logging
import re
from typing import Sequence

import matplotlib.axes
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import tmd.Topology
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

logger = logging.getLogger(__name__)


def plot_tree(
    tree,
    ax,
    rot=0,
    scale=1.0,
    edge_color="green",
    node_size=0.0,
    width=1,
    center_nodes=True,
    show_axes=False,
):
    """Plot a tmd tree class as a 2D projection.

    Parameters
    ----------
    tree
        Tree data in form of a `tmd.Tree` class.
    ax
        Pyplot axes object.
    rot
        Rotation angle around the vertical axis in degrees.
    scale
        Overall scale of the plot.
    edge_color
        Color of the tree edges.
    node_size
        Size of the tree nodes.
    width
        Width of the tree branches.
    center_nodes : boo, optional
        Move all nodes so that the first node is at the origin.
    show_axes : bool, optional
        Show / hide plot axes.
    """
    if isinstance(edge_color, str):
        edge_color = [edge_color] * len(tree.p)
    if isinstance(width, (int, float)):
        width = [width] * len(tree.p)

    # Construct graph
    g = nx.Graph()
    g.add_nodes_from(range(len(tree.p)))
    for (parent, child), color, weight in zip(
        enumerate(tree.p[1:], start=1), edge_color, width
    ):
        g.add_edge(parent, child, color=color, weight=weight * scale)

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
    node_positions = dict(enumerate(zip(*coordinates[:2])))

    node_sizes = [node_size * scale] * len(g.nodes)
    node_sizes[0] *= 4

    node_colors = ["black"] * len(g.nodes)
    node_colors[0] = "red"

    edges = g.edges()
    colors = [g[u][v]["color"] for u, v in edges]
    weights = [g[u][v]["weight"] for u, v in edges]
    kwargs = {
        "ax": ax,
        "node_color": node_colors,
        "edge_color": colors,
        "node_size": node_sizes,
        "pos": node_positions,
        "width": weights,
        "with_labels": False,
    }

    nx.draw(g, **kwargs)

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


def plot_mean_ci(
    ax, values, label=None, color=None, weight=1, zorder=None, show_ci=True
):
    """Plot a cross-validation learning curve.

    the values of the evaluation metric should be stored in the array
    values with the dimension (n_splits, n_epochs)

    Parameters
    ----------
    ax
        A matplotlib axis object.
    values
        The metric evaluation values.
    label
        The label of the curve.
    color
        The color of the curve.
    weight
        Determines the thickness of lines and transparency.
    zorder
        Z-order of the curve to be plotted.
    show_ci : bool
        Whether to show the CIs.

    Returns
    -------
    color
        The color of the plotted curves.
    """
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    x = np.arange(len(mean))
    line_kwargs = {"zorder": zorder, "linewidth": 1.0 * weight}

    (base_line,) = ax.plot(x, mean, label=label, color=color, **line_kwargs)
    color = base_line.get_color()

    if show_ci:
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.05 * weight)
        ax.plot(x, mean + std, color=color, alpha=0.1 * weight, **line_kwargs)
        ax.plot(x, mean - std, color=color, alpha=0.1 * weight, **line_kwargs)

    return color


def plot_acc_cv(values, label=None, ax=None, title=None):
    """Plot cross-validated accuracy scores.

    Parameters
    ----------
    values : np.ndarray
        Accuracy values to be plotted. The array should be of shape
        (n_splits, n_epochs).
    label : str, optional
        The label which should appear on the legend, provided that the ax
        object hast the legend enabled.
    ax : matplotlib.axes.Axes, optional
        Matplotlib Axes object to plot onto.
    title : str, optional
        The title to be displayed above the axes.
    """
    if ax is None:
        fig = Figure(figsize=(7, 4))
        ax = fig.subplots()
    else:
        fig = None
    if title is not None:
        ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Epoch")
    plot_mean_ci(ax, values, label=label)
    return fig


def plot_learning_curves(
    acc,
    loss,
    val_acc,
    val_loss,
    color_acc="green",
    color_loss="orange",
    title=None,
    file_name=None,
):
    """Plot the learning curves for the train and validation sets.

    Parameters
    ----------
    acc : list, numpy.ndarray
        Train accuracy.
    loss : list, numpy.ndarray
        Train loss.
    val_acc : list, numpy.ndarray
        Validation accuracy.
    val_loss : list, numpy.ndarray
        Validation loss.
    color_acc : str
        Colour for the accuracy curves.
    color_loss : str
        Colour for the loss curves.
    title : str
        Title of the plot.
    file_name : str, pathlib.Path
        Path to where to store the plot.

    Returns
    -------
    matplotlib.figure.Figure
        Figure of the plot.
    """
    acc = np.array(acc) if isinstance(acc, list) else acc
    loss = np.array(loss) if isinstance(loss, list) else loss
    val_acc = np.array(val_acc) if isinstance(val_acc, list) else val_acc
    val_loss = np.array(val_loss) if isinstance(val_loss, list) else val_loss

    plot_losses = len(loss) > 0

    # Create the figure
    fig = Figure(figsize=(12, 12))
    ax_train_acc, ax_val_acc = fig.subplots(nrows=2, ncols=1)

    # Plot the training set results
    ax_train_acc.set_title("Training set")
    ax_train_acc.set_ylim(0, 1)
    ax_train_acc.set_yticks(np.linspace(0, 1, 11))
    ax_train_acc.set_xlabel("Epoch")
    ax_train_acc.set_ylabel("Trainig Accuracy")

    if plot_losses:
        ax_train_loss = ax_train_acc.twinx()
        ax_train_loss.set_ylim(0, 6)
        ax_train_loss.set_ylabel("Training Loss")

    plot_mean_ci(ax_train_acc, acc, "train acc", color_acc)
    if plot_losses:
        plot_mean_ci(ax_train_loss, loss, "train loss", color_loss)

    ax_train_acc.grid(color="gray", linestyle=":", linewidth=1)
    ax_train_acc.legend(loc="upper right")
    if plot_losses:
        ax_train_loss.legend(loc="lower right")

    # Plot the validation set results
    ax_val_acc.set_title("Validation set")
    ax_val_acc.set_ylim(0, 1)
    ax_val_acc.set_yticks(np.linspace(0, 1, 11))
    ax_val_acc.set_xlabel("Epoch")
    ax_val_acc.set_ylabel("Validation Accuracy")

    if plot_losses:
        ax_val_loss = ax_val_acc.twinx()
        ax_val_loss.set_ylim(0, 6)
        ax_val_loss.set_ylabel("Validation Loss")

    plot_mean_ci(ax_val_acc, val_acc, "val acc", color_acc)
    if plot_losses:
        plot_mean_ci(ax_val_loss, val_loss, "val loss", color_loss)

    ax_val_acc.grid(color="gray", linestyle=":", linewidth=1)
    ax_val_acc.legend(loc="upper right")
    if plot_losses:
        ax_val_loss.legend(loc="lower right")

    fig.suptitle(title)
    if file_name:
        fig.savefig(file_name)

    return fig


def plot_accuracies(
    acc,
    val_acc,
    color_train="blue",
    color_val="red",
    title=None,
    file_name=None,
    supported_ext=None,
):
    """Plot the learning curves for the train and validation sets.

    Parameters
    ----------
    acc : list, numpy.ndarray
        Train accuracy.
    val_acc : list, numpy.ndarray
        Validation accuracy.
    color_train : str, default "blue"
        Colour for the training curves.
    color_val : str, default "red"
        Colour for the validation curves.
    title : str, optional
        Title of the plot.
    file_name : str, pathlib.Path, optional
        Path to where to store the plot.
    supported_ext : list, optional
        Supported image extensions. If default, it will save as ".png", ".eps", ".pdf".

    Returns
    -------
    matplotlib.figure.Figure
        Figure of the plot.
    """
    acc = np.array(acc) if isinstance(acc, list) else acc
    val_acc = np.array(val_acc) if isinstance(val_acc, list) else val_acc

    # Create the figure
    fig = Figure(figsize=(7, 4), dpi=75)
    ax = fig.subplots()

    # Plot the training set results
    ax.set_title("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")

    plot_mean_ci(ax, acc, "train dataset", color_train)
    plot_mean_ci(ax, val_acc, "validation dataset", color_val)

    ax.grid(color="gray", linestyle=":", linewidth=1)
    ax.legend()
    ax.set_rasterized(True)
    fig.suptitle(title)
    if file_name:
        if not supported_ext:
            supported_ext = [".png", ".eps", ".pdf"]
        for ext in supported_ext:
            fig.savefig(file_name.with_suffix(ext))

    return fig


def plot_learning_acc(history, title=None, file_name=None):
    """Plot training/validation accuracies.

    Parameters
    ----------
    history : dict
        Dictionary with keys 'train_acc' and 'val_acc' and the corresponding
        values list_like objects with accuracy values.
    title : str, optional
        The title for the plot. If None then the default title
        "Learning Curves" will be used.
    file_name : str
        Path to where to save the plot.
    """
    with sns.axes_style("whitegrid"):
        train_acc = history["train_acc"]
        val_acc = history["val_acc"]

        fig = Figure(figsize=(3, 2))
        ax = fig.subplots()
        ax.set_title(
            title or "Learning curves",
        )
        ax.plot(train_acc, label="Training")
        ax.plot(val_acc, label="Validation")
        ax.legend()
        ax.set_ylim([0, 1])
        ax.set_xlim([0, len(train_acc)])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.yaxis.set_major_locator(MaxNLocator(11))
        fig.tight_layout()
        if file_name:
            fig.savefig(file_name)
        return fig


def plot_confusion_matrix(cm, file_name, labels, supported_ext=None):
    """Confusion matrix.

    Parameters
    ----------
    cm : np.ndarray
        The confusion matrix values.
    file_name : str, pathlib.Path
        File name where to save the CM plot.
    labels : list
        List of labels for prediction.
    """
    fig = Figure(dpi=75)
    ax = fig.subplots()

    sns.heatmap(cm, annot=True, ax=ax, xticklabels=labels, yticklabels=labels)
    ax.invert_yaxis()
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_rasterized(True)
    fig.tight_layout()

    if file_name:
        if not supported_ext:
            supported_ext = [".png", ".eps", ".pdf"]
        for ext in supported_ext:
            fig.savefig(file_name.with_suffix(ext))
    return fig


def get_color_map(labels: Sequence[str]) -> dict[str, tuple[int, int, int]]:
    """Map labels to colors in the default color palette.

    In matplotlib, what is the sequence of colors that will be used
    when multiple plots are created? This function gives access to this
    sequence in form of a map "given label" -> "color".

    Parameters
    ----------
    labels
        A sequence of arbitrary labels.

    Returns
    -------
    dict:
        A map from the input labels to colors. The colors are represented
        by triples of floats between 0 and 1.
    """
    if not len(labels) == len(set(labels)):
        logger.warning("Labels are not unique, duplicates will be dropped")
    palette = sns.color_palette(n_colors=len(labels))
    color_map = {label: palette[i] for i, label in enumerate(labels)}

    return color_map


def plot_counts_per_subclass(
    mtypes: pd.Series,
    figsize: tuple[float, float] = (10, 6),
) -> Figure:
    """Make a bar plot with the count of samples per m-type.

    The m-type labels should contain the layer prefix so that the
    bars in the plot can be grouped by layer.

    Parameters
    ----------
    mtypes
        A pandas series with the m-type labels of all samples. The labels
        are expected to contain the layer information in the form of the
        prefix "LXX_" or "XX_" where XX are an arbitrary number of digits.
    figsize
        The size of the figure

    Returns
    -------
    Figure
        The plot figure.
    """
    if not all(mtypes.map(lambda mtype: re.match(r"L?\d+_", mtype))):
        raise ValueError("All values must start with LXX_ where XX is the layer number")
    counts = mtypes.value_counts().sort_index().rename("Sample Count")
    layers = [mtype.partition("_")[0] for mtype in counts.index]

    color_map = get_color_map(sorted(set(layers)))
    colors = [color_map[layer] for layer in layers]

    fig = Figure(figsize=figsize, tight_layout=True)
    ax = fig.subplots()
    ax.set_title("Samples per M-Type")
    ax.tick_params(axis="x", labelrotation=90)
    sns.barplot(x=counts.index, y=counts, ax=ax, palette=colors)
    for i, value in enumerate(counts):
        ax.text(i, value, str(value), va="bottom", ha="center")

    return fig


def plot_counts_per_layer(
    mtypes: pd.Series,
    figsize: tuple[float, float] = (10, 6),
) -> Figure:
    """Make a bar plot with the count of samples per cortical layer.

    Parameters
    ----------
    mtypes
        A pandas series with the m-type labels of all samples. The labels
        are expected to contain the layer information in the form of the
        prefix "LXX_" or "XX_" where XX are an arbitrary number of digits.
    figsize
        The size of the figure

    Returns
    -------
    Figure
        The plot figure.
    """
    if not all(mtypes.map(lambda mtype: re.match(r"L?\d+_", mtype))):
        raise ValueError("All values must start with LXX_ where XX is the layer number")
    counts = mtypes.map(lambda mtype: mtype.partition("_")[0]).value_counts()
    counts = counts.sort_index().rename("Sample Count")

    fig = Figure(figsize=figsize, tight_layout=True)
    ax = fig.subplots()
    ax.set_title("Samples per Layer")
    sns.barplot(x=counts.index, y=counts, ax=ax)
    for i, value in enumerate(counts):
        ax.text(i, value, str(value), va="bottom", ha="center")

    return fig


def plot_histogram(
    title: str,
    data: Sequence[float],
    figsize: tuple[float, float] = (6, 4),
) -> Figure:
    """Make a histogram plot.

    Parameters
    ----------
    title
        The figure title.
    data
        The data to plot.
    figsize
        The size of the figure.

    Returns
    -------
    The figure with the histogram.
    """
    fig = Figure(figsize=figsize, tight_layout=True)
    ax = fig.subplots()
    ax.set_title(title, fontweight="bold")
    ax.hist(data, rwidth=0.9)
    text = f"Ø = {np.mean(data):.0f} ± {np.std(data):.0f}"
    ax.text(0.6, 0.7, text, transform=ax.transAxes)
    ax.tick_params(axis="x", labelrotation=45)

    return fig


def plot_histogram_panel(
    title: str,
    labeled_data: dict[str, Sequence[float]],
    figsize: tuple[float, float] = (12, 4),
) -> Figure:
    """Make a panel of histogram plots for each labeled data sequence.

    Parameters
    ----------
    title
        The figure title.
    labeled_data:
        Map of some labels to the corresponding data. The data is a
        sequence of floats.
    figsize
        The figure size.

    Returns
    -------
    Figure
        The plot.
    """
    fig = Figure(figsize=figsize, tight_layout=True)
    fig.suptitle(title, fontweight="bold")
    if len(labeled_data) == 1:
        axs = [fig.subplots()]
    else:
        axs = fig.subplots(ncols=len(labeled_data))
    color_map = get_color_map(sorted(labeled_data))

    for ax, (label, data) in zip(axs, sorted(labeled_data.items())):
        ax.hist(data, rwidth=0.6, color=color_map[label], label=label)
        ax.set_title(f"Ø = {np.mean(data):.0f} ± {np.std(data):.0f}")
        ax.tick_params(axis="x", labelrotation=45)
        ax.legend()

    return fig


def plot_number_of_nodes(
    k,
    neurite_and_nodes,
    rtype,
    imagedir,
    figsize=(6, 4),
    supported_ext=None,
):
    """Plot number of nodes in the graph.

    Parameters
    ----------
    k : str
        The neurite type: "axon", "basal", 'apical', "neurites".
    neurite_and_nodes : dict
        Dictionary with neurite types as keys, and graph/PD nodes as values.
    rtype : str
        Can have 2 values:0 'graph' or 'PD'.
    imagedir : pathlib.Path
        The path where to store the generated plot.
    figsize : tuple, default (6, 4)
        The figsize for the matplotlib figure.
    supported_ext : list, optional
        The extensions to store the images. By default it will store them
        as ".png", ".eps", ".pdf".

    Returns
    -------
    pathlib.Path
        Path where the image is stored.
    """
    with sns.axes_style("whitegrid"):
        fig = Figure(figsize=figsize, dpi=75)
        ax = fig.subplots()

        ax.set_title(
            f"Number of nodes in the {k} {rtype}",
            fontweight="bold",
        )
        ax.hist(neurite_and_nodes[k], rwidth=0.9)

        props = {"boxstyle": "round", "facecolor": "white", "alpha": 0.5}
        type_exists = len(neurite_and_nodes[k]) > 0 and not np.isnan(
            np.mean(neurite_and_nodes[k])
        )

        if type_exists:
            text = (
                f"Ø = {np.mean(neurite_and_nodes[k]):.0f} ± "
                + f"{np.std(neurite_and_nodes[k]):.0f}"
            )
            ax.text(
                0.6,
                0.7,
                text,
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=props,
            )
            ax.tick_params(axis="both")
        else:
            text = f"the {k} tree not available"
            ax.text(
                0.25,
                0.5,
                text,
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=props,
            )
            ax.tick_params(axis="both")

        ax.tick_params(axis="x", labelrotation=45)
        ax.set_rasterized(True)
        # fig.tight_layout()

        image_path = imagedir / f"number_of_nodes_{rtype.replace(' ', '_')}_{k}"
        if image_path:
            if not supported_ext:
                supported_ext = [".png", ".eps", ".pdf"]
            for ext in supported_ext:
                fig.savefig(image_path.with_suffix(ext))

        return image_path.with_suffix(".png")


def plot_number_of_nodes_per_layer(
    k,
    layer_neurite_and_nodes,
    rtype,
    imagedir,
    figsize=(12, 4),
    supported_ext=None,
):
    """Plot number of nodes in the graph per layer.

    Parameters
    ----------
    k : str
        The neurite type: "axon", "basal", 'apical', "neurites".
    layer_neurite_and_nodes:
        Nested dictionary with neurite types as first-level keys,
        layers as second-level keys, and graph/PD nodes as values.
    rtype : str
        Can have 2 values:0 'graph' or 'PD'.
    imagedir : pathlib.Path
        The path where to store the generated plot.
    figsize : tuple, default (12, 4)
        The figsize for the matplotlib figure.
    supported_ext : list, optional
        The extensions to store the images. By default it will store them
        as ".png", ".eps", ".pdf".

    Returns
    -------
    pathlib.Path
        Path where the image is stored.
    """
    with sns.axes_style("whitegrid"):
        fig = Figure(figsize=figsize, dpi=75)
        axs = fig.subplots(1, len(layer_neurite_and_nodes[k]))

        fig.suptitle(
            f"Number of nodes in the {k} {rtype} for individual layers",
            y=0.95,
            fontweight="bold",
        )

        cmap = sns.color_palette("tab10")
        colors = {l: cmap[i] for i, l in enumerate(sorted(layer_neurite_and_nodes[k]))}

        for i, l in enumerate(sorted(layer_neurite_and_nodes[k])):
            ax = axs[i] if len(layer_neurite_and_nodes[k]) > 1 else axs
            ax.tick_params(axis="x", labelrotation=45)
            type_exists = len(layer_neurite_and_nodes[k][l]) > 0 and not np.isnan(
                np.mean(layer_neurite_and_nodes[k][l])
            )

            if type_exists:
                text = (
                    f"Ø = {np.mean(layer_neurite_and_nodes[k][l]):.0f} ± "
                    f"{np.std(layer_neurite_and_nodes[k][l]):.0f}"
                )
                ax.set_title(text)
                ax.tick_params(axis="both")
                ax.hist(
                    layer_neurite_and_nodes[k][l], rwidth=0.6, color=colors[l], label=l
                )
                ax.legend()
                # fig.tight_layout()
            else:
                if i == len(layer_neurite_and_nodes[k]) // 2:
                    props = {"boxstyle": "round", "facecolor": "white", "alpha": 0.5}
                    text = f"the {k} tree not available"
                    fig.text(
                        0.0,
                        0.55,
                        text,
                        transform=ax.transAxes,
                        verticalalignment="top",
                        bbox=props,
                    )
                ax.tick_params(axis="both")
                # fig.tight_layout()
                ax.set_rasterized(True)

        image_path = (
            imagedir / f"number_of_nodes_{rtype.replace(' ', '_')}_per_layer_{k}"
        )
        if image_path:
            if not supported_ext:
                supported_ext = [".png", ".eps", ".pdf"]
            for ext in supported_ext:
                fig.savefig(image_path.with_suffix(ext))

        return image_path.with_suffix(".png")


def plot_neurite(
    neurite: tmd.Tree.Tree,
    ax: matplotlib.axes.Axes,
    *,
    rotation: int = 0,
    soma_size: int = 1,
    edge_width: int = 1,
) -> None:
    """Plot a neurite.

    Parameters
    ----------
    neurite
        The neurite.
    ax
        Matplotlib axes.
    rotation
        The rotation around the y-axis (normally the vertical axis) in degrees.
    soma_size
        The size of the soma in the plot. It will be plotted in red.
    edge_width
        The width of the edges in the plot. They will be plotted in black.
    """
    g = nx.Graph()
    for child, parent in enumerate(neurite.p):
        if parent == -1:
            continue
        g.add_edge(child, parent)

    angle = rotation * np.pi / 180
    px = neurite.x * np.cos(angle) + neurite.z * np.sin(angle)
    py = neurite.y

    nx.draw(
        g,
        ax=ax,
        pos={n: xy for n, xy in enumerate(zip(px, py))},
        nodelist=[0],
        node_color="red",
        node_size=soma_size,
        edge_color="black",
        width=edge_width,
        with_labels=False,
    )


def plot_persistence_diagram(
    persistence_data: list[list[int]],
    ax: matplotlib.axes.Axes,
    size: int = 1,
) -> None:
    """Plot a persistence diagram.

    Parameters
    ----------
    persistence_data
        The persistence data: a sequence of pairs (x, y), the points of
        the persistence diagram.
    ax
        Matplotlib axes.
    size
        The size of the persistence points in the plot.
    """
    x, y = zip(*persistence_data)
    min_max = (min(*x, *y), max(*x, *y))
    ax.plot(min_max, min_max, color="black")
    ax.scatter(x, y, color="black", s=size)
    ax.set(xlim=min_max, ylim=min_max)


def plot_persistence_image(
    persistence_data: list[list[int]],
    ax: matplotlib.axes.Axes,
) -> None:
    """Plot a persistence diagram.

    Parameters
    ----------
    persistence_data
        The persistence data: a sequence of pairs (x, y), the points of
        the persistence diagram.
    ax
        Matplotlib axes.
    """
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    extent = xlims + ylims
    img = tmd.Topology.analysis.get_persistence_image_data(
        persistence_data,
        xlims=xlims,
        ylims=ylims,
    )
    ax.imshow(np.rot90(img), cmap="jet", extent=extent)


def plot_morphology_images(
    neuron: tmd.Neuron.Neuron,
    neuron_label: str,
    neuron_name: str,
    figsize: tuple[float, float] = (12, 4),
) -> matplotlib.figure.Figure:
    """Plot morphology images.

    Parameters
    ----------
    neuron
        A neuron morphology.
    neuron_label
        Morphology type.
    neuron_name
        Morphology name.
    figsize
        The size of the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The plot figure.
    """
    fig = Figure(figsize=figsize, tight_layout=True)
    title = f"{neuron_label} ({neuron_name}, {len(neuron.neurites)} neurites)"
    fig.suptitle(title, fontweight="bold")
    ax0, ax1, ax2 = fig.subplots(ncols=3)

    ax0.set(title="Front view", aspect="equal")
    ax1.set(title="Side view", aspect="equal")
    for neurite in neuron.neurites:
        plot_neurite(neurite, ax0, soma_size=25)
        plot_neurite(neurite, ax1, soma_size=25, rotation=90)

    ax2.set_title("Persistence diagram/image")
    ax2.grid(False)
    persistence_data = tmd.Topology.methods.get_ph_neuron(neuron)
    plot_persistence_diagram(persistence_data, ax=ax2, size=5)
    plot_persistence_image(persistence_data, ax=ax2)

    return fig


def plot_barcode_enhanced(ph, ax, valID=2, linewidth=1.2):
    """Plot colored barcode.

    Parameters
    ----------
    ph : np.ndarray
        Persistence diagram.
    ax : matplotlib.axes.Axes
        Axis for the plot.
    valID : int, optional
        [description], by default 2
    linewidth : float, optional
        Line width, by default 1.2

    Returns
    -------
    CS3 : ScalarMappable
        Colors for colorbar.
    colors : list_like
        List of bar colors.
    """
    from tmd.view.common import jet_map

    cmap = jet_map

    # Initialization of matplotlib figure and axes.
    val_max = np.max(ph, axis=0)[valID]

    # Hack for colorbar creation
    norm = Normalize(vmin=np.min(ph), vmax=np.max(ph))
    CS3 = ScalarMappable(cmap=cmap, norm=norm)
    CS3.set_array([])

    def sort_ph_enhanced(ph, valID):
        """Sorts barcode according to length."""
        ph_sort = [p[: valID + 1] + [np.abs(p[0] - p[1])] for p in ph]
        ph_sort = sorted(
            ph_sort, key=lambda x: x[valID + 1]  # type: ignore[no-any-return]
        )
        return ph_sort

    ph_sort = sort_ph_enhanced(ph, valID)

    colors = [cmap(p[valID] / val_max) for p in ph_sort]

    for ip, p in enumerate(ph_sort):
        ax.plot(p[:2], [ip, ip], c=colors[ip], linewidth=linewidth)

    ax.set_ylim([-1, len(ph_sort)])

    return CS3, colors


def plot_diagram_enhanced(ph, ax, alpha=1.0, valID=2, edgecolors="black", s=30):
    """Plot colored diagram.

    Parameters
    ----------
    ph : np.ndarray
        Persistence diagram.
    ax : matplotlib.axes.Axes
        Axis for the plot.
    alpha : float, optional
        Transperancey of plot, by default 1.0.
    valID : int, optional
        [description], by default 2
    edgecolors : float
        Transperancey of plotby default black.
    s : float, optional
        Size of soma, by default 30.

    Returns
    -------
    CS3 : ScalarMappable
        Colors for colorbar.
    colors : list_like
        List of bar colors.
    """
    from tmd.view.common import jet_map

    cmap = jet_map

    # Initialization of matplotlib figure and axes.
    val_max = np.max(ph, axis=0)[valID]

    # Hack for colorbar creation
    norm = Normalize(vmin=np.min(ph), vmax=np.max(ph))
    CS3 = ScalarMappable(cmap=cmap, norm=norm)
    CS3.set_array([])

    def sort_ph_enhanced(ph, valID):
        """Sorts barcode according to length."""
        ph_sort = [p[: valID + 1] + [np.abs(p[0] - p[1])] for p in ph]
        ph_sort.sort(key=lambda x: x[valID + 1])  # type: ignore[no-any-return]
        return ph_sort

    ph_sort = sort_ph_enhanced(ph, valID)

    x_min = np.array(ph_sort)[:, 0].min()
    x_max = np.array(ph_sort)[:, 0].max()
    y_min = np.array(ph_sort)[:, 1].min()
    y_max = np.array(ph_sort)[:, 1].max()
    # regression line
    ax.plot([x_min, x_max], [y_min, y_max], c="black")

    colors = [cmap(p[valID] / val_max) for p in ph_sort]

    ax.scatter(
        np.array(ph_sort)[:, 0],
        np.array(ph_sort)[:, 1],
        c=colors,
        alpha=alpha,
        edgecolors=edgecolors,
        s=s,
    )

    return CS3, colors

"""Explain model layers using GradCam."""
from __future__ import annotations

import numpy as np
import torch
from captum.attr import visualization as viz
from matplotlib.figure import Figure
from scipy.stats import gaussian_kde
from tmd.Topology.methods import _filtration_function
from tmd.Topology.methods import tree_to_property_barcode
from tmd.Topology.persistent_properties import NoProperty
from tmd.view.common import jet_map

from morphoclass.data import MorphologyDataLoader
from morphoclass.vis import plot_barcode_enhanced
from morphoclass.vis import plot_diagram_enhanced
from morphoclass.vis import plot_tree
from morphoclass.xai import GradCAMExplainer
from morphoclass.xai.plot_node_saliency import plot_node_saliency


def grad_cam_gnn_model(model, dataset, sample_id):
    """Explain GNN model.

    Plot with two rows:
    - Original graph and graph with GradCam values within the nodes.
    - Heatmap of the original graph (zero-values) and heatmap of
      the GradCam values on the graph.
    Only one morphology sample is visualized.

    Parameters
    ----------
    model : morphoclass.models.man_net.ManNet
        Model that will be explained.
    dataset : morphoclass.data.morphology_dataset.MorphologyDataset
        Dataset containing morphologies.
    sample_id : int
        The id of morphology in the dataset.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with explainable plots.
    """
    layers = [
        attribute
        for attribute in dir(model.feature_extractor)
        if isinstance(getattr(model.feature_extractor, attribute), torch.nn.Module)
    ]
    layers = [layer for layer in layers if layer != "relu"]

    sample_id = int(sample_id)
    sample = dataset[sample_id]
    tree = sample.tmd_neurites[0]

    figsize = np.array([14, 5 * len(layers)])
    dpi = 200
    dx, dy = figsize * dpi * 1j

    fig = Figure(figsize=figsize, dpi=dpi)
    axs = fig.subplots(2, len(layers) + 1)

    # Visualize original barcode
    axs[0][0].set_title("Original\nGraph")
    plot_node_saliency(
        tree, np.zeros(len(tree.x)), ax=axs[0][0], name="Grad-CAM", show_legend=True
    )
    axs[1][0].set_title("Original\nGraph Heatmap")
    plot_tree(tree, axs[1][0], node_size=1.0, edge_color="orange")
    pxmin = tree.x.min()
    pxmax = tree.x.max()
    pymin = tree.y.min()
    pymax = tree.y.max()
    x, y = np.mgrid[pxmin:pxmax:dx, pymin:pymax:dy]
    node_pos = np.array([tree.x, tree.y])
    z = np.zeros(x.shape)
    axs[1][0].imshow(
        np.rot90(z),
        cmap=jet_map,  # "inferno"
        aspect="auto",
        extent=[pxmin, pxmax, pymin, pymax],
        alpha=0.5,
    )
    for index, layer in enumerate(layers, start=1):
        explainer = GradCAMExplainer(model, getattr(model.feature_extractor, layer))
        logits, attributions = explainer.get_cam(
            sample,
            loader_cls=MorphologyDataLoader,
            cls_idx=None,
            relu_weights=False,
            relu_cam=False,
        )
        attributions = attributions / np.abs(attributions).max()
        # get attributions only for the tree.tmd_neurites[0]
        attributions = attributions[: len(tree.x)]

        axs[0][index].set_title(f"Graph GradCam\non {layer}")
        plot_node_saliency(
            tree, attributions, ax=axs[0][index], name="Grad-CAM", show_legend=True
        )

        axs[1][index].set_title(f"GradCam as Heatmap\non {layer}")
        plot_tree(tree, axs[1][index], node_size=1.0, edge_color="orange")
        kde_weights = np.maximum(1e-12, attributions)

        kernel = gaussian_kde(node_pos, weights=kde_weights)
        x, y = np.mgrid[pxmin:pxmax:dx, pymin:pymax:dy]
        positions = np.vstack([x.ravel(), y.ravel()])
        z = np.reshape(kernel(positions).T, x.shape)
        axs[1][index].imshow(
            np.rot90(z),
            cmap=jet_map,
            aspect="auto",
            extent=[pxmin, pxmax, pymin, pymax],
            alpha=0.5,
        )

    probabilities = np.exp(logits)
    prediction = dataset.y_to_label[probabilities.argmax(axis=0)]
    fig.suptitle(
        f"Ground truth: {sample.y_str}\nPrediction: {prediction}",
        fontsize=15,
        weight=30,
    )
    return fig


def grad_cam_cnn_model(model, dataset, sample_id):
    """Explain CNN model.

    Plot with feature maps after each feature extractor layer.
    Starting from original image to the last featrue extractor layer.
    Only one morphology sample is visualized.

    Parameters
    ----------
    model : morphoclass.models.cnnet.CNNet
        Model that will be explained.
    dataset : morphoclass.data.morphology_dataset.MorphologyEmbeddingDataset
        Dataset containing embeddings and morphologies.
    sample_id : int
        The id of embedding in the dataset.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with explainable plots.
    """
    layers = [
        attribute
        for attribute in dir(model.feature_extractor)
        if isinstance(getattr(model.feature_extractor, attribute), torch.nn.Module)
    ]
    layers = [layer for layer in layers if layer != "relu"]

    sample_id = int(sample_id)
    sample = dataset[sample_id]
    image = sample.image
    if isinstance(image, torch.Tensor):
        image = image.cpu().detach().numpy()

    figsize = (len(layers) * 3, 4)

    fig = Figure(figsize=figsize, dpi=200)
    axs = fig.subplots(1, len(layers) + 1)

    # Visualize original image
    _ = viz.visualize_image_attr(
        None,
        np.transpose(image, (2, 3, 0, 1)),
        method="original_image",
        title="Original Image",
        plt_fig_axis=(fig, axs[0]),
    )

    for index, layer in enumerate(layers, start=1):
        explainer = GradCAMExplainer(model, getattr(model.feature_extractor, layer))
        logits, attributions = explainer.get_cam(
            sample,
            loader_cls=MorphologyDataLoader,
            cls_idx=None,
            relu_weights=False,
            relu_cam=False,
        )
        attributions = attributions / np.abs(attributions).max()

        if len(attributions.shape) < 3:
            raise ValueError("There is no support for non-feature maps layers")

        scale = image.shape[-1] // attributions.shape[-1]

        attributions = attributions.reshape((-1, *attributions.shape[-3:]))
        # The additive composition of feature effects
        attributions = np.sum(attributions, axis=0)
        attributions = attributions.reshape((-1, *attributions.shape))

        attributions_new = np.empty(image.shape)
        for w in range(image.shape[-2]):
            for h in range(image.shape[-1]):
                i = w // scale
                j = h // scale
                attributions_new[:, :, w, h] = attributions[:, :, i, j]

        # Visualize the layer outputs
        viz.visualize_image_attr(
            np.transpose(attributions_new, (2, 3, 0, 1)),
            np.transpose(image, (2, 3, 0, 1)),
            method="blended_heat_map",
            sign="absolute_value",
            cmap=jet_map,
            show_colorbar=True,
            title=f"Overlayed GradCam\nExplainer on ${layer}$",
            plt_fig_axis=(fig, axs[index]),
        )

    probabilities = np.exp(logits)
    prediction = dataset.y_to_label[probabilities.argmax(axis=0)]
    fig.suptitle(
        f"Ground truth: {sample.y_str}\nPrediction: {prediction}",
        fontsize=15,
        weight=30,
    )
    return fig


def grad_cam_perslay_model(model, dataset, sample_id):
    """Explain PersLay model.

    Plot with 3 rows:
    - Barcodes: The original barcode and GradCam weighted barcode (colored bar)
      after each feature extraction layer.
    - Persistence diagrams: The original PD and GradCam weighted PD (colored dot)
      after each feature extraction layer.
    - Graph: The original graph and GradCam weighted graph (colored edge)
      after each feature extraction layer.

    Parameters
    ----------
    model : morphoclass.models.coriander_net.CorianderNet
        Model that will be explained.
    dataset : morphoclass.data.morphology_dataset.MorphologyEmbeddingDataset
        Dataset containing embeddings and morphologies.
    sample_id : int
        The id of embedding in the dataset.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with explainable plots.
    """
    layers = [
        attribute
        for attribute in dir(model.feature_extractor)
        if isinstance(getattr(model.feature_extractor, attribute), torch.nn.Module)
    ]
    layers = [layer for layer in layers if layer != "relu"]

    sample_id = int(sample_id)
    sample = dataset[sample_id]
    pd = sample.embedding
    tree = sample.morphology.tmd_neurites[0]

    figsize = (8, 4 * (len(layers) + 2))
    dpi = 200

    fig = Figure(figsize=figsize, dpi=dpi)
    axs = fig.subplots(3, len(layers) + 1)

    # Visualize original barcode
    axs[0][0].set_title("Original\nBarcode")
    pd_weights = np.array([[a] for a in np.ones(len(pd))])
    pd_enhanced = np.concatenate([pd, pd_weights], axis=1).tolist()
    CS3, colors = plot_barcode_enhanced(pd_enhanced, axs[0][0])
    fig.subplots_adjust(wspace=0.2, hspace=0.3)

    axs[1][0].set_title("Original PD")
    pd_weights = np.array([[a] for a in np.ones(len(pd))])
    pd_enhanced = np.concatenate([pd, pd_weights], axis=1).tolist()
    CS3, colors = plot_diagram_enhanced(pd_enhanced, axs[1][0])

    axs[2][0].set_title("Original\nGraph")
    plot_tree(tree, axs[2][0], node_size=1.0, edge_color="orange", width=2)

    for index, layer in enumerate(layers, start=1):
        explainer = GradCAMExplainer(model, getattr(model.feature_extractor, layer))
        logits, attributions = explainer.get_cam(
            sample,
            loader_cls=MorphologyDataLoader,
            cls_idx=None,
            relu_weights=False,
            relu_cam=False,
        )
        attributions = attributions / np.abs(attributions).max()

        axs[0][index].set_title(f"Barcode GradCam\non {layer}")
        pd_weights = np.array([[a] for a in attributions])
        pd_enhanced = np.concatenate([pd, pd_weights], axis=1).tolist()
        CS3, colors = plot_barcode_enhanced(pd_enhanced, axs[0][index])
        fig.colorbar(CS3, ax=axs[0][index], format="%.2f", fraction=0.046, pad=0.04)

        axs[1][index].set_title(f"PD GradCam\non {layer}")
        pd_weights = np.array([[a] for a in attributions])
        pd_enhanced = np.concatenate([pd, pd_weights], axis=1).tolist()
        CS3, colors = plot_diagram_enhanced(pd_enhanced, axs[1][index])
        fig.colorbar(CS3, ax=axs[1][index], format="%.2f", fraction=0.046, pad=0.04)

        axs[2][index].set_title(f"Graph GradCam\non {layer}")
        color_edges = get_edges_colors_based_on_barcode_colors(tree, colors)
        plot_tree(tree, axs[2][index], node_size=1.0, edge_color=color_edges, width=2)

    probabilities = np.exp(logits)
    prediction = dataset.y_to_label[probabilities.argmax(axis=0)]
    fig.suptitle(
        f"Ground truth: {sample.y_str}\nPrediction: {prediction}",
        fontsize=15,
        weight=30,
    )
    return fig


def get_edges_colors_based_on_barcode_colors(tree, colors):
    """Collect colors for edges based on barcode colors.

    Parameters
    ----------
    tree : tmd.Tree.Tree
        Morphology tree used to create barcode.
    colors : list_like
        List of barcode colors.

    Returns
    -------
    color_edges : list_like
        List of edge colors.
    """
    # WARNING: the same feature used in the feature extraction
    featrue_extractor_kwargs = {"feature": "projection"}
    _, bars_to_points = tree_to_property_barcode(
        tree,
        filtration_function=_filtration_function(**featrue_extractor_kwargs),
        property_class=NoProperty,
    )

    color_edges = []
    for _, parent in zip(*tree.sections):
        for i, bar in enumerate(bars_to_points):
            if parent in bar:
                color_edges.append(colors[i])
                break

    return color_edges

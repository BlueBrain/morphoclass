"""Explain model layers using GradShap."""
from __future__ import annotations

import sys
import textwrap
from typing import Any

import numpy as np
import shap
import torch
from captum.attr import visualization as viz
from matplotlib.figure import Figure
from scipy.stats import gaussian_kde
from sklearn import tree
from tmd.Topology.methods import _filtration_function
from tmd.Topology.methods import tree_to_property_barcode
from tmd.Topology.persistent_properties import NoProperty
from tmd.view.common import jet_map

from morphoclass.data import MorphologyDataLoader
from morphoclass.data import MorphologyEmbeddingDataLoader
from morphoclass.training import reset_seeds
from morphoclass.vis import plot_barcode_enhanced
from morphoclass.vis import plot_diagram_enhanced
from morphoclass.vis import plot_tree
from morphoclass.xai.plot_node_saliency import plot_node_saliency

# from captum.attr import GradientShap


def gnn_model_attributions(model, dataset, sample_id, interpretability_method_cls):
    """Explain GNN model.

    Plot with two rows:
    - Original graph and graph with GradShap values within the nodes.
    - Heatmap of the original graph (zero-values) and heatmap of
      the GradShap values on the graph.
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
    reset_seeds(numpy_seed=0, torch_seed=0)

    sample_id = int(sample_id)
    sample = dataset[sample_id]
    tree = sample.tmd_neurites[0]

    device = next(model.parameters()).device
    loader = MorphologyDataLoader(dataset.index_select([sample_id]))
    batch = next(iter(loader)).to(device)

    def _handle_batch(batch):
        batch_input_x = batch.x.reshape((-1, 1))
        dummy_fill = (
            torch.ones((2, len(batch_input_x) - batch.edge_index.shape[1])) * -2
        )
        dummy_fill = dummy_fill.to(device)
        batch_input_edge_index = torch.cat([batch.edge_index, dummy_fill], dim=1).T
        batch_input_batch = batch.batch.reshape((-1, 1))
        batch_input = torch.cat(
            [batch_input_x, batch_input_edge_index, batch_input_batch], dim=1
        ).to(device)
        batch_target = batch.y
        return batch_input, batch_target

    batch_input, batch_target = _handle_batch(batch)
    batch_input = batch_input.to(device)

    figsize = np.array([8, 8])
    dpi = 200
    dx, dy = figsize * dpi * 1j

    fig = Figure(figsize=figsize, dpi=dpi)
    axs = fig.subplots(2, 2)

    # Visualize original barcode
    axs[0][0].set_title("Original\nGraph")
    plot_node_saliency(
        tree,
        np.zeros(len(tree.x)),
        ax=axs[0][0],
        name=interpretability_method_cls.__name__,
        show_legend=True,
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
        cmap=jet_map,
        aspect="auto",
        extent=[pxmin, pxmax, pymin, pymax],
        alpha=0.5,
    )
    baseline_all = torch.cat([batch_input * 0, batch_input * 1])

    interpretability_method = interpretability_method_cls(model)
    kwargs = {}
    if "Shap" in interpretability_method.__class__.__name__:
        kwargs = {"baselines": baseline_all}
    elif "Occlusion" in interpretability_method.__class__.__name__:
        kwargs = {"sliding_window_shapes": (1, 10, 10)}

    attributions = interpretability_method.attribute(
        batch_input,
        target=batch_target,
        **kwargs,
    )
    attributions = attributions[:, 0]
    attributions = attributions.cpu().detach().numpy()
    # get attributions only for the tree.tmd_neurites[0]
    attributions = attributions[: len(tree.x)]

    if len(attributions) > 0 and np.abs(attributions).max() != 0:
        attributions = attributions / np.abs(attributions).max()
    else:
        attributions[attributions == 0] = 1e-10

    axs[0][1].set_title(f"Graph {interpretability_method_cls.__name__}")
    plot_node_saliency(
        tree,
        attributions,
        ax=axs[0][1],
        name=interpretability_method_cls.__name__,
        show_legend=True,
    )

    axs[1][1].set_title(f"{interpretability_method_cls.__name__} as Heatmap")
    plot_tree(tree, axs[1][1], node_size=1.0, edge_color="orange")
    kde_weights = np.maximum(0, attributions)
    kde_weights = np.nan_to_num(kde_weights)
    kde_weights[kde_weights == 0] = 1e-10
    kernel = gaussian_kde(node_pos, weights=kde_weights)
    x, y = np.mgrid[pxmin:pxmax:dx, pymin:pymax:dy]
    positions = np.vstack([x.ravel(), y.ravel()])
    z = np.reshape(kernel(positions).T, x.shape)
    axs[1][1].imshow(
        np.rot90(z),
        cmap=jet_map,
        aspect="auto",
        extent=[pxmin, pxmax, pymin, pymax],
        alpha=0.5,
    )

    # get prediction for this sample
    batch_probabilities = model(batch)
    batch_probabilities = torch.exp(batch_probabilities)
    prediction = dataset.y_to_label[batch_probabilities.argmax(axis=1)[0].item()]
    fig.suptitle(
        f"Ground truth: {sample.y_str}\nPrediction: {prediction}",
        fontsize=15,
        weight=30,
    )

    return fig


def cnn_model_attributions(model, dataset, sample_id, interpretability_method_cls):
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
    reset_seeds(numpy_seed=0, torch_seed=0)

    sample_id = int(sample_id)
    sample = dataset[sample_id]

    device = next(model.parameters()).device

    dataset_all = [s for s in dataset if sample.morph_name != s.morph_name]
    loader_all = MorphologyEmbeddingDataLoader(
        dataset_all,
        shuffle=False,
    )
    batch_all = next(iter(loader_all)).to(device)

    loader = MorphologyEmbeddingDataLoader(
        [sample],
        shuffle=False,
    )

    batch = next(iter(loader)).to(device)
    batch_input = batch.image.clone().detach().requires_grad_(True)
    batch_input = batch_input.type(torch.FloatTensor)
    batch_input = batch_input.to(device)
    batch_target = batch.y

    baseline_all = torch.cat(
        [
            batch_all.image.mean(axis=0).reshape(batch_input.shape),
            batch_all.image.mean(axis=0).reshape(batch_input.shape),
        ]
    )

    figsize = (2 * 3, 4)

    fig = Figure(figsize=figsize, dpi=200)
    axs = fig.subplots(1, 2)

    # Visualize original image
    _ = viz.visualize_image_attr(
        None,
        np.transpose(batch_input.cpu().detach().numpy(), (2, 3, 0, 1)),
        method="original_image",
        title="Original Image",
        plt_fig_axis=(fig, axs[0]),
    )

    interpretability_method = interpretability_method_cls(
        model,
    )
    kwargs = {}
    if "Shap" in interpretability_method.__class__.__name__:
        kwargs = {"baselines": baseline_all}
    elif "Occlusion" in interpretability_method.__class__.__name__:
        kwargs = {"sliding_window_shapes": (1, 10, 10)}

    attributions = interpretability_method.attribute(
        batch_input, target=batch_target, **kwargs
    )
    attributions = attributions.cpu().detach().numpy()
    if len(attributions) > 0 and np.abs(attributions).max() != 0:
        attributions = attributions / np.abs(attributions).max()
    else:
        attributions = np.ones(batch_input.shape) * 1e-5

    # Visualize baseline with all samples as baseline
    _ = viz.visualize_image_attr(
        np.transpose(attributions, (2, 3, 0, 1)),
        np.transpose(batch_input.cpu().detach().numpy(), (2, 3, 0, 1)),
        method="blended_heat_map",
        title=interpretability_method_cls.__name__,
        plt_fig_axis=(fig, axs[1]),
        cmap=jet_map,
        show_colorbar=True,
    )

    # get prediction for this sample
    batch_probabilities = model(batch)
    batch_probabilities = torch.exp(batch_probabilities)
    prediction = dataset.y_to_label[batch_probabilities.argmax(axis=1)[0].item()]
    fig.suptitle(
        f"Ground truth: {sample.y_str}\nPrediction: {prediction}",
        fontsize=15,
        weight=30,
    )
    return fig


def cnn_model_attributions_population(model, dataset):
    """Generate the SHAP explanation for a population of neurons."""
    reset_seeds()
    device = next(model.parameters()).device
    class_labels = list(dataset.y_to_label.values())

    # collect SHAP per sample
    average_shap: dict[str, dict[str, list]] = {
        l1: {l2: [] for l2 in class_labels} for l1 in class_labels
    }
    average_shap_means: dict[str, Any] = {}
    average_expected_value: dict[str, Any] = {}
    for sample in dataset:
        class_label = sample.y_str
        average_shap_means[class_label] = []
        average_expected_value[class_label] = []

        # class 1
        dataset_class = [
            s
            for s in dataset
            if sample.morph_name != s.morph_name and s.y_str == class_label
        ]
        loader_class = MorphologyEmbeddingDataLoader(
            dataset_class,
            batch_size=len(dataset_class),
        )
        batch_class = next(iter(loader_class)).to(device)
        baseline, _ = torch.median(batch_class.image, dim=0)
        baseline = baseline.reshape((1, *batch_class.image.shape[1:]))
        baseline = baseline.clone().detach().requires_grad_(True)
        baseline = baseline.type(torch.FloatTensor)
        baseline = baseline.to(device)

        # 1 sample
        loader = MorphologyEmbeddingDataLoader([sample])
        batch = next(iter(loader)).to(device)

        # Explain
        model = model.train()
        explainer = shap.DeepExplainer(model, baseline)
        average_expected_value[class_label].append(np.exp(explainer.expected_value))
        shap_values = explainer.shap_values(batch.image)
        mean_shaps = [shap_values[idx_j].sum() for idx_j, _ in enumerate(class_labels)]
        average_shap_means[class_label].append(mean_shaps)
        for i, label in enumerate(class_labels):
            average_shap[class_label][label].append(shap_values[i])

    # prepare for population plot
    for baseline in average_shap:
        average_expected_value[baseline] = np.median(
            np.array(average_expected_value[baseline]), axis=0
        )
        average_shap_means[baseline] = np.median(
            np.array(average_shap_means[baseline]), axis=0
        )
        for compare_to in average_shap[baseline]:
            average_shap[baseline][compare_to] = np.median(
                np.array(average_shap[baseline][compare_to]), axis=0
            )
    # TODO: Get rid of pyplot. Problem: shap.image_plot uses pyplot and we need
    #       to run plt.gcf() to get the figure instance.
    import matplotlib.pyplot as plt

    # population plot
    figures = []
    for class_label in class_labels:
        dataset_class = [s for s in dataset if s.y_str == class_label]
        loader_class = MorphologyEmbeddingDataLoader(
            dataset_class,
            batch_size=len(dataset_class),
        )
        batch_class = next(iter(loader_class)).to(device)
        baseline, _ = torch.median(batch_class.image, dim=0)
        baseline = baseline.reshape((1, *batch_class.image.shape[1:]))
        baseline = baseline.clone().detach().requires_grad_(True)

        shap_values = [
            average_shap[class_label][dataset.y_to_label[i]]
            for i in range(len(class_labels))
        ]
        shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
        test_numpy = np.swapaxes(
            np.swapaxes(baseline.cpu().detach().numpy(), 1, -1), 1, 2
        )

        # Generate subplot titles
        subtitles = []
        for i, label in enumerate(class_labels):
            shap_value = average_shap_means[class_label][i]
            pr_avg = average_expected_value[class_label][i] * 100
            subtitle = f"""
            {"baseline" if label == class_label else "compared to"}: {label}
            SHAP: {shap_value:.2f}
            Pr_avg: {pr_avg:.2f}
            """
            subtitles.append(textwrap.dedent(subtitle).strip())

        # Make SHAP plots
        subtitles = np.array(subtitles).reshape((-1, len(subtitles)))
        shap.image_plot(
            shap_numpy,
            -test_numpy,
            labels=subtitles,
            labelpad=10,
            width=150,
            show=False,
        )
        figures.append(plt.gcf())

    return figures, class_labels


def perslay_model_attributions(
    model, dataset, sample_id, interpretability_method_cls, aggregation_fn=np.sum
):
    """Explain PersLay model.

    Plot with 3 rows:
    - Barcodes: The original barcode and GradShap weighted barcode (colored bar)
      after each feature extraction layer.
    - Persistence diagrams: The original PD and GradShap weighted PD (colored dot)
      after each feature extraction layer.
    - Graph: The original graph and GradShap weighted graph (colored edge)
      after each feature extraction layer.

    Parameters
    ----------
    model : morphoclass.models.coriander_net.CorianderNet
        Model that will be explained.
    dataset : morphoclass.data.morphology_dataset.MorphologyEmbeddingDataset
        Dataset containing embeddings and morphologies.
    sample_id : int
        The id of embedding in the dataset.
    aggregation_fn : callable, default np.sum
        The attributions aggregation function, usually sum, mean, max.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with explainable plots.
    """
    reset_seeds(numpy_seed=0, torch_seed=0)

    sample_id = int(sample_id)
    sample = dataset[sample_id]

    device = next(model.parameters()).device

    dataset_all = [s for s in dataset if sample.morph_name != s.morph_name]
    loader_all = MorphologyEmbeddingDataLoader(
        dataset_all,
        shuffle=False,
    )
    batch_all = next(iter(loader_all)).to(device)

    loader = MorphologyEmbeddingDataLoader(
        [sample],
        shuffle=False,
    )
    batch = next(iter(loader)).to(device)

    batch_input1 = batch.point_index.unsqueeze(dim=1)
    batch_input2 = batch.embedding
    batch_input = (
        torch.cat([batch_input2, batch_input1], dim=1)
        .clone()
        .detach()
        .requires_grad_(True)
    )
    batch_input = batch_input.to(device)
    batch_target = batch.y

    batch_all_input1 = batch_all.point_index.unsqueeze(dim=1)
    batch_all_input2 = batch_all.embedding
    batch_all_input = torch.cat([batch_all_input2, batch_all_input1], dim=1)
    baseline_all = torch.cat([batch_all_input, batch_all_input])

    figsize = (2 * 6, 14)

    fig = Figure(figsize=figsize, dpi=200)
    axs = fig.subplots(3, 2)

    pd = sample.embedding
    tree = sample.morphology.tmd_neurites[0]

    # Visualize original
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
    plot_tree(tree, axs[2][0], node_size=1.0, edge_color="k", width=2)

    # Visualize shap on all samples
    interpretability_method = interpretability_method_cls(
        model,
    )
    kwargs = {}
    if "Shap" in interpretability_method.__class__.__name__:
        kwargs = {"baselines": baseline_all}
    elif "Occlusion" in interpretability_method.__class__.__name__:
        kwargs = {"sliding_window_shapes": (1, 10, 10)}

    attributions = interpretability_method.attribute(
        batch_input, target=batch_target, **kwargs
    )
    attributions = attributions.cpu().detach().numpy()
    if len(attributions) > 0 and np.abs(attributions).max() != 0:
        attributions = attributions / np.abs(attributions).max()
    else:
        attributions = np.ones(batch_input.shape) * 1e-5

    accumulator_limit = 3
    while np.isnan(attributions[0][0]):
        attributions = interpretability_method.attribute(
            batch_input, target=batch_target, **kwargs
        )
        attributions = attributions.cpu().detach().numpy()
        if len(attributions) > 0 and np.abs(attributions).max() != 0:
            attributions = attributions / np.abs(attributions).max()
        else:
            attributions = np.ones(batch_input.shape) * 1e-5

        if accumulator_limit == 0:
            break
        else:
            accumulator_limit -= 1

    axs[0][1].set_title(f"Barcode {interpretability_method_cls.__name__}")
    pd_weights = np.array([[sum(a)] for a in attributions])
    pd_enhanced = np.concatenate([pd, pd_weights], axis=1).tolist()
    CS3, colors = plot_barcode_enhanced(pd_enhanced, axs[0][1])
    fig.colorbar(CS3, ax=axs[0][1], format="%.2f", fraction=0.046, pad=0.04)

    axs[1][1].set_title(f"PD {interpretability_method_cls.__name__}")
    pd_weights = np.array([[sum(a)] for a in attributions])
    pd_enhanced = np.concatenate([pd, pd_weights], axis=1).tolist()
    CS3, colors = plot_diagram_enhanced(pd_enhanced, axs[1][1])
    fig.colorbar(CS3, ax=axs[1][1], format="%.2f", fraction=0.046, pad=0.04)

    axs[2][1].set_title(f"Graph {interpretability_method_cls.__name__}")
    color_edges = get_edges_colors_based_on_barcode_colors(tree, colors)
    plot_tree(tree, axs[2][1], node_size=1.0, edge_color=color_edges, width=2)

    # get prediction for this sample
    batch_probabilities = model(batch)
    batch_probabilities = torch.exp(batch_probabilities)  # .detach().cpu().numpy()
    prediction = dataset.y_to_label[batch_probabilities.argmax(axis=1)[0].item()]
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
    # the same feature used in the feature extraction
    featrue_extractor_kwargs = {"feature": "projection"}
    _, bars_to_points = tree_to_property_barcode(
        tree,
        filtration_function=_filtration_function(**featrue_extractor_kwargs),
        property_class=NoProperty,
    )

    color_edges = []
    for child, parent in zip(*tree.sections):
        for i, bar in enumerate(bars_to_points):
            if child in bar:
                color_edges.append(colors[i])
                break
        else:
            # if child wasn't found, then should be one parent
            for i, bar in enumerate(bars_to_points):
                if parent in bar:
                    color_edges.append(colors[i])
                    break
    return color_edges


def sklearn_model_attributions_shap(model, dataset, sample_id):
    """Explain sklearn model.

    Plot with feature maps after each feature extractor layer.
    Starting from original image to the last featrue extractor layer.
    Only one morphology sample is visualized.

    Parameters
    ----------
    model
        Model that will be explained.
    dataset : morphoclass.data.morphology_dataset.MorphologyEmbeddingDataset
        Dataset containing embeddings and morphologies.
    sample_id : int
        The id of embedding in the dataset.
    aggregation_fn : callable, default np.sum
        The attributions aggregation function, usually sum, mean, max.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with explainable plots.
    """
    reset_seeds(numpy_seed=0, torch_seed=0)

    sample_id = int(sample_id)
    sample = dataset[sample_id]

    module = sys.modules[__name__]

    labels_ids = sorted(dataset.y_to_label.keys())
    labels_unique = [dataset.y_to_label[s] for s in labels_ids]

    for label in labels_unique:
        dataset_class = [
            s for s in dataset if s.y_str == label and sample.morph_name != s.morph_name
        ]
        loader_class = MorphologyEmbeddingDataLoader(
            dataset_class,
            shuffle=False,
        )
        batch_class = next(iter(loader_class))
        setattr(module, f"dataset_class_{label}", dataset_class)
        setattr(module, f"loader_class_{label}", loader_class)
        setattr(module, f"batch_class_{label}", batch_class)

    dataset_all = [s for s in dataset if sample.morph_name != s.morph_name]
    loader_all = MorphologyEmbeddingDataLoader(
        dataset_all,
        shuffle=False,
    )
    batch_all = next(iter(loader_all))

    loader = MorphologyEmbeddingDataLoader(
        [sample],
        shuffle=False,
    )

    batch = next(iter(loader))

    batch_input = batch.image.numpy()

    for label in labels_unique:
        batch_class = getattr(module, f"batch_class_{label}")
        baseline = torch.cat(
            [
                batch_class.image.mean(axis=0).reshape(batch_input.shape),
                batch_class.image.mean(axis=0).reshape(batch_input.shape),
            ]
        )
        setattr(module, f"baseline_{label}", baseline)

    figsize = ((len(labels_unique) + 2) * 3, 4)

    fig = Figure(figsize=figsize, dpi=200)
    axs = fig.subplots(1, len(labels_unique) + 1)

    # Visualize original image
    _ = viz.visualize_image_attr(
        None,
        np.transpose(batch_input.reshape((1, 1, 100, 100)), (2, 3, 0, 1)),
        method="original_image",
        title="Original Image",
        plt_fig_axis=(fig, axs[0]),
    )

    explainer = shap.Explainer(model, batch_all.image.numpy().reshape((-1, 100 * 100)))
    attributions = explainer.shap_values(batch.image.numpy().reshape((-1, 100 * 100)))

    text = ""
    for index, label in enumerate(labels_unique, start=1):
        attributions_class = attributions[index - 1]
        attributions_class = attributions_class / np.abs(attributions_class).max()
        attributions_image = attributions_class.reshape((100, 100))
        x_coordinate, y_coordinate = np.where(attributions_image > 0)
        number_of_important_features = len(attributions_class[attributions_class > 0])
        text += f"Baseline: {label} - {number_of_important_features} pixel(s)\n"
        for x, y in zip(x_coordinate, y_coordinate):
            text += f"pixel {x}, {y} with SHAP value {attributions_image[x][y]:.2f}\n"

        _ = viz.visualize_image_attr(
            np.transpose(attributions_class.reshape((1, 1, 100, 100)), (2, 3, 0, 1)),
            np.transpose(batch_input.reshape((1, 1, 100, 100)), (2, 3, 0, 1)),
            method="blended_heat_map",
            title=f"Baseline: {label} - {number_of_important_features} pixel(s)",
            plt_fig_axis=(fig, axs[index]),
            cmap=jet_map,
        )
    # get prediction for this sample
    prediction = model.predict(batch.image.numpy().reshape((-1, 100 * 100)))
    prediction = dataset.y_to_label[prediction[0]]
    fig.suptitle(
        f"Ground truth: {sample.y_str}\nPrediction: {prediction}",
        fontsize=15,
        weight=30,
    )
    return fig, text


def sklearn_model_attributions_tree(model, dataset):
    """Explain sklearn tree model.

    Parameters
    ----------
    model
        Model that will be explained.
    dataset : morphoclass.data.morphology_dataset.MorphologyEmbeddingDataset
        Dataset containing embeddings and morphologies.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with explainable plots.
    """
    labels_ids = sorted(dataset.y_to_label.keys())
    labels_unique = [dataset.y_to_label[s] for s in labels_ids]

    # TODO: Get rid of pyplot. Problem: AttributeError: 'FigureCanvasBase'
    #       object has no attribute 'get_renderer'
    # from matplotlib.figure import Figure
    # fig = Figure(figsize=(10,10), dpi=70)
    # ax = fig.subplots()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(20, 20), dpi=200)

    feature_names = [f"pixel [{x}, {y}]" for x in range(100) for y in range(100)]
    tree.plot_tree(model, feature_names=feature_names, class_names=labels_unique, ax=ax)

    return fig

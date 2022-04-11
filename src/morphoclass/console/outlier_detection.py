"""Implementation of the `morphoclass outlier-detection` CLI command."""
from __future__ import annotations

import click

from morphoclass.console import helpers


@click.command(
    name="outlier-detection",
    help="""
    This is the outlier-detection help message.

    The outlier detection command runs a server with an interactive visualization with
    the latent features of trained checkpoint model in 2 dimensional space.

    Latent features are usually not a 2-dimensional vectors. Therefore,
    the embedding algorithm is used to perform dimensionality reduction.
    By default, PCA is used. However, user is able to explicitly specify wished
    embedding algorithms with parameters.

    The visualization consists of plotting the training and validation latent features.
    The ground truth and prediction are shown as colors.
    The user is then able to click on any of the embeddings in order to see the plots
    of their original morphologies, persistence diagrams and persistence images.
    Current selection is highlighted in red.

    The method takes more time if the dataset of morphologies is bigger since
    it collects all the images necessary for the interactive plot click event.

    \b
    Example how to use it:
    (i) outlier detection with default embedding algorithm (PCA):
    $ morphoclass outlier-detection --checkpoint-path CHECKPOINT_PATH
    (ii) outlier detection with explicitly specified embedding algorithm (e.g. Isomap):
    $ morphoclass outlier-detection --checkpoint-path CHECKPOINT_PATH \\
                                   --embedding-class sklearn.manifold.Isomap \\
                                   --embedding-params n_neighbors=10
    $ morphoclass outlier-detection --checkpoint-path CHECKPOINT_PATH \\
                                   --embedding-class umap.UMAP
    """,
)
@click.option(
    "--checkpoint-path",
    type=click.STRING,
    required=True,
    help="Checkpoint path.",
)
@click.option(
    "--embedding-class",
    type=click.STRING,
    required=False,
    help="Embedding class.",
)
@click.option(
    "--embedding-params",
    required=False,
    callback=helpers.validate_params,
    help="Embedding parameters, e.g., 'a=1 b=2'.",
)
def cli(checkpoint_path, embedding_class=None, embedding_params=None):
    """Application for outlier detection based on trained model.

    Parameters
    ----------
    checkpoint_path : str
        Checkpoint path where to store model output with statistics.
    embedding_class : str, optional
        Embeddings class.
    embedding_params : dict, optional
        Embeddings parameters, provided to click as string 'a=1 b=2', and
        converted to dict using callback `helpers.validate_params`.
    """
    import importlib

    import numpy as np
    import torch

    from morphoclass.data import MorphologyDataset
    from morphoclass.utils import dict2kwargs
    from morphoclass.xai.embedding_visualization import embed_latent_features
    from morphoclass.xai.embedding_visualization import get_embeddings_figure
    from morphoclass.xai.embedding_visualization import get_outlier_detection_app

    checkpoint = torch.load(checkpoint_path)

    embedding_class = embedding_class or "sklearn.decomposition.PCA"
    if embedding_class == "sklearn.decomposition.PCA":
        embedding_params = embedding_params or {
            "n_components": 2,
            "svd_solver": "arpack",
        }
    else:
        embedding_params = {}
    module_name, _, class_name = embedding_class.rpartition(".")
    embedding_cls = getattr(importlib.import_module(module_name), class_name)
    embedder = embedding_cls(**embedding_params)

    description_text = f"""
        * Embedder: `{embedding_class}({
            dict2kwargs(embedding_params)})`
        * Checkpoint: `{checkpoint_path}`
        * Self confidence is the holdout probability that an example
          belongs to its given class label.
        * Click on the any embedding below to see more information about the
          morphology it belongs to.
        * Double-click to remove the current selection.
    """

    figures = []
    cleanlab_ordered_label_errors = checkpoint["cleanlab_ordered_label_errors"]
    cleanlab_self_confidence = checkpoint["cleanlab_self_confidence"]

    # no need for feature extraction, we will only handle morphologies anyway
    dataset = MorphologyDataset.from_csv(checkpoint["input_csv"])

    # get figure with embeddings
    x_coordinates, y_coordinates = embed_latent_features(
        checkpoint["latent_features_all"],
        embedder=embedder,
    )

    figure = get_embeddings_figure(
        x_coordinates=x_coordinates,
        y_coordinates=y_coordinates,
        dataset=dataset,
        predictions=None,
        train_idx=range(len(dataset)),
        val_idx=None,
        cleanlab_ordered_label_errors=cleanlab_ordered_label_errors,
        cleanlab_self_confidence=cleanlab_self_confidence,
    )

    figures.append(figure)

    # collect figures per split
    for number_of_split in range(len(checkpoint["train_idx"])):
        # get indices present in validation set and cleanlab error detection
        val_indices = checkpoint["val_idx"][number_of_split]
        cleanlab_indices = np.intersect1d(val_indices, cleanlab_ordered_label_errors)
        cleanlab_confidence = cleanlab_self_confidence[
            np.isin(cleanlab_ordered_label_errors, cleanlab_indices)
        ]

        # get figure with embeddings
        x_coordinates, y_coordinates = embed_latent_features(
            checkpoint["latent_features"][number_of_split],
            embedder=embedder,
        )

        figure = get_embeddings_figure(
            x_coordinates=x_coordinates,
            y_coordinates=y_coordinates,
            dataset=dataset,
            predictions=checkpoint["predictions"],
            train_idx=checkpoint["train_idx"][number_of_split],
            val_idx=val_indices,
            cleanlab_ordered_label_errors=cleanlab_indices,
            cleanlab_self_confidence=cleanlab_confidence,
        )

        figures.append(figure)

    app = get_outlier_detection_app(
        figures=figures,
        dataset=dataset,
        description_text=description_text,
    )
    app.run_server(debug=True, use_reloader=False)

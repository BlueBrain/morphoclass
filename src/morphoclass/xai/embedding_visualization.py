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
"""Plot embeddings for outlier detection."""
from __future__ import annotations

import base64
import io
import logging
import pathlib

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as dhc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import sklearn
import tmd.io
from dash.dependencies import Input
from dash.dependencies import Output
from dash.dependencies import State
from dash.exceptions import PreventUpdate
from matplotlib.figure import Figure
from torch_geometric.data import Data

from morphoclass.vis import plot_morphology_images

logger = logging.getLogger(__name__)


def embed_latent_features(
    X,
    embedder=None,
):
    """Embed latent features to 2-dimensional space.

    Parameters
    ----------
    X : list_like
        Latent features of the dataset.
    embedder : sklearn.decomposition.PCA, umap, ..., optional
        The algorithm for dimensionality reduction.

    Returns
    -------
    embedding_x : np.ndarray
        The first dimension of the embedding.
    embedding_y : np.ndarray
        The second dimension of the embedding.
    """
    if embedder is None:
        embedder = sklearn.decomposition.PCA(n_components=2, svd_solver="arpack")

    # embed to a lower dimensional space
    X_2d = embedder.fit_transform(X)

    embedding_x = X_2d[:, 0]
    embedding_y = X_2d[:, 1]

    return embedding_x, embedding_y


def get_embeddings_figure(
    x_coordinates,
    y_coordinates,
    dataset,
    predictions,
    train_idx,
    val_idx,
    cleanlab_ordered_label_errors,
    cleanlab_self_confidence,
):
    """Get figure with latent feature embeddings in 2-dimensional space.

    Parameters
    ----------
    x_coordinates : np.ndarray
        The first dimension of the embedding. Shape (train_size + val_size,)
    y_coordinates : np.ndarray
        The second dimension of the embedding. Shape (train_size + val_size,)
    dataset : morphoclass.data.MorphologyEmbeddingDataset,
              morphoclass.data.MorphologyDataset
        Dataset with morphologies. Size train_size + val_size
    predictions : list_like
        The predicted values (both validation and training indices).
        Shape (train_size + val_size,)
    train_idx : list_like
        The indices of the training dataset.
    val_idx : list_like
        The indices of the validation dataset.
    cleanlab_ordered_label_errors : list_like
        Ordered label indices representing samples with wrong label.
    cleanlab_self_confidence : list_like
        Self confidence is the holdout probability that an example
        belongs to its given class label.

    Returns
    -------
    figure : plotly.graph_objects.Figure
        Figure with embeddings.

    """
    # only keep validation set predictions
    predictions_val = np.empty(len(dataset), dtype=object)
    if val_idx is not None:
        predictions_val[val_idx] = [
            dataset.y_to_label[prediction] for prediction in predictions[val_idx]
        ]

        cleanlab_indices = np.intersect1d(val_idx, cleanlab_ordered_label_errors)
        cleanlab_confidence = cleanlab_self_confidence[
            np.isin(cleanlab_ordered_label_errors, cleanlab_indices)
        ]
    else:
        cleanlab_indices = cleanlab_ordered_label_errors
        cleanlab_confidence = cleanlab_self_confidence

    df = pd.DataFrame(
        data={
            "morph_name": dataset.morph_names,
            "mtype": dataset.labels,
            "mtype_pred": predictions_val,
            "morph_path": dataset.morph_paths,
            "x": x_coordinates,
            "y": y_coordinates,
        }
    )
    df_cleanlab = df.iloc[cleanlab_indices]
    if val_idx is not None:
        df_val = df.iloc[val_idx]
    df_train = df.iloc[train_idx]

    # build figure
    figure = go.Figure()

    colors_dict = {
        morph_type: px.colors.qualitative.Plotly[i]
        for i, morph_type in enumerate(np.unique(dataset.labels))
    }

    if val_idx is not None:
        # true val
        for morph_type in dataset.label_to_y:
            dft = df_val[df_val["mtype"] == morph_type]

            figure.add_trace(
                go.Scatter(
                    marker_symbol="circle",
                    mode="markers",
                    x=dft["x"],
                    y=dft["y"],
                    name=f"{morph_type} | valication - ground truth",
                    marker={"color": colors_dict[morph_type], "size": 15},
                    showlegend=True,
                    text=dft["morph_name"],
                    hovertemplate="%{text}",
                )
            )

        # predicted val
        for morph_type in dataset.label_to_y:
            dft = df_val[df_val["mtype_pred"] == morph_type]

            figure.add_trace(
                go.Scatter(
                    marker_symbol="circle-open",
                    mode="markers",
                    x=dft["x"],
                    y=dft["y"],
                    name=f"{morph_type} | validation - prediction",
                    marker={
                        "color": colors_dict[morph_type],
                        "size": 25,
                        "line": {"width": 4},
                    },
                    showlegend=True,
                    text=dft["morph_name"],
                    hovertemplate="%{text}",
                )
            )

    # true train
    for morph_type in dataset.label_to_y:
        dft = df_train[df_train["mtype"] == morph_type]

        figure.add_trace(
            go.Scatter(
                marker_symbol="square",
                mode="markers",
                x=dft["x"],
                y=dft["y"],
                name=f"{morph_type} | train - ground truth",
                marker={"color": colors_dict[morph_type], "size": 15},
                showlegend=True,
                text=dft["morph_name"],
                hovertemplate="%{text}",
            )
        )

    # cleanlab part
    figure.add_trace(
        go.Scatter(
            marker_symbol="cross",
            mode="markers",
            x=df_cleanlab["x"],
            y=df_cleanlab["y"],
            name="Cleanlab Detected Outlier",
            marker={"color": "black", "size": 11},
            showlegend=True,
            text=[
                f"self-confidence {confidence*100:.1f} %, {morph_name}"
                for morph_name, confidence in zip(
                    df_cleanlab["morph_name"], cleanlab_confidence
                )
            ],
            hovertemplate="%{text}",
        )
    )

    # highlighter
    for i, (x, y, morph_name) in enumerate(zip(df["x"], df["y"], df["morph_name"])):
        figure.add_trace(
            go.Scatter(
                marker_symbol="octagon",
                mode="markers",
                x=[x],
                y=[y],
                marker={"color": "red", "size": 40, "line": {"width": 2}, "opacity": 0},
                text=morph_name,
                showlegend=False,
                hoverinfo="none",
                name=f"highlighter-{i}",
            )
        )

    # layout and axes
    figure.update_layout(
        height=500,
        width=900,
        paper_bgcolor="white",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 0, "r": 0, "b": 0, "pad": 0},
        title_text="Embeddings",
        font={"size": 15},
        yaxis={"position": 0.0},
    )
    margin = 0.2
    figure.update_xaxes(
        title="PC 1",
        showline=True,
        linecolor="lightgrey",
        gridcolor="lightgrey",
        zeroline=True,
        zerolinecolor="lightgrey",
        range=(df["x"].min() - margin, df["x"].max() + margin),
    )
    figure.update_yaxes(
        title="PC 2",
        showline=True,
        linecolor="lightgrey",
        gridcolor="lightgrey",
        zeroline=True,
        zerolinecolor="lightgrey",
        range=(df["y"].min() - margin, df["y"].max() + margin),
    )

    return figure


def get_images(dataset):
    """Get information images of morphologies for click event.

    Parameters
    ----------
    dataset : morphoclass.data.MorphologyEmbeddingDataset,
              morphoclass.data.MorphologyDataset
        Dataset with morphologies.

    Returns
    -------
    image_data_sources : dict
        Dictionary with morphology name key and encoded image value.
    """
    # collect images for click event
    image_data_sources = {}

    for neuron_path, neuron_type, neuron_name in zip(
        dataset.morph_paths, dataset.labels, dataset.morph_names
    ):
        neuron_path = neuron_path.resolve()
        image_file = io.BytesIO()
        neuron = tmd.io.load_neuron(str(neuron_path))
        figure = plot_morphology_images(
            neuron,
            neuron_type,
            neuron_name,
            figsize=(12, 4),
        )
        figure.savefig(image_file, format="png", bbox_inches="tight")
        encoded_image = (
            base64.b64encode(image_file.getvalue()).decode("utf-8").replace("\n", "")
        )

        image_data_sources[neuron_name] = f"data:image/png;base64,{encoded_image}"

    # as a dummy take the last image
    image_data_sources["dummy"] = image_data_sources[neuron_name]

    return image_data_sources


def generate_empty_image() -> Figure:
    """Generate empty image as a placeholder.

    Returns
    -------
    matplotlib.figure.Figure
        An empty matplotlib figure.
    """
    figure = Figure(figsize=(12, 4), dpi=50)
    return figure


def generate_image(sample: Data) -> Figure:
    """Generate image based on morphology.

    Parameters
    ----------
    sample
        A morphology data sample.

    Returns
    -------
    matplotlib.figure.Figure
        A matplotlib figure with the morphology plot.
    """
    neuron_path = sample.path
    neuron_type = sample.y_str
    neuron_name = pathlib.Path(sample.path).stem

    neuron_path = neuron_path.resolve()
    neuron = tmd.io.load_neuron(str(neuron_path))
    figure = plot_morphology_images(
        neuron,
        neuron_type,
        neuron_name,
        figsize=(12, 4),
    )
    return figure


def get_outlier_detection_app(
    figures,
    dataset,
    description_text=None,
):
    """Visualize outlier detection in dash app for interactivity.

    Parameters
    ----------
    figures : list
        List of plotly figures.
    dataset : morphoclass.morphology_dataset.MorphologyDataset
        Dataset with neuronal morphologies.
    description_text : str, optional
        Checkpoint information.

    Returns
    -------
    app : dash.Dash
        Application instance.
    """
    # start interactive app
    app = dash.Dash(title="Outlier Detection", external_stylesheets=[dbc.themes.COSMO])

    # The first figure is model trained on all samples,
    # the following figures are model splits
    number_of_splits = len(figures)

    encoded_image = io.BytesIO()
    figure = generate_empty_image()
    figure.savefig(encoded_image, format="png", bbox_inches="tight")
    encoded_image_str = (
        base64.b64encode(encoded_image.getvalue()).decode("utf-8").replace("\n", "")
    )
    image_cache = {"empty": f"data:image/png;base64,{encoded_image_str}"}

    html_blocks = []
    # generate n callbacks (i.e. 1 callback for every split)
    for split in range(number_of_splits):

        @app.callback(
            [
                Output(f"graph-{split}", "figure"),
                Output(f"images-{split}", "src"),
                Output(f"images-{split}", "width"),
                Output(f"images-{split}", "height"),
            ],
            [
                Input(f"graph-{split}", "clickData"),
                State(f"graph-{split}", "figure"),
            ],
            prevent_initial_call=True,
        )
        def update_image(click_data, figure):
            """Update the image on click.

            Parameters
            ----------
            click_data : dict
                Click data for the graph click event.
            figure : dict
                The figure data of the clicked graph.

            Returns
            -------
            figure : dict
                The updated figure data.
            src : str
                The encoded morphology plot image data.
            width : int
                The morphology plot width.
            height : int
                The morphology plot height.
            """
            logger.debug("Graph click callback called")

            # get the morphology name that corresponds to the clicked point
            if len(click_data["points"]) > 0 and "text" in click_data["points"][0]:
                *_, morph_name = click_data["points"][0]["text"].rpartition(", ")
            else:
                morph_name = None
            logger.debug("Morph name: %s", morph_name)

            # highlight the appropriate point in the scatter plot
            for data in figure["data"]:
                if data["name"].startswith("highlighter"):
                    if data["text"] == morph_name:
                        data["marker"]["opacity"] = 0.3
                    else:
                        data["marker"]["opacity"] = 0

            # if no morphology selected then hide the plots
            if morph_name is None:
                logger.debug("No morphology was selected - hiding the plot")
                return figure, image_cache["empty"], 0, 0

            # if the figure is not found in the cache then generate it
            if morph_name not in image_cache:
                logger.debug("Image not found in cache, generating...")
                logger.debug("> Getting the sample...")
                sample = dataset.get_sample_by_morph_name(morph_name)
                if sample is None:
                    logger.debug(
                        "> Sample not found - figure image cannot be generated"
                    )
                    raise PreventUpdate
                logger.debug("> Sample found, generating figure image")
                encoded_image = io.BytesIO()
                figure_clicked = generate_image(sample)
                figure_clicked.savefig(encoded_image, format="png", bbox_inches="tight")
                encoded_image_str = (
                    base64.b64encode(encoded_image.getvalue())
                    .decode("utf-8")
                    .replace("\n", "")
                )
                image_cache[morph_name] = f"data:image/png;base64,{encoded_image_str}"

            logger.debug("Callback finished, returning the new plot image")
            return figure, image_cache[morph_name], 1000, 300

        if split == 0:
            header = "Embeddings of model trained on whole dataset with no predictions"
        else:
            header = f"Embeddings of model split {split}"

        html_blocks.extend(
            [
                dhc.Hr(),
                dhc.Br(),
                dhc.Div(
                    [
                        dhc.Br(),
                        dhc.H3(header),
                        dhc.Br(),
                        dhc.Div(id=f"num-split-{split}", children=split, hidden=True),
                        dcc.Graph(
                            id=f"graph-{split}", figure=go.FigureWidget(figures[split])
                        ),
                        dcc.Loading(
                            dhc.Img(
                                id=f"images-{split}",
                                src=image_cache["empty"],
                                n_clicks_timestamp=0,
                                width=0,
                                height=0,
                                hidden=False,
                            ),
                            type="graph",
                        ),
                    ]
                ),
                dhc.Br(),
            ]
        )

    html_all = [
        dhc.Br(),
        dhc.Br(),
        dhc.H2("Outlier Detection - Embeddings"),
        dhc.Br(),
        dcc.Markdown(description_text or "", style={"textAlign": "left"}),
        dhc.Br(),
        *html_blocks,
    ]

    app.layout = dbc.Container(
        dbc.Row(dbc.Col(html_all)),
        style={
            "verticalAlign": "middle",
            "textAlign": "center",
            "backgroundColor": "rgb(255, 255, 255)",
            "position": "relative",
            "width": "100%",
            "bottom": "0px",
            "left": "0px",
            "zIndex": "1000",
        },
    )

    return app

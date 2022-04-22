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
"""XAI subcommands."""
from __future__ import annotations

import logging
import os
import pathlib
import tempfile
from datetime import datetime

import click

logger = logging.getLogger(__name__)


@click.group(
    name="xai",
    short_help="Explain model predictions",
    help="""
    Use explainable AI (XAI) to understand the model predictions.

    Please use one of the available commands listed below.
    """,
    add_help_option=False,
)
def cli():
    """Run the "xai" subcommand."""
    pass


@cli.command(name="report", help="Create an XAI report")
@click.option(
    "--checkpoint-path",
    type=click.Path(dir_okay=False),
    required=True,
    help="Path to a model checkpoint.",
)
@click.option(
    "--results-file",
    required=True,
    type=click.Path(dir_okay=False),
    help="The HTML report output path.",
)
def report(results_file: str | os.PathLike, checkpoint_path: str | os.PathLike) -> None:
    """Create an XAI report.

    Parameters
    ----------
    results_file
        The HTML report output path.
    checkpoint_path
        Path to a model checkpoint.
    """
    logger.info("Loading libraries and modules")
    import morphoclass.xai.report
    from morphoclass.training.training_log import TrainingLog

    logger.info("Loading the checkpoint")
    training_log = TrainingLog.load(pathlib.Path(checkpoint_path))

    logger.info("Creating the XAI report")
    morphoclass.xai.report.make_report(
        results_file,
        training_log=training_log,
    )


@cli.command(
    name="single",
    deprecated=True,
    short_help="Explain the GNN prediction for a single morphology",
    help="""
    Explains the prediction of a trained GNN on a single morphology file.

    This command was developed a long time ago and has been superseded by
    the "report" command. If there is need for single morphology GNN
    explanation, then this command should be re-implemented. In its current
    state it's unmaintained and probably not working.
    """,
)
@click.option(
    "-i",
    "--input-file",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="The path to the input morphology file",
)
@click.option(
    "-c",
    "--checkpoint",
    "checkpoint_file",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="The path to the pre-trained model checkpoint.",
)
@click.option(
    "-o",
    "--output-dir",
    required=True,
    type=click.Path(exists=False, file_okay=False, writable=True),
    help="Output directory for the trained model.",
)
@click.option(
    "-n",
    "--results-name",
    required=False,
    type=click.STRING,
    help="The filename of the results file",
)
def single(input_file, checkpoint_file, output_dir, results_name):
    """Run the `morphoclass explain` CLI command.

    Parameters
    ----------
    input_file
        Input morphology file.
    checkpoint_file
        Model checkpoint file.
    output_dir
        Directory for writing results.
    results_name
        File prefix for results output files.
    """
    input_file = pathlib.Path(input_file).resolve()
    output_dir = pathlib.Path(output_dir).resolve()
    checkpoint_file = pathlib.Path(checkpoint_file).resolve()
    if results_name is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_name = f"xai_{input_file.stem}_{timestamp}"

    click.secho(f"Input file   : {input_file}", fg="yellow")
    click.secho(f"Checkpoint   : {checkpoint_file}", fg="yellow")
    click.secho(f"Output files : {output_dir / results_name}_[...].png", fg="yellow")

    node_saliency_file = output_dir / (results_name + "_node_saliency.png")
    node_heatmap_file = output_dir / (results_name + "_node_heatmap.png")
    if not check_overwrite_file(node_saliency_file):
        click.secho("Stopping.", fg="red")
        return
    if not check_overwrite_file(node_heatmap_file):
        click.secho("Stopping.", fg="red")
        return

    click.secho("✔ Loading checkpoint...", fg="green", bold=True)
    import torch

    checkpoint = torch.load(checkpoint_file)
    model_name = checkpoint["model_name"]
    click.secho(f"Model        : {model_name}", fg="yellow")
    if "metadata" in checkpoint:
        timestamp = checkpoint["metadata"]["timestamp"]
        click.secho(f"Created on   : {timestamp}", fg="yellow")

    if model_name != "GNN":
        click.secho(
            "XAI only available for GNNs, choose a different checkpoint.", fg="red"
        )
        click.secho("Stopping.", fg="red")
        return

    click.secho("✔ Loading libraries...", fg="green", bold=True)
    import torch

    import morphoclass as mc
    import morphoclass.models
    import morphoclass.transforms
    import morphoclass.xai
    from morphoclass.data.morphology_data_loader import MorphologyDataLoader

    click.secho("✔ Loading model...", fg="green", bold=True)
    model_cls = getattr(mc.models, checkpoint["model_class"])
    model = model_cls(**checkpoint["model_params"])
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    click.secho("✔ Loading data...", fg="green", bold=True)

    import numpy as np

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = pathlib.Path(tmpdir_str)
        morphology_csv = tmpdir / "morphology.csv"
        with open(morphology_csv, "w") as f:
            f.write(f"{input_file}\n")
        scaler_config = checkpoint["scaler_config"]
        fitted_scaler = mc.transforms.scaler_from_config(scaler_config)
        dataset, dataset_pi, fitted_scaler = get_datasets(morphology_csv, fitted_scaler)
    assert len(dataset) == 1

    click.secho("✔ Explaining...", fg="green", bold=True)
    sample = dataset[0]
    explainer = mc.xai.GradCAMExplainer(model, model.embedder.bi2)
    logits, cam = explainer.get_cam(
        sample,
        loader_cls=MorphologyDataLoader,
        cls_idx=None,
        relu_weights=False,
        relu_cam=True,
    )
    cam = cam / np.abs(cam).max()

    click.secho("✔ Creating figures...", fg="green", bold=True)
    from matplotlib.figure import Figure
    from scipy.stats import gaussian_kde

    import morphoclass.vis

    figsize = np.array([6, 6])
    dpi = 60
    dx, dy = figsize * dpi * 1j
    pred_class = checkpoint["class_dict"][logits.argmax().item()]

    fig = Figure(figsize=figsize)
    ax = fig.subplots()
    fig.suptitle(f"Predicted class: {pred_class}\nNode Saliency")
    tree = sample.tmd_neurites[0]
    morphoclass.xai.plot_node_saliency(
        tree, cam, ax=ax, name="Grad-CAM", show_legend=np.any(cam < 0)
    )
    fig.savefig(node_saliency_file)

    # Generate and plot heatmap
    fig = Figure(figsize=figsize)
    ax = fig.subplots()
    fig.suptitle(f"Predicted class: {pred_class}\nGrad-CAM as Heatmap")
    morphoclass.vis.plot_tree(tree, ax, node_size=1.0, edge_color="orange")
    node_pos = np.array([tree.x, tree.y])
    kde_weights = np.maximum(0, cam)
    kernel = gaussian_kde(node_pos, weights=kde_weights)

    pxmin = tree.x.min()
    pxmax = tree.x.max()
    pymin = tree.y.min()
    pymax = tree.y.max()

    x, y = np.mgrid[pxmin:pxmax:dx, pymin:pymax:dy]
    positions = np.vstack([x.ravel(), y.ravel()])
    z = np.reshape(kernel(positions).T, x.shape)
    ax.imshow(
        np.rot90(z),
        cmap="inferno",
        aspect="auto",
        extent=[pxmin, pxmax, pymin, pymax],
        alpha=0.5,
    )
    fig.savefig(node_heatmap_file)

    click.secho("✔ Done.", fg="green", bold=True)


def check_overwrite_file(file_path):
    """Ask user if an existing file can be overwritten.

    Parameters
    ----------
    file_path
        A file path that shall be overwritten.

    Returns
    -------
    bool
        Whether or not the file can be overwritten.
    """
    if file_path.exists():
        msg = f'File "{file_path}" exists, overwrite? (y/[n]) '
        click.secho(msg, fg="red", bold=True, nl=False)
        response = input()
        if response.strip().lower() != "y":
            return False
        else:
            click.secho("You chose to overwrite, proceeding...", fg="red")
            return True
    return True


def get_datasets(input_csv, fitted_scaler=None):
    """Load datasets from a CSV file."""
    import morphoclass as mc
    import morphoclass.data
    import morphoclass.training
    import morphoclass.transforms

    pre_transform = mc.transforms.Compose(
        [
            mc.transforms.ExtractTMDNeurites(neurite_type="apical"),
            mc.transforms.OrientApicals(
                special_treatment_ipcs=False, special_treatment_hpcs=False
            ),
            mc.transforms.BranchingOnlyNeurites(),
            mc.transforms.ExtractEdgeIndex(),
        ]
    )

    dataset = mc.data.MorphologyDataset.from_csv(
        csv_file=input_csv, pre_transform=pre_transform
    )

    if dataset.guess_layer() in [2, 6]:
        feature = "projection"
        feature_extractor = mc.transforms.ExtractVerticalDistances(
            vertical_axis="y", negative_ipcs=False, negative_bpcs=False
        )
    else:
        feature = "radial_distances"
        feature_extractor = mc.transforms.ExtractRadialDistances(
            negative_ipcs=False, negative_bpcs=False
        )
    logger.info(f"Using feature: {feature}")

    if len(dataset) > 0:
        transform, fitted_scaler = mc.training.make_transform(
            dataset=dataset,
            feature_extractor=feature_extractor,
            n_features=1,
            fitted_scaler=fitted_scaler,
        )
        dataset.transform = transform
        dataset_pi = to_persistence_dataset(dataset, feature=feature)
    else:
        dataset_pi = [None, None, None]
        fitted_scaler = None

    return dataset, dataset_pi, fitted_scaler


def to_persistence_dataset(dataset, feature="radial_distances"):
    """Create a persistence dataset from MorphologyDataset.

    Parameters
    ----------
    dataset
        an instance of `MorphologyDataset`
    feature
        which feature to use as a filtration function. The default
         feature 'radial_distances' should be used for all datasets
         except those containing inverted cells (IPCs). For the
         inverted cells the feature 'projection' might be more useful
         as it retains information about orientation. Note, however,
         that 'projection' is not rotation-invariant, so the cells
         have to be properly oriented. For more information on this
         parameter see `get_persistence_diagram` in the `tmd` package.

    Returns
    -------
        diagrams
            persistence diagrams
        images
            persistence images
        labels
            labels
    """
    import numpy as np
    from tmd.Topology.analysis import get_limits
    from tmd.Topology.analysis import get_persistence_image_data
    from tmd.Topology.methods import get_persistence_diagram

    # Get the labels
    labels = np.array([sample.y for sample in dataset])

    # Compute persistence diagrams
    diagrams = [
        get_persistence_diagram(sample.tmd_apicals[0], feature=feature)
        for sample in dataset
    ]
    diagrams = [np.array(diagram) for diagram in diagrams]

    # Compute persistence diagrams limits
    xlims, ylims = get_limits(diagrams)

    # Compute persistence images
    images = [
        get_persistence_image_data(pd, xlims=xlims, ylims=ylims) for pd in diagrams
    ]
    images = [np.rot90(img) for img in images]
    images = np.array(images)

    return diagrams, images, labels

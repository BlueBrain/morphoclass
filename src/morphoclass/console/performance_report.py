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
"""Utilities for the morphoclass performance-report command."""
from __future__ import annotations

import logging
import pathlib
import shutil
import textwrap

import click
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from pandas.io.formats.style import Styler

import morphoclass as mc
import morphoclass.report.plumbing
from morphoclass.data import MorphologyDataset
from morphoclass.types import StrPath
from morphoclass.utils import dict2kwargs
from morphoclass.xai.embedding_visualization import embed_latent_features
from morphoclass.xai.embedding_visualization import generate_empty_image
from morphoclass.xai.embedding_visualization import generate_image
from morphoclass.xai.embedding_visualization import get_embeddings_figure

logger = logging.getLogger(__name__)


def make_performance_report(checkpoint_dir: StrPath, output_dir: StrPath) -> None:
    """Create a performance report for trained models.

    Parameters
    ----------
    checkpoint_dir
        The directory with all checkpoints to be included in the report.
    output_dir
        The report output directory.
    """
    # Resolve paths from parameters
    checkpoint_dir = pathlib.Path(checkpoint_dir).resolve()
    output_dir = pathlib.Path(output_dir).resolve()
    output_dir.mkdir(exist_ok=True, parents=True)

    # Additional output paths
    results_file = output_dir / "results_table.html"
    best_models_path = output_dir / "best_models.txt"
    dataset_img_dir = output_dir / "images"
    dataset_img_dir.mkdir(exist_ok=True, parents=True)
    results_table_data_dir = output_dir / "results_table_data"
    results_table_data_dir.mkdir(exist_ok=True, parents=True)

    # read checkpoint files that will be summarized in this report
    files = [path for path in checkpoint_dir.glob("*.chk") if path.is_file()]
    if len(files) == 0:
        raise click.ClickException(
            f"There are no files in {checkpoint_dir}, "
            "please fix the input arguments!"
        )
    logger.info(f"Found {len(files)} checkpoint files")

    cached_figures: dict[str, dict[str, Figure]] = {}
    table_rows = []
    for i, metrics_file in enumerate(files):
        logger.info(
            f"Processing checkpoint {i + 1} of {len(files)}: {metrics_file.name!r}"
        )
        data = torch.load(metrics_file)

        images_directory = pathlib.Path(data["images_directory"])
        model_name = images_directory.stem
        model_report_directory = results_table_data_dir / model_name
        model_report_images_directory = model_report_directory / "images"
        model_report_images_directory.mkdir(exist_ok=True, parents=True)
        detected_outliers_file = model_report_directory / "detected_outliers.html"
        latent_features_results_file = model_report_directory / "latent_features.html"
        performance_report_file = model_report_directory / "performance_report.html"

        # move images to report directory
        for image_file in images_directory.glob("*.png"):
            shutil.copy(
                image_file,
                model_report_images_directory / image_file.name,
            )

        data["metrics_file"] = metrics_file
        data["model_report_directory"] = (
            pathlib.Path("results_table_data") / model_report_directory.stem
        )
        data["results_table_file"] = results_file.name
        data["latent_features_file"] = latent_features_results_file.name
        data["performance_report_file"] = performance_report_file.name
        data["detected_outliers_file"] = detected_outliers_file.name

        # collect outliers report
        if data["dataset_name"] not in cached_figures:
            cached_figures[data["dataset_name"]] = {}
        visualize_cleanlab_outliers(
            data,
            detected_outliers_file,
            cached_figures[data["dataset_name"]],
            dataset_img_dir,
        )
        visualize_latent_features(data, latent_features_results_file)
        visualize_model_performance(data, performance_report_file)
        table_rows.append(make_report_row(data))

    df = pd.DataFrame(table_rows)
    # sort dataframe by f1 score and dataset
    df = df.sort_values(by=["dataset", "f1_weighted"], ascending=[True, False])
    df = df.reset_index(drop=True).reset_index()
    # map the links to corresponding model HTML

    def make_links(model_report_dir):
        metrics_link = (
            f"<a href='{model_report_dir}/performance_report.html'>Metrics</a>"
        )
        outliers_link = (
            f"<a href='{model_report_dir}/detected_outliers.html'>Outliers</a>"
        )
        latent_features_link = (
            f"<a href='{model_report_dir}/latent_features.html'>Latent Features</a>"
        )
        return f"{metrics_link} - {outliers_link} - {latent_features_link}"

    df["details"] = df["model_report_directory"].map(make_links)

    # collect paths to best performing models
    mask = df.duplicated(subset="dataset", keep="first")
    df_best_models = df[~mask]["metrics_file"]
    df_best_models.to_csv(best_models_path, header=False, index=False)

    # checkpoint name
    df["checkpoint_name"] = df["metrics_file"].apply(lambda x: pathlib.Path(x).stem)
    # work on table with performances
    # the same order will be in the report
    table_columns = [
        "details",
        "dataset",
        "feature_extractor",
        "model_class",
        "oversampled",
        "splitter_class",
        "accuracy",
        "f1_micro",
        "f1_macro",
        "f1_weighted",
        "checkpoint_name",
    ]
    df = df[table_columns]

    # Apply styling to the HTML table
    # Remove the random UUID from the HTML table "id" property and also
    # don't include the "id" properties in the cells. This makes the table
    # generation reproducible and removes unnecessary HTML garbage.
    # Ref:
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html#1.-Remove-UUID-and-cell_ids
    df_style = Styler(df, uuid_len=0, cell_ids=False)
    cell_style = "padding-right: 1em; white-space: nowrap;"
    df_style.set_table_styles([{"selector": "th, td", "props": cell_style}])

    # generate a HTML report
    template = mc.report.plumbing.load_template("results-table")
    mc.report.plumbing.render(template, {"df_results": df_style.render()}, results_file)
    logger.info(f"Report stored in: {results_file}")

    logger.info("Done.")


def make_report_row(data: dict) -> dict:
    """Create a row for the summary report table.

    Parameters
    ----------
    data
        The checkpoint data.

    Returns
    -------
    dict
        The dictionary representing a data frame row.
    """
    return {
        "dataset": data["dataset_name"],
        "feature_extractor": data["feature_extractor_name"],
        "model_class": data["model_class"],
        "model_params": data["model_params"],
        "oversampled": bool(data["oversampling"]),
        "splitter_class": data["splitter_class"].rpartition(".")[2],
        "splitter_params": data["splitter_params"],
        # "accuracy_mean": data["accuracy_mean"],
        "accuracy": f"{data['accuracy_mean']:.3f}±{data['accuracy_std']:.2f}",
        # "f1_weighted_mean": data["f1_weighted_mean"],
        "f1_weighted": f"{data['f1_weighted_mean']:.3f}±{data['f1_weighted_std']:.2f}",
        "f1_micro": f"{data['f1_micro_mean']:.3f}±{data['f1_micro_std']:.2f}",
        "f1_macro": f"{data['f1_macro_mean']:.3f}±{data['f1_macro_std']:.2f}",
        "metrics_file": data["metrics_file"],
        "model_report_directory": data["model_report_directory"],
    }


def visualize_model_performance(data: dict, results_file: pathlib.Path) -> None:
    """Visualize cleanlab outliers.

    Parameters
    ----------
    data
        Dictionary with checkpoint information.
    results_file
        Path to results file
    """

    def _images_path(x):
        return (pathlib.Path("images") / pathlib.Path(x).stem).with_suffix(".png")

    new_line = "\n"
    # model report on all samples
    classification_report_all = data["classification_report"]
    confusion_matrix_all = _images_path(data["confusion_matrix"])
    report_all = textwrap.dedent(
        f"""
    <div class="row">
        <br/><hr/><br/>
        <div class='col-md-6'>
            <br/><h5>Average performance scores:</h5><br/>
                {classification_report_all.replace(new_line, "<br/>")
                .replace(" ", "&nbsp;"*2)}
        </div>
        <div class='col-md-6'>
            <img src=
            'file:{confusion_matrix_all}'
            width='80%'>
        </div>
    </div>
    """
    ).strip()

    # models report per split
    report_per_split_str = ""
    # ignore per model statistics if the splitter is LOO
    if data["splitter_class"] != "sklearn.model_selection.LeaveOneOut":
        report_per_split = []
        for s, split in enumerate(data["splits"]):
            classification_report_split = split["classification_report"]
            confusion_matrix_split = split["confusion_matrix"]

            report_per_split.append(
                textwrap.dedent(
                    f"""
                <br/>
                <div class="row">
                    <div class='col-md-6'>
                        <h5>Split {s}/{len(data["splits"])}</h5><br/>
                        {classification_report_split
                        .replace(new_line, "<br/>")
                        .replace(" ", "&nbsp;"*2)}
                    </div>
                    <div class='col-md-6'>
                        <img src=
                        'file:{_images_path(confusion_matrix_split)}'
                        width='80%'>
                    </div>
                </div>"""
                ).strip()
            )
        # join into one HTML string
        report_per_split_str = "<br/><br/>".join(report_per_split)

    report_html = report_all + report_per_split_str

    result_performance_images = ""
    if data.get("accuracy_image"):
        result_performance_images = (
            "<img src='file:" f"{_images_path(data['accuracy_image'])}' " "width='90%'>"
        )

    morphoclass_model_info = ""
    if data["n_epochs"] is not None:
        morphoclass_model_info = f"""num. epochs: <b>{data['n_epochs']}</b><br/>
            optimizer: <b>{data['optimizer_class']}
            {'('+dict2kwargs(data['optimizer_params'])+')'
            if dict2kwargs(data['optimizer_params'])
            else ''}</b><br/>"""

    result_parameters = f"""
        Checkpoint: <b>{data['metrics_file']}</b><br/>
        F1-score weighted: <b>{data['f1_weighted']}</b><br/>
        F1-score micro: <b>{data['f1_micro']}</b><br/>
        F1-score macro: <b>{data['f1_macro']}</b><br/>
        accuracy: <b>{data['accuracy']}</b><br/>
        dataset: <b>{data['dataset_name']}</b><br/>
        feature extractor: <b>{data['feature_extractor_name']}</b><br/>
        model class:
        <b>{data['model_class']}({dict2kwargs(data['model_params'])})</b><br/>
        oversampling:
        <b>{bool(data['oversampling'])}</b><br/>
        splitter_class:
        <b>{data['splitter_class']}({dict2kwargs(data['splitter_params'])})</b><br/>
        {morphoclass_model_info}
    """

    # generate an HTML report
    template = mc.report.plumbing.load_template("performance-report")
    template_vars = {
        "result_parameters": result_parameters,
        "result_performance_images": result_performance_images,
        "result_report": report_html,
    }
    results_file.parent.mkdir(exist_ok=True, parents=True)
    mc.report.plumbing.render(template, template_vars, results_file)


def visualize_cleanlab_outliers(data, results_file, cached_figures, image_directory):
    """Visualize cleanlab outliers.

    Parameters
    ----------
    data : dict
        Dictionary with checkpoint information.
    results_file : pathlib.Path, str
        Path to results file
    cached_figures : dict
        Dictionary with morphology figures generated so far.
    image_directory : pathlib.Path
        The path with the model evaluation images.
    """
    cleanlab_ordered_label_errors = list(data["cleanlab_ordered_label_errors"])
    cleanlab_self_confidence = data["cleanlab_self_confidence"]

    # no need for feature extraction, we will only handle morphologies anyway
    dataset = MorphologyDataset.from_csv(data["input_csv"])

    all_probabilities = data["probabilities"]
    proposed_predictions = []
    for probabilities_per_sample in all_probabilities:
        proposed_predictions_per_sample = []
        for i, probability in enumerate(probabilities_per_sample):
            proposed_predictions_per_sample.append((dataset.y_to_label[i], probability))
        proposed_predictions_per_sample.sort(
            key=lambda x: float(x[1]),
            reverse=True,
        )
        proposed_predictions_per_sample_str = ", ".join(
            f"{m_type}: {probability*100:.1f} %"
            for m_type, probability in proposed_predictions_per_sample
        )
        proposed_predictions.append(proposed_predictions_per_sample_str)

    morph_names_cleanlab = np.array(dataset.morph_names)[cleanlab_ordered_label_errors]
    image_data_sources_cleanlab = []
    for morph_name in morph_names_cleanlab:
        if morph_name not in cached_figures:
            sample = dataset.get_sample_by_morph_name(morph_name)
            if sample is None:
                fig = generate_empty_image()
            else:
                fig = generate_image(sample)
            cached_figures[morph_name] = fig
        fig = cached_figures[morph_name]
        file_name = f"{morph_name}.png"
        fig.savefig(image_directory / file_name)
        image_data_sources_cleanlab.append(f"./../../images/{file_name}")

    expert_labels_cleanlab = np.array(dataset.labels)[cleanlab_ordered_label_errors]
    cleanlab_outliers = ",<br/>".join(
        f"""<b>{name.rpartition('/')[2]}</b> belongs to <b>{true_label}</b>
            with confidence <b>{confidence*100:.1f} %</b><br/>
            proposed labels and their confidence: {proposed_prediction}
            <img src='{img}' width='80%'>
        """
        for name, true_label, confidence, img, proposed_prediction in zip(
            morph_names_cleanlab,
            expert_labels_cleanlab,
            cleanlab_self_confidence,
            image_data_sources_cleanlab,
            proposed_predictions,
        )
    )
    cleanlab_outliers_html = f"""
        * Self confidence is the holdout probability that an example
        belongs to its given class label.
        * List of detected outliers using cleanlab:<br/><br/>
        {cleanlab_outliers}
    """
    # generate an HTML report
    template = mc.report.plumbing.load_template("detected-outliers")
    template_vars = {"cleanlab_outliers_html": cleanlab_outliers_html}
    results_file.parent.mkdir(exist_ok=True, parents=True)
    mc.report.plumbing.render(template, template_vars, results_file)


def visualize_latent_features(data, results_file):
    """Visualize latent features for report.

    Parameters
    ----------
    data : dict
        Dictionary with checkpoint information.
    results_file : pathlib.Path, str
        Path to results file
    """
    cleanlab_ordered_label_errors = list(data["cleanlab_ordered_label_errors"])
    cleanlab_self_confidence = data["cleanlab_self_confidence"]

    # no need for feature extraction, we will only handle morphologies anyway
    dataset = MorphologyDataset.from_csv(data["input_csv"])

    # get figure with embeddings
    x_coordinates, y_coordinates = embed_latent_features(
        data["all"]["latent_features"],
    )

    fig_whole_dataset = get_embeddings_figure(
        x_coordinates=x_coordinates,
        y_coordinates=y_coordinates,
        dataset=dataset,
        predictions=None,
        train_idx=range(len(dataset)),
        val_idx=None,
        cleanlab_ordered_label_errors=cleanlab_ordered_label_errors,
        cleanlab_self_confidence=cleanlab_self_confidence,
    )

    # collect figures per split
    split_figs = []
    for split in data["splits"]:
        # get indices present in validation set and cleanlab error detection
        val_indices = split["val_idx"]
        cleanlab_indices = np.intersect1d(val_indices, cleanlab_ordered_label_errors)
        cleanlab_confidence = cleanlab_self_confidence[
            np.isin(cleanlab_ordered_label_errors, cleanlab_indices)
        ]
        # get figure with embeddings
        x_coordinates, y_coordinates = embed_latent_features(split["latent_features"])

        fig = get_embeddings_figure(
            x_coordinates=x_coordinates,
            y_coordinates=y_coordinates,
            dataset=dataset,
            predictions=data["predictions"],
            train_idx=split["train_idx"],
            val_idx=split["val_idx"],
            cleanlab_ordered_label_errors=cleanlab_indices,
            cleanlab_self_confidence=cleanlab_confidence,
        )
        split_figs.append(fig)

    # TODO: to_html produces a <div> with a random UUID as id. In newer versions
    #   of plotly it's possible to override it via the `div_id` parameters, but
    #   the plotly we currently use doesn't have this parameter yet.
    latent_features_html = (
        "<br/><br/><h5>Model trained on the whole dataset</h5><br/>"
        + fig_whole_dataset.to_html(full_html=False, include_plotlyjs="cdn")
    )
    # don't collect latent feature plots if the splitter is LOO
    if data["splitter_class"] != "sklearn.model_selection.LeaveOneOut":
        latent_features_html += "<br/><br/>".join(
            f"<h5>Split {i + 1}/{len(split_figs)}</h5><br/>"
            + fig.to_html(full_html=False, include_plotlyjs="cdn")
            for i, fig in enumerate(split_figs)
        )

    # generate an HTML cleanlab report
    template = mc.report.plumbing.load_template("latent-features")
    template_vars = {"latent_features_html": latent_features_html}
    results_file.parent.mkdir(exist_ok=True, parents=True)
    mc.report.plumbing.render(template, template_vars, results_file)

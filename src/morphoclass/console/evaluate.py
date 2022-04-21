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
"""Utilities for the morphoclass evaluate command."""
from __future__ import annotations

import logging
import pathlib
import textwrap

import numpy as np

from morphoclass import report

# from morphoclass.console.train import split_metrics
from morphoclass.data import MorphologyDataset
from morphoclass.training.cli import collect_metrics
from morphoclass.training.training_log import TrainingLog
from morphoclass.vis import plot_confusion_matrix
from morphoclass.xai.embedding_visualization import embed_latent_features
from morphoclass.xai.embedding_visualization import generate_empty_image
from morphoclass.xai.embedding_visualization import generate_image
from morphoclass.xai.embedding_visualization import get_embeddings_figure

logger = logging.getLogger(__name__)


def visualize_model_performance(
    training_log: TrainingLog,
    checkpoint_path: pathlib.Path,
    img_dir: pathlib.Path,
    report_path: pathlib.Path,
) -> None:
    """Visualize cleanlab outliers."""
    logger.info("Computing the metrics")
    metrics = collect_metrics(
        training_log.targets,
        training_log.preds,
        training_log.labels_str,
    )
    # models report per split
    split_metrics = []
    for split_history in training_log.split_history:
        split_metrics_dict = collect_metrics(
            split_history["ground_truths"],
            split_history["predictions"],
            training_log.labels_str,
        )
        split_metrics.append(split_metrics_dict)
    # TODO: was this one used in any of the reports? If not then don't
    #       compute
    # split_metrics_dict = split_metrics(training_log.split_history)

    def str_params(params: dict) -> str:
        return ", ".join(f"{k}={v}" for k, v in params.items())

    logger.info("Collecting the checkpoint summary")
    result_parameters = f"""
        Checkpoint: <b>{checkpoint_path}</b><br/>
        F1-score weighted: <b>{metrics['f1_weighted']:.2%}</b><br/>
        F1-score micro: <b>{metrics['f1_micro']:.2%}</b><br/>
        F1-score macro: <b>{metrics['f1_macro']:.2%}</b><br/>
        accuracy: <b>{metrics['accuracy']:.2%}</b><br/>
        dataset: <b>{training_log.config.dataset_name}</b><br/>
        feature extractor: <b>{training_log.config.feature_extractor_name}</b><br/>
        model class:
        <b>
        {training_log.config.model_class}({str_params(training_log.config.model_params)})
        </b><br/>
        oversampling: <b>{training_log.config.oversampling}</b><br/>
        splitter_class:
        <b>
        {training_log.config.splitter_class}({str_params(training_log.config.splitter_params)})
        </b><br/>
    """
    result_parameters = textwrap.dedent(result_parameters)
    # If it's a deeplearning model then include optimizer info
    if training_log.config.n_epochs is not None:
        if training_log.config.optimizer_params:
            str_optimizer_params = str_params(training_log.config.optimizer_params)
            str_optimizer_params = f"({str_optimizer_params})"
        else:
            str_optimizer_params = ""
        extra_optim_info = f"""
        num. epochs: <b>{training_log.config.n_epochs}</b><br/>
        optimizer:
        <b>{training_log.config.optimizer_class}{str_optimizer_params}</b><br/>
        """
        extra_optim_info = textwrap.dedent(extra_optim_info)
        result_parameters += extra_optim_info

    def relative_to_html(img_path):
        return img_path.relative_to(report_path.parent)

    logger.info("Creating confusion matrices and classification reports")
    img_dir.mkdir(exist_ok=True, parents=True)

    # model report on all samples
    cm_path = img_dir / "confusion_matrix.png"
    plot_confusion_matrix(
        metrics["confusion_matrix"], cm_path, labels=training_log.labels_str
    )
    report_html = f"""
    <div class="row">
        <br/><hr/><br/>
        <div class='col-md-6'>
            <h5>Average performance scores:</h5>
                <pre>{metrics["classification_report"]}</pre>
        </div>
        <div class='col-md-6'>
            <img src='file:{relative_to_html(cm_path)}' width='80%'>
        </div>
    </div>
    """
    report_html = textwrap.dedent(report_html).strip()

    # ignore per model statistics if the splitter is LOO
    if training_log.config.splitter_class != "sklearn.model_selection.LeaveOneOut":
        split_reports = []
        for n, split_metrics_dict in enumerate(split_metrics):
            confusion_matrix_split = img_dir / f"confusion_matrix_split_{n}.png"
            plot_confusion_matrix(
                split_metrics_dict["confusion_matrix"],
                confusion_matrix_split,
                labels=training_log.labels_str,
            )
            split_report = f"""
            <div class="row">
                <div class='col-md-6'>
                    <h5>Split {n + 1}/{len(split_metrics)}</h5>
                    <pre>{split_metrics_dict["classification_report"]}</pre>
                </div>
                <div class='col-md-6'>
                    <img src=
                    'file:{relative_to_html(confusion_matrix_split)}'
                    width='80%'>
                </div>
            </div>"""
            split_reports.append(textwrap.dedent(split_report).strip())
        # Add to the overall report
        report_html += "<br/>".join(split_reports)

    # TODO: repair the accuracy plot
    # result_performance_images = ""
    # if data.get("accuracy_image"):
    #     result_performance_images = (
    #         "<img src='file:" f"{relative_to_html(data['accuracy_image'])}' "
    #         "width='90%'>"
    #     )

    logger.info(f"Rendering the performance report to {report_path.resolve().as_uri()}")
    template = report.load_template("performance-report")
    template_vars = {
        "result_parameters": result_parameters,
        # "result_performance_images": result_performance_images,
        "result_report": report_html,
    }
    report_path.parent.mkdir(exist_ok=True, parents=True)
    report.render(template, template_vars, report_path)


def visualize_latent_features(
    training_log: TrainingLog,
    cleanlab_errors: np.ndarray,
    cleanlab_self_confidence: np.ndarray,
    report_path: pathlib.Path,
) -> None:
    """Visualize latent features for report."""
    if training_log.all_history is None:
        raise ValueError("training_log.all_history is missing")

    # no need for feature extraction, we will only handle morphologies anyway
    dataset = MorphologyDataset.from_csv(training_log.config.input_csv)

    # get figure with embeddings
    x_coordinates, y_coordinates = embed_latent_features(
        training_log.all_history["latent_features"]
    )
    fig_whole_dataset = get_embeddings_figure(
        x_coordinates=x_coordinates,
        y_coordinates=y_coordinates,
        dataset=dataset,
        predictions=None,
        train_idx=range(len(dataset)),
        val_idx=None,
        cleanlab_ordered_label_errors=cleanlab_errors,
        cleanlab_self_confidence=cleanlab_self_confidence,
    )

    # collect figures per split
    split_figs = []
    for split in training_log.split_history:
        # get indices present in validation set and cleanlab error detection
        val_indices = split["val_idx"]
        cleanlab_indices = np.intersect1d(val_indices, cleanlab_errors)
        cleanlab_confidence = cleanlab_self_confidence[
            np.isin(cleanlab_errors, cleanlab_indices)
        ]
        # get figure with embeddings
        x_coordinates, y_coordinates = embed_latent_features(split["latent_features"])

        fig = get_embeddings_figure(
            x_coordinates=x_coordinates,
            y_coordinates=y_coordinates,
            dataset=dataset,
            predictions=training_log.preds,
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
    if training_log.config.splitter_class != "sklearn.model_selection.LeaveOneOut":
        latent_features_html += "<br/><br/>".join(
            f"<h5>Split {i + 1}/{len(split_figs)}</h5><br/>"
            + fig.to_html(full_html=False, include_plotlyjs="cdn")
            for i, fig in enumerate(split_figs)
        )

    logger.info(
        f"Rendering the latent features report to {report_path.resolve().as_uri()}"
    )
    template = report.load_template("latent-features")
    template_vars = {"latent_features_html": latent_features_html}
    report_path.parent.mkdir(exist_ok=True, parents=True)
    report.render(template, template_vars, report_path)


def visualize_cleanlab_outliers(
    training_log: TrainingLog,
    cleanlab_errors: np.ndarray,
    cleanlab_self_confidence: np.ndarray,
    img_dir: pathlib.Path,
    report_path: pathlib.Path,
) -> None:
    """Visualize cleanlab outliers."""
    cleanlab_ordered_label_errors = list(cleanlab_errors)

    # no need for feature extraction, we will only handle morphologies anyway
    dataset = MorphologyDataset.from_csv(training_log.config.input_csv)

    all_probabilities = training_log.probas
    if all_probabilities is None:
        raise ValueError("training_log.probas is missing")
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
            f"{m_type}: {probability:.2%}"
            for m_type, probability in proposed_predictions_per_sample
        )
        proposed_predictions.append(proposed_predictions_per_sample_str)

    morph_names_cleanlab = np.array(dataset.morph_names)[cleanlab_ordered_label_errors]
    image_data_sources_cleanlab = []
    cached_figures = {}
    img_dir.mkdir()
    logger.info(f"Visualising outlier morphologies ({len(morph_names_cleanlab)} items)")
    for morph_name in morph_names_cleanlab:
        logger.info(f"Rendering morphology {morph_name}")
        if morph_name not in cached_figures:
            sample = dataset.get_sample_by_morph_name(morph_name)
            if sample is None:
                fig = generate_empty_image()
            else:
                fig = generate_image(sample)
            cached_figures[morph_name] = fig
        fig = cached_figures[morph_name]
        img_path = img_dir / f"{morph_name}.png"
        fig.savefig(img_path)
        image_data_sources_cleanlab.append(img_path)

    expert_labels_cleanlab = np.array(dataset.labels)[cleanlab_ordered_label_errors]
    cleanlab_outliers = []
    for name, true_label, confidence, img, proposed_prediction in zip(
        morph_names_cleanlab,
        expert_labels_cleanlab,
        cleanlab_self_confidence,
        image_data_sources_cleanlab,
        proposed_predictions,
    ):
        outlier_info = f"""
        Morphology <b>{name.rpartition('/')[2]}</b> belongs to <b>{true_label}</b>
        with confidence <b>{confidence:.2%}</b><br/>
        proposed labels and their confidence: {proposed_prediction}
        <img src='{img.relative_to(report_path.parent)}' width='80%'>
        """
        cleanlab_outliers.append(textwrap.dedent(outlier_info).strip())

    cleanlab_outliers_html = f"""
    <ul>
    <li>Self confidence is the holdout probability that an example
    belongs to its given class label.</li>
    <li>List of detected outliers using cleanlab</li>
    </ul>
    {"<br/>".join(cleanlab_outliers)}
    """
    logger.info(
        f"Rendering the cleanlab outliers report to {report_path.resolve().as_uri()}"
    )
    template = report.load_template("detected-outliers")
    template_vars = {"cleanlab_outliers_html": cleanlab_outliers_html}
    report_path.parent.mkdir(exist_ok=True, parents=True)
    report.render(template, template_vars, report_path)

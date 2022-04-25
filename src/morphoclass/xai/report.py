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
"""XAI report creation."""
from __future__ import annotations

import logging
import os
import pathlib
import textwrap

import neurom as nm
import numpy as np
import torch
from captum.attr import Deconvolution  # DeepLiftShap,; GuidedBackprop,; InputXGradient,
from captum.attr import GradientShap
from captum.attr import IntegratedGradients
from captum.attr import Saliency

from morphoclass import transforms
from morphoclass.data.morphology_data import MorphologyData
from morphoclass.data.morphology_dataset import MorphologyDataset
from morphoclass.report.xai import XAIReport
from morphoclass.training import reset_seeds
from morphoclass.training.training_log import TrainingLog
from morphoclass.utils import make_torch_deterministic
from morphoclass.utils import warn_if_nondeterministic
from morphoclass.xai import cnn_model_attributions
from morphoclass.xai import cnn_model_attributions_population
from morphoclass.xai import gnn_model_attributions
from morphoclass.xai import grad_cam_cnn_model
from morphoclass.xai import grad_cam_gnn_model
from morphoclass.xai import grad_cam_perslay_model
from morphoclass.xai import perslay_model_attributions
from morphoclass.xai import sklearn_model_attributions_shap
from morphoclass.xai import sklearn_model_attributions_tree

logger = logging.getLogger(__name__)


def make_report(training_log: TrainingLog, output_dir: str | os.PathLike) -> None:
    """Generate XAI report.

    GradCam and GradShap are available for the morphoclass models only.
    SHAP is available for ML models (sklearn).

    The report will store the explanation plots for the best
    and worst class representatives based on the prediction
    probability.

    training_log
        The training log with the data and model information.
    output_dir
        The report output directory.
    """
    logger.info("Ensuring reproducibility")
    reset_seeds(numpy_seed=1234, torch_seed=5678)
    make_torch_deterministic()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    warn_if_nondeterministic(device)

    logger.info("Restoring the dataset")
    dataset = _restore_dataset(training_log)

    logger.info("Restoring the model and computing probabilities")
    model, probas = _get_model_and_probas(training_log, dataset)

    xai_report = XAIReport("xai-report", output_dir)
    logger.info("Generating XAI reports")
    if model.__module__.startswith("sklearn.tree"):
        logger.info("A tree-model found - computing sklearn attributions for trees")
        _add_tree_report(model, dataset, xai_report)
    else:
        logger.info("A non-tree model found")
        _add_non_tree_report(model, dataset, probas, xai_report)

    if model.__module__.startswith("morphoclass.models.cnnet"):
        logger.info("Generating the neuron population SHAP report for CNN")
        _add_cnn_report(model, dataset, xai_report)

    logger.info("Rendering the XAI report and writing it to disk")
    xai_report.write()


def _restore_dataset(training_log):
    logger.info("Restoring the dataset from pre-computed features")
    data = []
    for path in sorted(training_log.config.features_dir.glob("*.features")):
        data.append(MorphologyData.load(path))
    dataset = MorphologyDataset(data)

    logger.info("Restoring the original neurites")
    _restore_neurites(training_log.config.features_dir, dataset)

    return dataset


def _restore_neurites(
    features_dir: str | os.PathLike, dataset: MorphologyDataset
) -> None:
    """Restore TMD neurites for a features-only dataset."""
    # Get the neurite type that the features correspond to. Assuming that
    # the features dir has the form ".../<neurite-type>/<feature-type>"
    # TODO: this is an ad-hoc solution and assumes a certain form of the
    #  directory path where the features are stored. One should think about
    #  a better way of coupling the extracted features with the original
    #  morphologies and their neurites.
    features_dir = pathlib.Path(features_dir)
    neurite_type = features_dir.parent.name
    # TODO: as with the features directory above, this makes assumptions and
    #  does not generalise. This configuration is taken from the
    #  "extract-features" command, where it's configured via the command
    #  line arguments, but the configuration is not stored anywhere, so here
    #  we take the one that has always been used in our experiments.
    transform = transforms.Compose(
        [
            transforms.ExtractTMDNeurites(neurite_type=neurite_type),
            transforms.OrientApicals(),
            transforms.BranchingOnlyNeurites(),
        ]
    )
    for data in dataset:
        data.morphology = nm.load_morphology(str(data.path))
        transform(data)


def _get_model_and_probas(training_log, dataset):
    model_class = training_log.config.model_class
    if model_class.startswith("sklearn") or model_class.startswith("xgboost"):
        model = training_log.all_history["model"]
        val_idx = np.arange(len(dataset))
        images_val = np.array(
            [sample.image for sample in dataset.index_select(val_idx)]
        )
        images_val = images_val.reshape(-1, 10000)
        probas = model.predict_proba(images_val)
    elif model_class.startswith("morphoclass"):
        model = training_log.config.model_cls(**training_log.config.model_params)
        model.load_state_dict(training_log.all_history["model"])

        optimizer = training_log.config.optimizer_cls(
            model.parameters(), **training_log.config.optimizer_params
        )

        # Forward prop
        from morphoclass.data.morphology_data_loader import MorphologyDataLoader
        from morphoclass.training.trainers import Trainer

        trainer = Trainer(
            net=model,
            dataset=dataset,
            loader_class=MorphologyDataLoader,
            optimizer=optimizer,
        )
        _, logits, _ = trainer.predict(idx=np.arange(len(dataset)))
        probas = logits.exp().cpu().numpy()
    else:
        raise ValueError(f"Unknown model class: {model_class}")

    return model, probas


def _add_tree_report(model, dataset, xai_report):
    fig = sklearn_model_attributions_tree(model, dataset)
    if fig:
        img_path = xai_report.add_figure(fig, "tree")
        html = f"""
            <br/>
            <div class="row">
            <img class="center-block" src='file:{img_path}' width='150%'>
            </div>
            """
        xai_report.add_section("Model Attributions", "model-attrib", html)


def _add_non_tree_report(model, dataset, probas, xai_report):
    unique_ys = sorted(dataset.y_to_label)
    all_ys = np.array([sample.y for sample in dataset])
    model_mod = model.__module__

    captum_classes = [
        Deconvolution,
        IntegratedGradients,
        GradientShap,
        Saliency,
        # GuidedBackprop,
        # DeepLiftShap,
        # InputXGradient,
    ]

    logger.info("Creating HTML headings")
    gradcam_parts: list[str] = []
    shap_parts: list[str] = []
    captum_parts_d: dict[str, list[str]] = {}

    for y in unique_ys:
        label = dataset.y_to_label[y]
        logger.info(f"Processing label {label}")
        (ids,) = np.where(all_ys == y)
        sample_bad = np.where(probas == probas[ids, y].min())[0][-1]
        sample_good = np.where(probas == probas[ids, y].max())[0][-1]
        morphology_name_bad = dataset[sample_bad].path
        morphology_name_good = dataset[sample_good].path

        # GradCam
        logger.info("> Running GradCAM analysis")
        if model_mod == "morphoclass.models.man_net":
            fig_gradcam_bad = grad_cam_gnn_model(model, dataset, sample_bad)
            fig_gradcam_good = grad_cam_gnn_model(model, dataset, sample_good)
        elif model_mod == "morphoclass.models.coriander_net":
            fig_gradcam_bad = grad_cam_perslay_model(model, dataset, sample_bad)
            fig_gradcam_good = grad_cam_perslay_model(model, dataset, sample_good)
        elif model_mod == "morphoclass.models.cnnet":
            fig_gradcam_bad = grad_cam_cnn_model(model, dataset, sample_bad)
            fig_gradcam_good = grad_cam_cnn_model(model, dataset, sample_good)
        elif model_mod.startswith("sklearn") or model_mod.startswith("xgboost"):
            fig_gradcam_bad = fig_gradcam_good = None
        else:
            raise ValueError("There is no GradCAM supported for this model")

        if all([fig_gradcam_bad, fig_gradcam_good]):
            path_good = xai_report.add_figure(fig_gradcam_good, f"{label}_gradcam_good")
            path_bad = xai_report.add_figure(fig_gradcam_bad, f"{label}_gradcam_bad")
            gradcam_parts.append(
                f"""
                <br/>
                <div class="row">
                <div class='col-md-12'>
                <h3>Morphology type {label}</h3><br/>
                <h4>Good Representative</h4>
                <p>Morphology name: <b>{morphology_name_good}</b></p>
                <p>Probability of belonging to this class:
                    <b>{probas[sample_good, y]:.2%}</b>
                </p>
                <img src='file:{path_good}' width='90%'>
                <h4>Bad Representative</h4>
                <p>Morphology name: <b>{morphology_name_bad}</b></p>
                <p>Probability of belonging to this class:
                    <b>{probas[sample_bad, y]:.2%}</b>
                </p>
                <img src='file:{path_bad}' width='90%'>
                </div>
                </div>
                """
            )

        # sklearn
        logger.info("> Running SHAP analysis")
        if model_mod.startswith("sklearn") or model_mod.startswith("xgboost"):
            fig_bad_shap, text_bad = sklearn_model_attributions_shap(
                model, dataset, sample_bad
            )
            fig_good_shap, text_good = sklearn_model_attributions_shap(
                model, dataset, sample_good
            )
        elif "morphoclass" in model_mod:
            fig_good_shap = fig_bad_shap = None
            text_good = text_bad = None
        else:
            raise ValueError("There is no sklearn supported for this model")

        if all([fig_good_shap, fig_bad_shap]):
            path_good = xai_report.add_figure(fig_good_shap, f"{label}_shap_good")
            path_bad = xai_report.add_figure(fig_bad_shap, f"{label}_shap_bad")
            text_good = text_good.replace("\n", "<br/>")
            text_bad = text_bad.replace("\n", "<br/>")
            shap_parts.append(
                f"""
                <br/>
                <div class="row">
                <div class='col-md-12'>
                <h3>Morphology type {label}</h3><br/>
                <h4>Good Representative</h4>
                <p>Morphology name: <b>{morphology_name_good}</b></p>
                <p>Pixels: <b>{text_good}</b></p>
                <p>Probability of belonging to this class:
                    <b>{probas[sample_good, y]}</b>
                </p>
                <img src='file:{path_good}' width='90%'>
                <h4>Bad Representative</h4>
                <p>Morphology name: <b>{morphology_name_bad}</b></p>
                <p>Pixels: <b>{text_bad}</b></p>
                <p>Probability of belonging to this class:
                    <b>{probas[sample_bad, y]}</b>
                </p>
                <img src= 'file:{path_bad}' width='90%'>
                </div>
                </div>
                """
            )

        logger.info("> Running various captum interpretability analyses")
        # captum interpretability models
        for captum_cls in captum_classes:
            method_name = captum_cls.__name__
            logger.info(f">> Running method {method_name!r}")

            if model_mod == "morphoclass.models.man_net":
                fig_good = gnn_model_attributions(
                    model,
                    dataset,
                    sample_id=sample_good,
                    interpretability_method_cls=captum_cls,
                )
                fig_bad = gnn_model_attributions(
                    model,
                    dataset,
                    sample_id=sample_bad,
                    interpretability_method_cls=captum_cls,
                )
            elif model_mod == "morphoclass.models.coriander_net":
                fig_good = perslay_model_attributions(
                    model,
                    dataset,
                    sample_id=sample_good,
                    interpretability_method_cls=captum_cls,
                )
                fig_bad = perslay_model_attributions(
                    model,
                    dataset,
                    sample_id=sample_bad,
                    interpretability_method_cls=captum_cls,
                )
            elif model_mod == "morphoclass.models.cnnet":
                fig_good = cnn_model_attributions(
                    model,
                    dataset,
                    sample_id=sample_good,
                    interpretability_method_cls=captum_cls,
                )
                fig_bad = cnn_model_attributions(
                    model,
                    dataset,
                    sample_id=sample_bad,
                    interpretability_method_cls=captum_cls,
                )
            elif model_mod.startswith("sklearn") or model_mod.startswith("xgboost"):
                fig_bad = fig_good = None
            else:
                raise ValueError("There is no GradCAM supported for this model")

            if all([fig_bad, fig_good]):
                path_good = xai_report.add_figure(
                    fig_good, f"{label}_{method_name}_good"
                )
                path_bad = xai_report.add_figure(fig_bad, f"{label}_{method_name}_bad")

                key = f"Captum - {method_name}"
                if key not in captum_parts_d:
                    captum_parts_d[key] = []
                captum_parts_d[key].append(
                    f"""
                    <br/>
                    <div class="row">
                    <div class='col-md-12'>
                    <h3>Morphology type {label}</h3><br/>
                    <h4>Good Representative</h4>
                    <p>Morphology name: <b>{morphology_name_good}</b></p>
                    <p>Probability of belonging to this class:
                        <b>{probas[sample_good, y]}</b>
                    </p>
                    <img src='file:{path_good}' width='100%'>
                    <h4>Bad Representative</h4>
                    <p>Morphology name: <b>{morphology_name_bad}</b></p>
                    <p>Probability of belonging to this class:
                        <b>{probas[sample_bad, y]}</b>
                    </p>
                    <img src='file:{path_bad}' width='100%'>
                    </div>
                    </div>
                    """
                )
            break
        break

    def join_parts(parts: list[str]) -> str:
        return "<br/>".join(textwrap.dedent(part).strip() for part in parts)

    logger.info("Collecting report parts")
    if gradcam_parts:
        xai_report.add_section("GradCAM", "grad-cam", join_parts(gradcam_parts))
    if shap_parts:
        xai_report.add_section("SHAP", "shap", join_parts(shap_parts))
    for title, captum_parts in captum_parts_d.items():
        if captum_parts:
            _id = title.lower().replace(" ", "-")
            xai_report.add_section(title, _id, join_parts(captum_parts))


def _add_cnn_report(model, dataset, xai_report):
    # Generate figures
    figures, labels = cnn_model_attributions_population(model, dataset)

    # Save figures
    img_lines = []
    for fig, label in zip(figures, labels):
        fig_path = xai_report.add_figure(fig, f"{label}_population_compared_to_others")
        img_lines.append(f"<img src='file:{fig_path}' width='120%'>")

    # Generate the report HTML data
    xai_report.add_section(
        "Compare neuron population with SHAP of CNN",
        "population-cnn",
        "<br/><br/><hr/><br/>".join(img_lines),
    )

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


def make_report(
    report_path: str | os.PathLike,
    training_log: TrainingLog,
    seed: int | None = None,
) -> None:
    """Generate XAI report.

    GradCam and GradShap are available for the morphoclass models only.
    SHAP is available for ML models (sklearn).

    The report will store the explanation plots for the best
    and worst class representatives based on the prediction
    probability.

    results_file
        The path of the output results report.
    training_log
        The training log with the data and model information.
    seed
        A shared NumPy and PyTorch seed.
    """
    report_path = pathlib.Path(report_path).with_suffix(".html")
    results_dir = report_path.parent
    results_dir.mkdir(exist_ok=True, parents=True)
    img_dir = report_path.with_suffix("")
    img_dir.mkdir(exist_ok=True)

    make_torch_deterministic()

    logger.info("Restoring the dataset from pre-computed features")
    data = []
    for path in sorted(training_log.config.features_dir.glob("*.features")):
        data.append(MorphologyData.load(path))
    dataset = MorphologyDataset(data)

    logger.info("Restoring the original neurites")
    _restore_neurites(training_log.config.features_dir, dataset)

    if seed is not None:
        reset_seeds(numpy_seed=seed, torch_seed=seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    warn_if_nondeterministic(device)

    logger.info("Predicting probabilities")
    model_class = training_log.config.model_class
    if model_class.startswith("sklearn") or model_class.startswith("xgboost"):
        model = training_log.all_history["model"]
        val_idx = np.arange(len(dataset))
        images_val = np.array(
            [sample.image for sample in dataset.index_select(val_idx)]
        )
        images_val = images_val.reshape(-1, 10000)
        probabilities = model.predict_proba(images_val)
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
        _, probabilities_t, _ = trainer.predict(idx=np.arange(len(dataset)))
        probabilities = probabilities_t.cpu().numpy()
    else:
        raise ValueError(f"Unknown model class: {model_class}")

    report_parts: list[str] = []
    section_refs: list[tuple[str, str]] = []  # title, HTML tag ID
    logger.info("Generating XAI reports")
    if model.__module__.startswith("sklearn.tree"):
        logger.info("A tree-model found - computing sklearn attributions for trees")
        fig = sklearn_model_attributions_tree(model, dataset)
        if fig:
            img_path = img_dir / "tree.png"
            fig.savefig(img_path)
            img_rel_path = img_path.relative_to(results_dir)
        else:
            img_rel_path = ""

        report_parts.append(
            f"""
            <br/>
            <div class="row">
            <img class="center-block" src='file:{img_rel_path}' width='150%'>
            </div>
            """
        )
    else:
        logger.info("A non-tree model found")
        non_tree_report_parts, non_tree_section_refs = _xai_non_tree(
            dataset,
            model,
            probabilities,
            results_dir,
            img_dir,
        )
        section_refs.extend(non_tree_section_refs)
        report_parts.extend(non_tree_report_parts)

    # CNN population
    if model.__module__.startswith("morphoclass.models.cnnet"):
        logger.info("Generating the neuron population SHAP report for CNN")
        cnn_report, cnn_section_ref = _make_cnn_report(
            model,
            dataset,
            img_dir,
            results_dir,
        )
        section_refs.append(cnn_section_ref)
        report_parts.append(cnn_report)

    toc_html = "<br/>\n".join(
        f"<a href='#{_id}'>{title}</a>" for title, _id in section_refs
    )
    report_html = "\n".join(textwrap.dedent(part).strip() for part in report_parts)

    logger.info("Rendering the XAI report and writing it to disk")
    _write_xai_report(report_path, toc_html, report_html)
    logger.info(f"XAI report written to {report_path.resolve().as_uri()}")


def _xai_non_tree(dataset, model, probas, results_dir, img_dir):
    unique_ys = sorted(dataset.y_to_label)
    all_ys = np.array([sample.y for sample in dataset])
    model_mod = model.__module__

    section_refs: list[tuple[str, str]] = []  # title, href
    report_parts: list[str] = []
    xai_html: dict[str, list] = {}

    interpretability_method_classes = [
        Deconvolution,
        IntegratedGradients,
        GradientShap,
        Saliency,
        # GuidedBackprop,
        # DeepLiftShap,
        # InputXGradient,
    ]

    logger.info("Creating HTML headings")
    gradcam_xai_html = [
        """
        <div class="page-header">
            <h1 id="grad-cam">GradCam XAI Report</h1>
        </div>
        """
    ]
    section_refs.append(("GradCam", "grad-cam"))

    shap_xai_html = [
        """
        <div class="page-header">
            <h1 id="shap">Shap XAI Report</h1>
        </div>
        """
    ]
    section_refs.append(("SHAP", "shap"))

    for interpretability_method_cls in interpretability_method_classes:
        method_name = interpretability_method_cls.__name__
        xai_html[method_name] = [
            f"""
            <div class="page-header">
                <h1 id="{method_name}">{method_name} XAI Report</h1>
            </div>
            """
        ]
        section_refs.append((method_name, method_name))

    for y in unique_ys:
        label = dataset.y_to_label[y]
        logger.info(f"Processing sample at index {y}")
        ids = np.where(all_ys == y)[0]
        sample_bad = np.where(probas == probas[ids, y].min())[0][-1]
        sample_good = np.where(probas == probas[ids, y].max())[0][-1]
        morphology_name_bad = dataset[sample_bad].path
        morphology_name_good = dataset[sample_good].path

        # GradCam
        logger.info("> Generating GradCAM figures")
        if model_mod == "morphoclass.models.man_net":
            fig_gradcam_bad = grad_cam_gnn_model(model, dataset, sample_id=sample_bad)
            fig_gradcam_good = grad_cam_gnn_model(model, dataset, sample_id=sample_good)
        elif model_mod == "morphoclass.models.coriander_net":
            fig_gradcam_bad = grad_cam_perslay_model(
                model, dataset, sample_id=sample_bad
            )
            fig_gradcam_good = grad_cam_perslay_model(
                model, dataset, sample_id=sample_good
            )
        elif model_mod == "morphoclass.models.cnnet":
            fig_gradcam_bad = grad_cam_cnn_model(model, dataset, sample_id=sample_bad)
            fig_gradcam_good = grad_cam_cnn_model(model, dataset, sample_id=sample_good)
        elif model_mod.startswith("sklearn") or model_mod.startswith("xgboost"):
            fig_gradcam_bad = None
            fig_gradcam_good = None
        else:
            raise ValueError("There is no GradCam supported for this model")

        logger.info("> Saving figures")
        fig_gradcam_bad_path = img_dir / f"{label}_gradcam_bad.png"
        fig_gradcam_good_path = img_dir / f"{label}_gradcam_good.png"
        if all([fig_gradcam_bad, fig_gradcam_good]):
            fig_gradcam_bad.savefig(fig_gradcam_bad_path)
            fig_gradcam_good.savefig(fig_gradcam_good_path)

        if all([fig_gradcam_bad, fig_gradcam_good]):
            gradcam_xai_html.append(
                f"""
                <br/>
                <div class="row">
                <div class='col-md-12'>
                <h3>Morphology type {label}</h3><br/>
                <h4>Good Representative</h4>
                <p>Morphology name: <b>{morphology_name_good}</b></p>
                <p>Probability of belonging to this class:
                    <b>{probas[sample_good, y]}</b></p>
                <img src=
                    'file:{fig_gradcam_good_path.relative_to(
                            results_dir) if fig_gradcam_good else ''}'
                    width='90%'>
                <h4>Bad Representative</h4>
                <p>Morphology name: <b>{morphology_name_bad}</b></p>
                <p>Probability of belonging to this class:
                    <b>{probas[sample_bad, y]}</b></p>
                <img src=
                    'file:{fig_gradcam_bad_path.relative_to(
                            results_dir) if fig_gradcam_bad else ''}'
                    width='90%'>
                </div>
                </div>
                """
            )

        # sklearn
        logger.info("> Computing SHAP")
        fig_bad_filename_shap = img_dir / f"{label}_shap_bad.png"
        fig_good_filename_shap = img_dir / f"{label}_shap_good.png"

        if model_mod.startswith("sklearn") or model_mod.startswith("xgboost"):
            fig_bad_shap, text_bad = sklearn_model_attributions_shap(
                model, dataset, sample_id=sample_bad
            )
            fig_good_shap, text_good = sklearn_model_attributions_shap(
                model, dataset, sample_id=sample_good
            )

            fig_bad_shap.savefig(fig_bad_filename_shap)
            fig_good_shap.savefig(fig_good_filename_shap)

            shap_xai_html.append(
                f"""
                <br/>
                <div class="row">
                <div class='col-md-12'>
                <h3>Morphology type {label}</h3><br/>
                <h4>Good Representative</h4>
                <p>Morphology name: <b>{morphology_name_good}</b></p>
                <p>Probability of belonging to this class:
                    <b>{probas[sample_good, y]}</b></p>
                <img src=
                    'file:{fig_good_filename_shap.relative_to(
                            results_dir) if fig_good_shap else ''}'
                    width='90%'>
                <h4>Bad Representative</h4>
                <p>Morphology name: <b>{morphology_name_bad}</b></p>
                <p>Probability of belonging to this class:
                    <b>{probas[sample_bad, y]}</b></p>
                <img src=
                    'file:{fig_bad_filename_shap.relative_to(
                            results_dir) if fig_bad_shap else ''}'
                    width='90%'>
                </div>
                </div>
                """
            )
        elif "morphoclass" in model_mod:
            pass
        else:
            raise ValueError("There is no sklearn supported for this model")

        logger.info("> Captum interpretability methods")
        # captum interpretability models
        for interpretability_method_cls in interpretability_method_classes:
            method_name = interpretability_method_cls.__name__
            logger.info(f">> Running method {method_name}")
            fig_bad_filename = img_dir / f"{label}_{method_name}_bad.png"
            fig_good_filename = img_dir / f"{label}_{method_name}_good.png"
            text_bad, text_good = "", ""
            if model_mod == "morphoclass.models.man_net":
                fig_bad = gnn_model_attributions(
                    model,
                    dataset,
                    sample_id=sample_bad,
                    interpretability_method_cls=interpretability_method_cls,
                )
                fig_good = gnn_model_attributions(
                    model,
                    dataset,
                    sample_id=sample_good,
                    interpretability_method_cls=interpretability_method_cls,
                )
            elif model_mod == "morphoclass.models.coriander_net":
                fig_bad = perslay_model_attributions(
                    model,
                    dataset,
                    sample_id=sample_bad,
                    interpretability_method_cls=interpretability_method_cls,
                )
                fig_good = perslay_model_attributions(
                    model,
                    dataset,
                    sample_id=sample_good,
                    interpretability_method_cls=interpretability_method_cls,
                )
            elif model_mod == "morphoclass.models.cnnet":
                fig_bad = cnn_model_attributions(
                    model,
                    dataset,
                    sample_id=sample_bad,
                    interpretability_method_cls=interpretability_method_cls,
                )
                fig_good = cnn_model_attributions(
                    model,
                    dataset,
                    sample_id=sample_good,
                    interpretability_method_cls=interpretability_method_cls,
                )
            elif model_mod.startswith("sklearn") or model_mod.startswith("xgboost"):
                fig_bad = None
                fig_good = None
            else:
                raise ValueError("There is no GradCAM supported for this model")

            if all([fig_bad, fig_good]):
                fig_bad.savefig(fig_bad_filename)
                fig_good.savefig(fig_good_filename)

            new_line = "\n"
            if all([fig_bad, fig_good]):
                xai_html[method_name].append(
                    f"""
                    <br/>
                    <div class="row">
                    <div class='col-md-12'>
                    <h3>Morphology type {label}</h3><br/>
                    <h4>Good Representative</h4>
                    <p>Morphology name: <b>{morphology_name_good}</b></p>
                    <p>Pixels: <b>{text_good.replace(new_line, '<br/>')}</b></p>
                    <p>Probability of belonging to this class:
                        <b>{probas[sample_good, y]}</b></p>
                    <img
                        src='file:{fig_good_filename.relative_to(results_dir)}'
                        width='100%'
                    >
                    <h4>Bad Representative</h4>
                    <p>Morphology name: <b>{morphology_name_bad}</b></p>
                    <p>Pixels: <b>{text_bad.replace(new_line, '<br/>')}</b></p>
                    <p>Probability of belonging to this class:
                        <b>{probas[sample_bad, y]}</b></p>
                    <img
                        src='file:{fig_bad_filename.relative_to(results_dir)}'
                        width='100%'
                    >
                    </div>
                    </div>
                    """
                )

    logger.info("Collecting reports")
    # Collect GradCAM
    if len(gradcam_xai_html) == 1:
        gradcam_xai_html_str = ""
    else:
        gradcam_xai_html_str = "<br/><br/><hr/><br/>".join(
            textwrap.dedent(part).strip() for part in gradcam_xai_html
        )
    report_parts.append(gradcam_xai_html_str)
    # collect SHAP
    if len(shap_xai_html) == 1:
        shap_xai_html_str = ""
    else:
        shap_xai_html_str = "<br/><br/><hr/><br/>".join(
            textwrap.dedent(part).strip() for part in shap_xai_html
        )
    report_parts.append(shap_xai_html_str)
    # collect captum
    for v in xai_html.values():
        if len(v) == 1:
            xai_html_str = ""
        else:
            xai_html_str = "<br/><br/><hr/><br/>".join(textwrap.dedent(v).strip())
        report_parts.append(xai_html_str)

    return report_parts, section_refs


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


def _make_cnn_report(model, dataset, img_dir, target_dir):
    # Generate figures
    figures, labels = cnn_model_attributions_population(model, dataset)

    # Save figures
    fig_paths = []
    for fig, label in zip(figures, labels):
        fig_path = img_dir / f"{label}_population_compared_to_others.png"
        fig.savefig(fig_path)
        fig_paths.append(fig_path)

    # Generate the report HTML data
    cnn_title = "Compare population of neurons with SHAP"
    cnn_link_title = "Neuron Population - SHAP of CNN model"
    cnn_anchor = "population-cnn"
    report_parts = [f"<h3 id='{cnn_anchor}'>{cnn_title}</h3>"]
    for fig_path in fig_paths:
        img_rel_path = fig_path.relative_to(target_dir)
        report_parts.append(f"<img src='file:{img_rel_path}' width='120%'>")
    report = "<br/><br/><hr/><br/>".join(report_parts)

    return report, (cnn_link_title, cnn_anchor)


def _write_xai_report(output_path, content_html, xai_report_final):
    """Render and write the XAI report to disk.

    Parameters
    ----------
    output_path : str or pathlib.Path
        Write the XAI report to this file.
    content_html : str
        The ``content_html`` part of the XAI report template.
    xai_report_final : str
        The ``xai_report_final`` part of the XAI report template.
    """
    import morphoclass as mc
    import morphoclass.report

    template_vars = {
        "content_html": content_html,
        "xai_report_final": xai_report_final,
    }

    template = mc.report.load_template("xai-report")
    mc.report.render(template, template_vars, output_path)
    logger.info(f"Report stored in: {output_path.resolve().as_uri()}")

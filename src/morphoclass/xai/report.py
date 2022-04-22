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
import pathlib
import textwrap

import numpy as np
import torch
from captum.attr import Deconvolution  # DeepLiftShap,; GuidedBackprop,; InputXGradient,
from captum.attr import GradientShap
from captum.attr import IntegratedGradients
from captum.attr import Saliency

from morphoclass.data.morphology_data import MorphologyData
from morphoclass.data.morphology_dataset import MorphologyDataset
from morphoclass.training import reset_seeds
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
    results_file,
    training_log,
    seed=None,
):
    """Generate XAI report.

    GradCam and GradShap are available for the morphoclass models only.
    SHAP is available for ML models (sklearn).

    The report will store the explanation plots for the best
    and worst class representatives based on the prediction
    probability.

    results_file : str or pathlib.Path
        Path to the report file.
    dataset
        A morphology dataset.
    model_class : str
        Name of the model class.
    model_params : dict
        Model parameters.
    model_old : dict
        Model's state dictionary.
    seed : int, optional
        Seed.
    """
    results_file = pathlib.Path(results_file)
    results_directory = results_file.parent
    results_directory.mkdir(exist_ok=True, parents=True)
    images_directory = results_directory / "images"
    images_directory.mkdir(exist_ok=True, parents=True)

    make_torch_deterministic()

    logger.info("Restoring the dataset from pre-computed features")
    data = []
    for path in sorted(training_log.config.features_dir.glob("*.features")):
        data.append(MorphologyData.load(path))
    dataset = MorphologyDataset(data)

    labels_ids = sorted(dataset.y_to_label.keys())
    labels_str = [dataset.y_to_label[s] for s in labels_ids]

    if seed is not None:
        reset_seeds(numpy_seed=seed, torch_seed=seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    warn_if_nondeterministic(device)

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
            **training_log.config.optimizer_params
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

    labels_all = np.array([sample.y_str for sample in dataset])

    content_html = ""
    xai_report_final = ""
    if model.__module__.startswith("sklearn.tree"):
        fig = sklearn_model_attributions_tree(model, dataset)
        if fig:
            img_path = images_directory / "tree.png"
            _save_figure(fig, img_path)
            img_rel_path = img_path.relative_to(results_directory)
        else:
            img_rel_path = ""

        tree_report_html = textwrap.dedent(
            f"""
            <br/>
            <div class="row">
            <img class="center-block" src='file:{img_rel_path}' width='150%'>
            </div>
            """
        ).strip()
        # collect tree
        xai_report_final += tree_report_html
    else:
        gradcam_xai_html = [
            textwrap.dedent(
                """
                <div class="page-header">
                    <h1 id="grad-cam">GradCam XAI Report</h1>
                </div>
                """
            ).strip()
        ]
        content_html += "<br/><a href='#grad-cam'>GradCam</a>"

        shap_xai_html = [
            textwrap.dedent(
                """
                <div class="page-header">
                    <h1 id="shap">Shap XAI Report</h1>
                </div>
                """
            ).strip()
        ]
        content_html += "<br/><a href='#shap'>SHAP</a>"

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

        for interpretability_method_cls in interpretability_method_classes:
            xai_html[interpretability_method_cls.__name__] = [
                textwrap.dedent(
                    f"""
                        <div class="page-header">
                            <h1 id="{interpretability_method_cls.__name__}">
                            {interpretability_method_cls.__name__} XAI Report</h1>
                        </div>"""
                ).strip()
            ]
            content_html += (
                f"<br/><a href='#{interpretability_method_cls.__name__}'>"
                f"{interpretability_method_cls.__name__}</a>"
            )

        for label_id, label_str in zip(labels_ids, labels_str):
            indices = np.where(labels_all == label_str)[0]
            sample_bad = np.where(
                probabilities == probabilities[indices, label_id].min()
            )[0][-1]
            sample_good = np.where(
                probabilities == probabilities[indices, label_id].max()
            )[0][-1]
            morphology_name_bad = dataset.morph_names[sample_bad]
            morphology_name_good = dataset.morph_names[sample_good]

            # GradCam
            fig_bad_filename_gradcam = images_directory / f"{label_str}_gradcam_bad.png"
            fig_good_filename_gradcam = (
                images_directory / f"{label_str}_gradcam_good.png"
            )

            if model.__module__ == "morphoclass.models.man_net":
                fig_bad_gradcam = grad_cam_gnn_model(
                    model, dataset, sample_id=sample_bad
                )
                fig_good_gradcam = grad_cam_gnn_model(
                    model, dataset, sample_id=sample_good
                )
            elif model.__module__ == "morphoclass.models.coriander_net":
                fig_bad_gradcam = grad_cam_perslay_model(
                    model, dataset, sample_id=sample_bad
                )
                fig_good_gradcam = grad_cam_perslay_model(
                    model, dataset, sample_id=sample_good
                )
            elif model.__module__ == "morphoclass.models.cnnet":
                fig_bad_gradcam = grad_cam_cnn_model(
                    model, dataset, sample_id=sample_bad
                )
                fig_good_gradcam = grad_cam_cnn_model(
                    model, dataset, sample_id=sample_good
                )
            elif model.__module__.startswith("sklearn") or model.__module__.startswith(
                "xgboost"
            ):
                fig_bad_gradcam = None
                fig_good_gradcam = None
            else:
                raise ValueError("There is no GradCam supported for this model")

            if None not in [fig_bad_gradcam, fig_good_gradcam]:
                _save_figure(fig_bad_gradcam, fig_bad_filename_gradcam)
                _save_figure(fig_good_gradcam, fig_good_filename_gradcam)

            if None not in [fig_bad_gradcam, fig_good_gradcam]:
                gradcam_xai_html.append(
                    textwrap.dedent(
                        f"""
                <br/>
                <div class="row">
                <div class='col-md-12'>
                <h3>Morphology type {label_str}</h3><br/>
                <h4>Good Representative</h4>
                <p>Morphology name: <b>{morphology_name_good}</b></p>
                <p>Probability of belonging to this class:
                    <b>{probabilities[sample_good, label_id]}</b></p>
                <img src=
                    'file:{fig_good_filename_gradcam.relative_to(
                    results_directory) if fig_good_gradcam else ''}'
                    width='90%'>
                <h4>Bad Representative</h4>
                <p>Morphology name: <b>{morphology_name_bad}</b></p>
                <p>Probability of belonging to this class:
                    <b>{probabilities[sample_bad, label_id]}</b></p>
                <img src=
                    'file:{fig_bad_filename_gradcam.relative_to(
                    results_directory) if fig_bad_gradcam else ''}'
                    width='90%'>
                </div>
                </div>"""
                    ).strip()
                )

            # sklearn
            fig_bad_filename_shap = images_directory / f"{label_str}_shap_bad.png"
            fig_good_filename_shap = images_directory / f"{label_str}_shap_good.png"

            if model.__module__.startswith("sklearn") or model.__module__.startswith(
                "xgboost"
            ):
                fig_bad_shap, text_bad = sklearn_model_attributions_shap(
                    model, dataset, sample_id=sample_bad
                )
                fig_good_shap, text_good = sklearn_model_attributions_shap(
                    model, dataset, sample_id=sample_good
                )

                _save_figure(fig_bad_shap, fig_bad_filename_shap)
                _save_figure(fig_good_shap, fig_good_filename_shap)

                shap_xai_html.append(
                    textwrap.dedent(
                        f"""
                <br/>
                <div class="row">
                <div class='col-md-12'>
                <h3>Morphology type {label_str}</h3><br/>
                <h4>Good Representative</h4>
                <p>Morphology name: <b>{morphology_name_good}</b></p>
                <p>Probability of belonging to this class:
                    <b>{probabilities[sample_good, label_id]}</b></p>
                <img src=
                    'file:{fig_good_filename_shap.relative_to(
                    results_directory) if fig_good_shap else ''}'
                    width='90%'>
                <h4>Bad Representative</h4>
                <p>Morphology name: <b>{morphology_name_bad}</b></p>
                <p>Probability of belonging to this class:
                    <b>{probabilities[sample_bad, label_id]}</b></p>
                <img src=
                    'file:{fig_bad_filename_shap.relative_to(
                    results_directory) if fig_bad_shap else ''}'
                    width='90%'>
                </div>
                </div>"""
                    ).strip()
                )
            elif "morphoclass" in model.__module__:
                pass
            else:
                raise ValueError("There is no sklearn supported for this model")

            # captum interpretability models
            for interpretability_method_cls in interpretability_method_classes:
                fig_bad_filename = (
                    images_directory
                    / f"{label_str}_{interpretability_method_cls.__name__}_bad.png"
                )
                fig_good_filename = (
                    images_directory
                    / f"{label_str}_{interpretability_method_cls.__name__}_good.png"
                )
                text_bad, text_good = "", ""
                if model.__module__ == "morphoclass.models.man_net":

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
                elif model.__module__ == "morphoclass.models.coriander_net":
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
                elif model.__module__ == "morphoclass.models.cnnet":
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
                elif model.__module__.startswith(
                    "sklearn"
                ) or model.__module__.startswith("xgboost"):
                    fig_bad = None
                    fig_good = None
                else:
                    raise ValueError("There is no GradCam supported for this model")

                if None not in [fig_bad, fig_good]:
                    _save_figure(fig_bad, fig_bad_filename)
                    _save_figure(fig_good, fig_good_filename)

                new_line = "\n"
                if None not in [fig_bad, fig_good]:
                    xai_html[interpretability_method_cls.__name__].append(
                        textwrap.dedent(
                            f"""
                    <br/>
                    <div class="row">
                    <div class='col-md-12'>
                    <h3>Morphology type {label_str}</h3><br/>
                    <h4>Good Representative</h4>
                    <p>Morphology name: <b>{morphology_name_good}</b></p>
                    <p>Pixels: <b>{text_good.replace(new_line, '<br/>')}</b></p>
                    <p>Probability of belonging to this class:
                        <b>{probabilities[sample_good, label_id]}</b></p>
                    <img
                        src='file:{fig_good_filename.relative_to(results_directory)}'
                        width='100%'
                    >
                    <h4>Bad Representative</h4>
                    <p>Morphology name: <b>{morphology_name_bad}</b></p>
                    <p>Pixels: <b>{text_bad.replace(new_line, '<br/>')}</b></p>
                    <p>Probability of belonging to this class:
                        <b>{probabilities[sample_bad, label_id]}</b></p>
                    <img
                        src='file:{fig_bad_filename.relative_to(results_directory)}'
                        width='100%'
                    >
                    </div>
                    </div>
                    """
                        ).strip()
                    )

        # Collect GradCAM
        if len(gradcam_xai_html) == 1:
            gradcam_xai_html_str = ""
        else:
            gradcam_xai_html_str = "<br/><br/><hr/><br/>".join(gradcam_xai_html)
        xai_report_final += gradcam_xai_html_str
        # collect shap
        if len(shap_xai_html) == 1:
            shap_xai_html_str = ""
        else:
            shap_xai_html_str = "<br/><br/><hr/><br/>".join(shap_xai_html)
        xai_report_final += shap_xai_html_str
        # collect captum
        for k, v in xai_html.items():
            if len(xai_html[k]) == 1:
                xai_html_str = ""
            else:
                xai_html_str = "<br/><br/><hr/><br/>".join(v)
            xai_report_final += xai_html_str

    # CNN population
    if model.__module__.startswith("morphoclass.models.cnnet"):
        logger.info("Generating the neuron population SHAP report for CNN")
        cnn_report_link, cnn_report = _make_cnn_report(
            model,
            dataset,
            images_directory,
            results_directory,
        )
        content_html += f"<br/>{cnn_report_link}"
        xai_report_final += cnn_report

    logger.info("Rendering the XAI report and writing it to disk")
    _write_xai_report(results_file, content_html, xai_report_final)
    logger.info(f"XAI report written to {results_file.resolve().as_uri()}")


def _make_cnn_report(model, dataset, img_dir, target_dir):
    # Generate figures
    figures, labels = cnn_model_attributions_population(model, dataset)

    # Save figures
    fig_paths = []
    for fig, label in zip(figures, labels):
        fig_path = img_dir / f"{label}_population_compared_to_others.png"
        _save_figure(fig, fig_path)
        fig_paths.append(fig_path)

    # Generate the report HTML data
    cnn_title = "Compare population of neurons with SHAP"
    cnn_link_title = "Neuron Population - SHAP of CNN model"
    cnn_anchor = "population-cnn"
    report_link = f"<a href='#{cnn_anchor}'>{cnn_link_title}</a>"
    report_parts = [f"<h3 id='{cnn_anchor}'>{cnn_title}</h3>"]
    for fig_path in fig_paths:
        img_rel_path = fig_path.relative_to(target_dir)
        report_parts.append(f"<img src='file:{img_rel_path}' width='120%'>")
    report = "<br/><br/><hr/><br/>".join(report_parts)

    return report_link, report


def _save_figure(fig, file_path):
    for ext in {file_path.suffix, ".png", ".eps", ".pdf"}:
        fig.savefig(file_path.with_suffix(ext))


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

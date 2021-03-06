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

import captum.attr
import neurom as nm
import numpy as np
import torch
from matplotlib.figure import Figure

from morphoclass import training
from morphoclass import transforms
from morphoclass import utils
from morphoclass import xai
from morphoclass.data.morphology_data import MorphologyData
from morphoclass.data.morphology_dataset import MorphologyDataset
from morphoclass.report import plumbing
from morphoclass.training.training_log import TrainingLog
from morphoclass.types import StrPath

logger = logging.getLogger(__name__)


class XAIReport:
    """XAI report manager."""

    def __init__(self, output_dir: StrPath) -> None:
        self.output_dir = pathlib.Path(output_dir).resolve()
        self.img_dir = self.output_dir / "images"
        self.sections: dict[str, str] = {}
        self.titles: dict[str, str] = {}
        self.figures: list[tuple[Figure, pathlib.Path]] = []

    def add_section(self, title: str, _id: str, section_html: str) -> None:
        """Add a new XAI section to the report."""
        _id = _id.replace(" ", "-")
        self.titles[_id] = title.strip()
        self.sections[_id] = textwrap.dedent(section_html).strip()

    def add_figure(self, fig: Figure, name: str) -> pathlib.Path:
        """Add a new figure to the report.

        The figure will be kept in memory until the ``write()`` call. It
        is only when ``write()`` is called that the figures will be written
        to disk together with the actual report.

        This method returns a path that can be used to reference the
        figure image file on disk.

        Parameters
        ----------
        fig
            A matplotlib figure.
        name
            The name of the figure. This will be used as the file name of
            the image file saved to disk. The name should not contain
            a file extension - it will be added automatically.

        Returns
        -------
        pathlib.Path
            The path under which the image will be stored on disk. This path
            is relative to the base directory for reproducibility reasons
            and can be used to refer to the figure image file in HTML blocks.
        """
        fig_path = (self.img_dir / name).with_suffix(".png")
        self.figures.append((fig, fig_path))

        return fig_path.relative_to(self.output_dir)

    @property
    def _template_vars(self):
        toc_lines = []
        section_blocks = []
        for _id, toc_title in self.titles.items():
            toc_lines.append(f"<a href='#{_id}'>{toc_title}</a>")

            # Prepend a section header to each section block
            section_header = f"""
            <div class="page-header">
                <h1 id="{_id}">{self.titles[_id]}</h1>
            </div>
            """
            section_header = textwrap.dedent(section_header).strip()
            section_blocks.append(f"{section_header}<br/>{self.sections[_id]}")
        toc_html = "<br/>".join(toc_lines)
        report_html = "<br/><br/><hr/><br/>".join(section_blocks)

        return {"toc_html": toc_html, "report_html": report_html}

    def write(self, file_stem: str) -> None:
        """Render and write the XAI report to disk."""
        if file_stem.lower().endswith(".html"):
            raise ValueError(
                "The report file stem must not contain the file extension and "
                f'therefore cannot end on ".html". Got: {file_stem!r}'
            )

        template = plumbing.load_template("xai-report")
        self.output_dir.mkdir(exist_ok=True, parents=True)

        logger.info(f"Writing {len(self.figures)} figures")
        self.img_dir.mkdir(exist_ok=True)
        for fig, path in self.figures:
            fig.savefig(path)

        report_path = (self.output_dir / file_stem).with_suffix(".html")
        logger.info(f"Writing report to {report_path.as_uri()}")
        plumbing.render(template, self._template_vars, report_path)


def populate_report(training_log: TrainingLog, xai_report: XAIReport) -> XAIReport:
    """Generate XAI report.

    GradCam and GradShap are available for the morphoclass models only.
    SHAP is available for ML models (sklearn).

    The report will store the explanation plots for the best
    and worst class representatives based on the prediction
    probability.

    Parameters
    ----------
    training_log
        The training log with the data and model information.
    xai_report
        The XAI report to populate.

    Returns
    -------
    XAIReport
        The generated XAI report.
    """
    logger.info("Ensuring reproducibility")
    training.reset_seeds(numpy_seed=1234, torch_seed=5678)
    utils.make_torch_deterministic()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    utils.warn_if_nondeterministic(device)

    logger.info("Restoring the dataset")
    dataset = _restore_dataset(training_log)

    logger.info("Restoring the model and computing probabilities")
    model, probas = _get_model_and_probas(training_log, dataset)

    logger.info("Generating XAI reports")
    if model.__module__.startswith("sklearn.tree"):
        logger.info(
            "Generating reports for a tree model - computing sklearn attributions"
        )
        _add_tree_report(model, dataset, xai_report)
    else:
        logger.info("Generating reports for a non-tree model")
        _add_non_tree_report(model, dataset, probas, xai_report)

    if model.__module__.startswith("morphoclass.models.cnnet"):
        logger.info("Generating the neuron population SHAP report for CNN")
        _add_cnn_report(model, dataset, xai_report)

    return xai_report


def _restore_dataset(training_log):
    logger.info("Restoring the dataset from pre-computed features")
    data = []
    for path in sorted(training_log.config.features_dir.glob("*.features")):
        data.append(MorphologyData.load(path))
    dataset = MorphologyDataset(data)

    logger.info("Restoring the original neurites")
    _restore_neurites(training_log.config.features_dir, dataset)

    return dataset


def _restore_neurites(features_dir: StrPath, dataset: MorphologyDataset) -> None:
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
            [sample.image.cpu().numpy() for sample in dataset.index_select(val_idx)]
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
    fig = xai.sklearn_model_attributions_tree(model, dataset)
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
        captum.attr.Deconvolution,
        captum.attr.IntegratedGradients,
        captum.attr.GradientShap,
        captum.attr.Saliency,
        # captum.attr.GuidedBackprop,
        # captum.attr.DeepLiftShap,
        # captum.attr.InputXGradient,
    ]

    gradcam_parts: list[str] = []
    shap_parts: list[str] = []
    captum_parts_d: dict[str, list[str]] = {}
    for y in unique_ys:
        label = dataset.y_to_label[y]
        logger.info(f"Processing label {label}")
        (ids,) = np.where(all_ys == y)
        good_idx = ids[probas[ids, y].argmax()]
        bad_idx = ids[probas[ids, y].argmin()]
        morphology_name_bad = dataset[bad_idx].path
        morphology_name_good = dataset[good_idx].path

        # GradCam
        logger.info("> Running GradCAM analysis")
        if model_mod == "morphoclass.models.man_net":
            fig_gradcam_bad = xai.grad_cam_gnn_model(model, dataset, bad_idx)
            fig_gradcam_good = xai.grad_cam_gnn_model(model, dataset, good_idx)
        elif model_mod == "morphoclass.models.coriander_net":
            fig_gradcam_bad = xai.grad_cam_perslay_model(model, dataset, bad_idx)
            fig_gradcam_good = xai.grad_cam_perslay_model(model, dataset, good_idx)
        elif model_mod == "morphoclass.models.cnnet":
            fig_gradcam_bad = xai.grad_cam_cnn_model(model, dataset, bad_idx)
            fig_gradcam_good = xai.grad_cam_cnn_model(model, dataset, good_idx)
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
                    <b>{probas[good_idx, y]:.2%}</b>
                </p>
                <img src='file:{path_good}' width='90%'>
                <h4>Bad Representative</h4>
                <p>Morphology name: <b>{morphology_name_bad}</b></p>
                <p>Probability of belonging to this class:
                    <b>{probas[bad_idx, y]:.2%}</b>
                </p>
                <img src='file:{path_bad}' width='90%'>
                </div>
                </div>
                """
            )

        # sklearn
        logger.info("> Running SHAP analysis")
        if model_mod.startswith("sklearn") or model_mod.startswith("xgboost"):
            fig_bad_shap, text_bad = xai.sklearn_model_attributions_shap(
                model, dataset, bad_idx
            )
            fig_good_shap, text_good = xai.sklearn_model_attributions_shap(
                model, dataset, good_idx
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
                    <b>{probas[good_idx, y]:.2%}</b>
                </p>
                <img src='file:{path_good}' width='90%'>
                <h4>Bad Representative</h4>
                <p>Morphology name: <b>{morphology_name_bad}</b></p>
                <p>Pixels: <b>{text_bad}</b></p>
                <p>Probability of belonging to this class:
                    <b>{probas[bad_idx, y]:.2%}</b>
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
                fig_good = xai.gnn_model_attributions(
                    model,
                    dataset,
                    sample_id=good_idx,
                    interpretability_method_cls=captum_cls,
                )
                fig_bad = xai.gnn_model_attributions(
                    model,
                    dataset,
                    sample_id=bad_idx,
                    interpretability_method_cls=captum_cls,
                )
            elif model_mod == "morphoclass.models.coriander_net":
                fig_good = xai.perslay_model_attributions(
                    model,
                    dataset,
                    sample_id=good_idx,
                    interpretability_method_cls=captum_cls,
                )
                fig_bad = xai.perslay_model_attributions(
                    model,
                    dataset,
                    sample_id=bad_idx,
                    interpretability_method_cls=captum_cls,
                )
            elif model_mod == "morphoclass.models.cnnet":
                fig_good = xai.cnn_model_attributions(
                    model,
                    dataset,
                    sample_id=good_idx,
                    interpretability_method_cls=captum_cls,
                )
                fig_bad = xai.cnn_model_attributions(
                    model,
                    dataset,
                    sample_id=bad_idx,
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
                        <b>{probas[good_idx, y]:.2%}</b>
                    </p>
                    <img src='file:{path_good}' width='100%'>
                    <h4>Bad Representative</h4>
                    <p>Morphology name: <b>{morphology_name_bad}</b></p>
                    <p>Probability of belonging to this class:
                        <b>{probas[bad_idx, y]:.2%}</b>
                    </p>
                    <img src='file:{path_bad}' width='100%'>
                    </div>
                    </div>
                    """
                )

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
    figures, labels = xai.cnn_model_attributions_population(model, dataset)

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

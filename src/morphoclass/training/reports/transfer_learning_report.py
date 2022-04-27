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
"""Collect plots from different pretraining datasets and compile a report."""
from __future__ import annotations

import glob
import logging
import pathlib
from itertools import product

from jinja2 import Environment
from jinja2 import FileSystemLoader

from morphoclass.utils import dict2kwargs

logger = logging.getLogger(__name__)


def transfer_learning_report(
    results_file,
    checkpoint_paths_pretrained,
    input_csv,
    features_dir,
    dataset_name,
    feature_extractor_name,
    model_class,
    model_params,
    optimizer_class,
    optimizer_params,
    n_epochs,
    batch_size,
    seed,
):
    """Generate the TL report."""
    from morphoclass.training import transfer_learning_curves

    # create the results directory if it doesn't exist
    results_directory = pathlib.Path(results_file).parent.resolve()
    results_directory.mkdir(exist_ok=True, parents=True)

    images_directory = results_directory / "images"
    checkpoints_directory = results_directory / "checkpoints"

    # read checkpoint files that will me summamrized in this report
    checkpoint_paths_pretrained = glob.glob(checkpoint_paths_pretrained)
    checkpoint_paths_pretrained = [
        pathlib.Path(x)
        for x in checkpoint_paths_pretrained
        if pathlib.Path(x).is_file()
    ]

    transfer_learning_report_final = [
        f"""
        <br/>
        <div class="row">
        <div class='col-md-12'>
        dataset: <b>{dataset_name}</b><br/>
        feature extractor: <b>{feature_extractor_name}</b><br/>
        model class:
        <b>{model_class}({dict2kwargs(model_params)})</b><br/>
        optimizer class:
        <b>{optimizer_class}({dict2kwargs(optimizer_params)})</b><br/>
        </div>
        </div>
    """
    ]
    frozen_backbones = [False]
    for checkpoint_path_pretrained, frozen_backbone in product(
        checkpoint_paths_pretrained, frozen_backbones
    ):
        logger.info(f"Pretrain on {checkpoint_path_pretrained}")
        image_path = images_directory / checkpoint_path_pretrained.stem
        try:
            transfer_learning_curves(
                input_csv,
                features_dir,
                image_path,
                checkpoints_directory,
                model_class,
                model_params,
                dataset_name,
                feature_extractor_name,
                optimizer_class,
                optimizer_params,
                n_epochs,
                batch_size,
                seed,
                checkpoint_path_pretrained=checkpoint_path_pretrained,
                frozen_backbone=frozen_backbone,
            )
            image_rel_path = image_path.with_suffix(".png").relavive_to(
                results_directory
            )
            transfer_learning_report_final.append(
                f"""
                    <br/>
                    <h2> Pretrained on: {checkpoint_path_pretrained}</h2>
                    frozen backbone: <b>{frozen_backbone}</b><br/>
                    <br/>
                    <img src='{image_rel_path}' width='90%'>
                """
            )
            logger.info(f"Update content of {results_file}")
            report_template = (
                pathlib.Path(__file__).resolve().parent
                / "transfer_learning_report_template.html"
            )
            if not report_template.is_file():
                raise FileNotFoundError(
                    "Couldn't find the transfer_learning_report_template.html "
                    f"in {report_template.parent}"
                )

            transfer_learning_report_final_str = "<br/><br/><hr/><br/>".join(
                transfer_learning_report_final
            )

            template_vars = {
                "transfer_learning_report_final": transfer_learning_report_final_str,
            }
            e = Environment(loader=FileSystemLoader(report_template.parent))
            # generate a HTML report
            template = e.get_template(report_template.name)
            html_out = template.render(template_vars, zip=zip)
            with open(results_file, "w") as f:
                f.write(html_out)
        except ValueError:
            logger.info("Skip. Pretrained checkpoint is trained on the same dataset.")

    logger.info("✔ Done.")

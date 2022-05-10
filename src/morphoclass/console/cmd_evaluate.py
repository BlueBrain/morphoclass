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
"""Implementation of the `morphoclass evaluate` CLI command."""
from __future__ import annotations

import logging
import pathlib

import click

from morphoclass.types import StrPath

logger = logging.getLogger(__name__)


@click.group(name="evaluate")
def cli():
    """Run the evaluate subcommand."""
    pass


@cli.command(
    name="performance",
    short_help="Generate a model performance report",
    help="""
    Load the trained checkpoint from CHECKPOINT_PATH and generate a
    performance report. The report will be saved as an HTML file under
    REPORT_PATH.
    """,
)
@click.argument("checkpoint_path", type=click.Path(dir_okay=False, exists=True))
@click.argument("report_path", type=click.Path(dir_okay=False, exists=False))
def performance(checkpoint_path: StrPath, report_path: StrPath) -> None:
    """Run the `evaluate performance` subcommand."""
    logger.info("Setting up paths")
    checkpoint_path = pathlib.Path(checkpoint_path)
    report_path = pathlib.Path(report_path)
    img_dir = report_path.with_suffix("")
    if not report_path.suffix.lower() == ".html":
        raise click.ClickException('The report path must have the extension ".html"')
    if img_dir.exists():
        raise click.ClickException(
            f"The images directory {img_dir.resolve()} already exists."
        )

    logger.info("Loading libraries")
    from morphoclass.console.evaluate import visualize_model_performance
    from morphoclass.training.training_log import TrainingLog

    logger.info("Loading the checkpoint")
    training_log = TrainingLog.load(checkpoint_path)

    logger.info("Generating the performance report")
    visualize_model_performance(training_log, checkpoint_path, img_dir, report_path)

    logger.info("Done.")


@cli.command(
    name="latent-features",
    short_help="Generate plots of latent features.",
    help="""
    This command loads a trained checkpoint from CHECKPOINT_PATH and
    generates an HTML report with model latent features that is saved under
    REPORT_PATH.

    The report contains 2D PCA plots of model latent features for all data
    with highlights for points that were flagged as outliers by CleanLab. The
    plots are generated for model trained on all data, as well as for
    individual cross-validation splits.
    """,
)
@click.argument("checkpoint_path", type=click.Path(dir_okay=False, exists=True))
@click.argument("report_path", type=click.Path(dir_okay=False, exists=False))
def latent_features(checkpoint_path: StrPath, report_path: StrPath) -> None:
    """Run the `evaluate latent-features subcommand."""
    logger.info("Setting up paths")
    checkpoint_path = pathlib.Path(checkpoint_path)
    report_path = pathlib.Path(report_path)
    if not report_path.suffix.lower() == ".html":
        raise click.ClickException('The report path must have the extension ".html"')

    logger.info("Loading libraries")
    from morphoclass import cleanlab

    if not cleanlab.check_installed():
        how_to_install = cleanlab.how_to_install_msg()
        raise click.ClickException(f"CleanLab not installed. {how_to_install}")
    from morphoclass.console.evaluate import visualize_latent_features
    from morphoclass.training.training_log import TrainingLog

    logger.info("Loading the checkpoint")
    training_log = TrainingLog.load(checkpoint_path)

    logger.info("Running CleanLab analysis")
    errors, self_confidence = cleanlab.analysis(
        training_log.targets, training_log.probas
    )

    logger.info("Generating the report")
    visualize_latent_features(training_log, errors, self_confidence, report_path)

    logger.info("Done.")


@cli.command(
    name="outliers",
    short_help="Visualize CleanLab outlier morphologies",
    help="""
    This command loads a trained checkpoint from CHECKPOINT_PATH and
    generates an HTML report with outlier plots that is saved under
    REPORT_PATH.

    The report contains renderings of morphologies that were flagged as
    outliers by CleanLab. Each morphology is rendered as a front and side
    view along with a plot if the corresponding persistence image and
    diagram. CleanLab produces a label confidence score as well as
    probabilities for the morphology to have each of the other labels. These
    scores are included in the outlier report.
    """,
)
@click.argument("checkpoint_path", type=click.Path(dir_okay=False, exists=True))
@click.argument("report_path", type=click.Path(dir_okay=False, exists=False))
def outliers(checkpoint_path: StrPath, report_path: StrPath) -> None:
    """Run the `evaluate outliers` subcommand."""
    logger.info("Setting up paths")
    checkpoint_path = pathlib.Path(checkpoint_path)
    report_path = pathlib.Path(report_path)
    img_dir = report_path.with_suffix("")
    if not report_path.suffix.lower() == ".html":
        raise click.ClickException('The report path must have the extension ".html"')
    if img_dir.exists():
        raise click.ClickException(
            f"The images directory {img_dir.resolve()} already exists."
        )

    logger.info("Loading libraries")
    from morphoclass import cleanlab

    if not cleanlab.check_installed():
        how_to_install = cleanlab.how_to_install_msg()
        raise click.ClickException(f"CleanLab not installed. {how_to_install}")
    from morphoclass.console.evaluate import visualize_cleanlab_outliers
    from morphoclass.training.training_log import TrainingLog

    logger.info("Loading the checkpoint")
    training_log = TrainingLog.load(checkpoint_path)

    logger.info("Running CleanLab analysis")
    errors, self_confidence = cleanlab.analysis(
        training_log.targets, training_log.probas
    )

    # TODO: Plotting morphologies is really slow. Can we make this faster?
    #  Sometimes there are many outliers and this step takes forever.
    logger.info("Visualising outliers")
    visualize_cleanlab_outliers(
        training_log,
        errors,
        self_confidence,
        img_dir,
        report_path,
    )

    logger.info("Done.")

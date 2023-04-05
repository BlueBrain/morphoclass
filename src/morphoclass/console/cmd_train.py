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
"""Implementation of the `morphoclass train-after-extraction` CLI command."""
from __future__ import annotations

import logging
import pathlib

import click
import yaml

from morphoclass.types import StrPath

logger = logging.getLogger(__name__)


@click.command(
    name="train-after-extraction",
    help="""
    Train a morphology classification model.
    Features need to be first extracted.""",
)
@click.option(
    "--features-dir",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="The directory with pre-extracted morphology features.",
)
@click.option(
    "--model-config",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="The model configuration file.",
)
@click.option(
    "--splitter-config",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="The splitter configuration file.",
)
@click.option(
    "--checkpoint-dir",
    type=click.Path(file_okay=False),
    required=True,
    help="The checkpoint output directory.",
)
@click.option(
    "-f",
    "--force",
    type=click.BOOL,
    default=False,
    is_flag=True,
    help="Don't ask for overwriting existing output files.",
)
def cli(
    features_dir: StrPath,
    model_config: StrPath,
    splitter_config: StrPath,
    checkpoint_dir: StrPath,
    force: bool,
) -> None:
    """Training and evaluation of the model."""
    return train(
        features_dir,
        model_config,
        splitter_config,
        checkpoint_dir,
        force,
    )

def train(
    features_dir: StrPath,
    model_config: StrPath,
    splitter_config: StrPath,
    checkpoint_dir: StrPath,
    force: bool,
) -> None:
    """Training and evaluation of the model.

    Parameters
    ----------
    features_dir
        The directory with pre-extracted morphology features.
    model_config
        The path to the model config file.
    splitter_config
        The path to the splitter config file.
    checkpoint_dir
        The checkpoint output directory.
    force
        Never ask the user for interactive input. Existing output files will
        be silently overwritten.
    """
    logger.info("Loading config files")
    with open(model_config) as fh:
        conf_model = yaml.safe_load(fh)
    with open(splitter_config) as fh:
        conf_splitter = yaml.safe_load(fh)

    logger.info("Setting up the output directories")
    features_dir = pathlib.Path(features_dir)
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    checkpoint_path = checkpoint_dir / "checkpoint.chk"
    images_dir = checkpoint_dir / "images"

    logger.info("Loading libraries")
    import xgboost  # noqa, import before torch, otherwise segfault

    from morphoclass import cleanlab

    if not cleanlab.check_installed():
        how_to_install = cleanlab.how_to_install_msg()
        raise click.ClickException(f"CleanLab not installed. {how_to_install}")
    from morphoclass.data.morphology_data import MorphologyData
    from morphoclass.data.morphology_dataset import MorphologyDataset
    from morphoclass.training.cli import ask
    from morphoclass.training.cli import collect_metrics
    from morphoclass.training.cli import plot_confusion_matrices
    from morphoclass.training.cli import run_training
    from morphoclass.training.cli import split_metrics
    from morphoclass.training.training_config import TrainingConfig

    config = TrainingConfig.from_separate_configs(
        conf_model, conf_splitter, features_dir
    )

    logger.info(f"Features    : {features_dir}")
    logger.info(f"Model       : {config.model_class}")
    logger.info(f"Checkpoint  : {checkpoint_path}")
    logger.info(f"Images      : {images_dir}")
    logger.info(f"Seed        : {config.seed}")
    if checkpoint_path.exists():
        if force or ask(f"Checkpoint {str(checkpoint_path)!r} exists, overwrite?"):
            logger.warning("The checkpoint exists: will overwrite")
        else:
            logger.info("Stopping.")
            return

    logger.info("Restoring the dataset from pre-computed features")
    data = []
    # Sorting to ensure reproducibility
    # TODO: consider tracking the order using a CSV like for datasets
    for path in sorted(features_dir.glob("*.features")):
        data.append(MorphologyData.load(path))
    dataset = MorphologyDataset(data)

    logger.info("Training")
    training_log = run_training(dataset, config)
    training_log.interactive = not force

    logger.info("Evaluation")
    # TODO: metrics should be computed at evaluation time, not during training
    # compute metrics
    for i in range(len(training_log.split_history)):
        metrics_dict = collect_metrics(
            training_log.split_history[i]["ground_truths"],
            training_log.split_history[i]["predictions"],
            training_log.labels_str,
        )
        training_log.split_history[i].update(metrics_dict)
        # TODO: ugly renaming accuracy => val_acc_final
        acc = training_log.split_history[i].pop("accuracy")
        training_log.split_history[i]["val_acc_final"] = acc
    metrics_dict = collect_metrics(
        training_log.targets,
        training_log.preds,
        training_log.labels_str,
    )
    training_log.set_metrics(metrics_dict, split_metrics(training_log.split_history))

    # plot the confusion matrices
    # TODO: this replaces training_log.metrics["confusion_matrix"] and
    #       training_log.split_history[i]["confusion_matrix"] by the path of
    #       the plot - why???
    images_dir.mkdir(exist_ok=True, parents=True)
    plot_confusion_matrices(images_dir, training_log)

    # cleanlab outlier detection
    errors, self_confidence = cleanlab.analysis(
        training_log.targets, training_log.probas
    )
    training_log.set_cleanlab_stats(errors, self_confidence)

    logger.info(f"Saving checkpoint to {checkpoint_path}")
    checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
    training_log.save(checkpoint_path)

    logger.info("Done.")

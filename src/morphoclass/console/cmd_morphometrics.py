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
"""Implementation of the `morphoclass morphometrics` CLI command."""
from __future__ import annotations

import logging
import pathlib

import click
import neurom
import pandas as pd
import yaml
from neurom.apps import morph_stats

from morphoclass.training.training_log import TrainingConfig
from morphoclass.training.training_log import TrainingLog

logger = logging.getLogger(__name__)


@click.group(name="morphometrics")
def cli():
    """Run the morphometrics subcommand."""
    pass


@cli.command(
    name="extract-features",
    short_help="Extract morphometrics features",
    help="Extract morphometrics features",
)
@click.option(
    "--organised-dataset-directory",
    "organised_dataset_dir",
    type=click.Path(
        dir_okay=True, file_okay=False, exists=True, path_type=pathlib.Path
    ),
    help="Where to read final preprocessed morphology data.",
    required=True,
)
@click.option(
    "--morphometrics-config",
    "morphometrics_config_file",
    type=click.Path(
        dir_okay=False, file_okay=True, exists=False, path_type=pathlib.Path
    ),
    help="Where to read the YAML morphometric feature configurations.",
    required=True,
)
@click.option(
    "-o",
    "--output-features-directory",
    "output_features_dir",
    type=click.Path(
        dir_okay=True, file_okay=False, exists=False, path_type=pathlib.Path
    ),
    help="Where to write extracted morphometrics features.",
    required=True,
)
def extract_features(
    organised_dataset_dir,
    morphometrics_config_file,
    output_features_dir,
):
    """Run the `morphometrics extract-features` subcommand."""
    logger.info("Morphometrics feature extraction started...")
    logger.info(f" - Input organised data path  : {organised_dataset_dir}")
    logger.info(f" - Morphometrics config       : {morphometrics_config_file}")
    logger.info(f" - Output features path       : {output_features_dir}")

    logger.debug("Read morphometrics config file.")
    if not morphometrics_config_file.is_file():
        raise click.ClickException(
            f"Morphometrics config file not found: {morphometrics_config_file}"
        )
    with morphometrics_config_file.open() as fh:
        morphometrics_config = yaml.safe_load(fh)
    logger.debug(f"Morphometrics configurations: {morphometrics_config}")

    for layer_dir in sorted(f for f in organised_dataset_dir.iterdir() if f.is_dir()):
        logger.info(f"Process morphologies in layer {layer_dir}")

        logger.debug(" - Read CSV file with dataset.")
        dataset_csv = layer_dir / "dataset.csv"
        if not dataset_csv.is_file():
            raise click.ClickException(f"No dataset.csv file found in {layer_dir}")

        logger.debug(" - Load morphologies with NeuroM.")
        df = pd.read_csv(dataset_csv, names=["filepath", "m_type"])
        paths = [layer_dir / file_path for file_path in df["filepath"]]
        population = neurom.load_morphologies(paths)

        logger.debug(" - Extract morphometrics features.")
        df_metrics = morph_stats.extract_dataframe(population, morphometrics_config)

        df_metrics.columns = df_metrics.columns.map("|".join).str.strip("|")
        df["property|name"] = df["filepath"].str.split("/").str[-1]
        df_metrics = df_metrics.merge(df, how="inner", on="property|name")

        logger.debug(f" - Write results to {output_features_dir}.")
        output_features_dir.mkdir(exist_ok=True, parents=True)
        output_file = output_features_dir / f"{layer_dir.stem}.csv"
        df_metrics.to_csv(output_file, index=False)

    logger.info("Done!")


@cli.command(
    name="train",
    short_help="Train classification model based on morphometrics.",
)
@click.option(
    "--morphometrics-features",
    "morphometrics_features_file",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path
    ),
    required=True,
    help="The CSV file with pre-extracted morphology features.",
)
@click.option(
    "--model-config",
    "model_config_file",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path
    ),
    required=True,
    help="The model configuration file.",
)
@click.option(
    "--splitter-config",
    "splitter_config_file",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path
    ),
    required=True,
    help="The splitter configuration file.",
)
@click.option(
    "-o",
    "--output-checkpoint-directory",
    "output_checkpoint_dir",
    type=click.Path(file_okay=False),
    required=True,
    help="The checkpoint output directory.",
)
def train(
    morphometrics_features_file,
    model_config_file,
    splitter_config_file,
    output_checkpoint_dir,
):
    """Run the `morphometrics train` subcommand."""
    from morphoclass.training.cli import collect_metrics
    from morphoclass.training.cli import split_metrics

    logger.info("Morphometrics model training started...")
    logger.info(
        f" - Input morphometrics features file  : {morphometrics_features_file}"
    )
    logger.info(f" - Model config                       : {model_config_file}")
    logger.info(f" - Splitter config                    : {splitter_config_file}")
    logger.info(f" - Output checkpoint dir              : {output_checkpoint_dir}")

    logger.info("Reading config files for model and splitter.")
    with model_config_file.open() as fh:
        model_config = yaml.safe_load(fh)
    with splitter_config_file.open() as fh:
        splitter_config = yaml.safe_load(fh)

    from morphoclass.training.training_config import TrainingConfig

    training_config = TrainingConfig.from_separate_configs(
        conf_model=model_config, conf_splitter=splitter_config
    )

    logger.info("Reading morphometrics features file.")
    df_morphometrics = pd.read_csv(morphometrics_features_file)

    logger.info("Training")
    training_log = run_training(df_morphometrics, training_config)

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

    logger.info(f"Saving checkpoint to {output_checkpoint_dir}")
    output_checkpoint_dir.parent.mkdir(exist_ok=True, parents=True)
    training_log.save(output_checkpoint_dir)

    logger.info("Done!")


# TODO: This function (and the following ones) adapt the content of
#  morphoclass.training.cli. We had to re-implement those functionalities because
#  here we are working with morphometrics,
#  so we cannot have a morphoclass.data.MorphologyDataset.
#  It would however be nice to merge these functionalities together, by creating
#  more generic functions that are independent from the type of Dataset being used.
def run_training(
    df_morphometrics: pd.DataFrame, training_config: TrainingConfig
) -> TrainingLog:
    """Training and evaluation of the model."""
    import numpy as np
    from sklearn.preprocessing import LabelEncoder

    from morphoclass.training._helpers import reset_seeds
    from morphoclass.utils import make_torch_deterministic

    make_torch_deterministic()

    label_enc = LabelEncoder()
    x_features = df_morphometrics.drop(columns=["property|name", "filepath", "m_type"])
    all_labels = label_enc.fit_transform(
        df_morphometrics["m_type"]
    )  # [0, 1, 2, 1, 1, ...]
    labels_unique_str = sorted(df_morphometrics["m_type"])

    if training_config.model_class.startswith("xgboost"):
        training_config.model_params["num_class"] = len(labels_unique_str)
        training_config.model_params["use_label_encoder"] = False  # suppress warning
    elif training_config.model_class.startswith("morphoclass"):
        training_config.model_params["n_classes"] = len(labels_unique_str)

    if training_config.seed is not None:
        reset_seeds(numpy_seed=training_config.seed, torch_seed=training_config.seed)

    splitter = training_config.splitter_cls(**training_config.splitter_params)
    split = splitter.split(X=all_labels, y=all_labels)  # X doesn't matter
    n_splits = splitter.get_n_splits(X=all_labels, y=all_labels)

    probabilities = np.empty((len(all_labels), len(all_labels.unique())))
    predictions = np.empty(len(all_labels), dtype=int)

    training_log = TrainingLog(config=training_config, labels_str=labels_unique_str)
    # SPLIT MODEL
    for n, (train_idx, val_idx) in enumerate(split):
        logger.info(f"Split {n + 1}/{n_splits}, ratio: {len(train_idx)}:{len(val_idx)}")
        history = train_model(
            training_config, train_idx, val_idx, x_features, all_labels
        )
        history["ground_truths"] = all_labels[val_idx]
        history["train_idx"] = list(train_idx)
        history["val_idx"] = list(val_idx)

        all_labels[val_idx] = history["ground_truths"]
        predictions[val_idx] = history["predictions"]
        probabilities[val_idx] = np.array(history["probabilities"])

        # Save split history
        training_log.add_split(history)

    # collect results
    training_log.set_y(all_labels, predictions, probabilities)

    # MODEL ALL
    if training_config.train_all_samples:
        logger.info("Fit model on all samples")
        train_idx = np.arange(len(all_labels))

        history = train_model(
            training_config, train_idx, val_idx, x_features, all_labels
        )
        training_log.set_all_history(history)

    return training_log


def train_model(config, train_idx, val_idx, x_features, all_labels) -> dict:
    """Train a model."""
    pass

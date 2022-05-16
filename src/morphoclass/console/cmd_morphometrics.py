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
    "--dataset-csv",
    "dataset_csv",
    type=click.Path(
        dir_okay=False, file_okay=True, exists=True, path_type=pathlib.Path
    ),
    help="Where to read the CSV with the morphology dataset.",
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
    dataset_csv,
    morphometrics_config_file,
    output_features_dir,
):
    """Run the `morphometrics extract-features` subcommand."""
    import neurom
    import pandas as pd
    import yaml
    from neurom.apps import morph_stats

    logger.info("Morphometrics feature extraction started...")
    logger.info(f" - Input CSV dataset file     : {dataset_csv}")
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

    logger.info(f"Process morphologies in {dataset_csv}")
    logger.debug(" - Read CSV file with dataset.")
    df = pd.read_csv(dataset_csv, names=["filepath", "m_type"])

    logger.debug(" - Load morphologies with NeuroM.")
    parent_dir = dataset_csv.parent
    paths = [parent_dir / file_path for file_path in df["filepath"]]
    population = neurom.load_morphologies(paths)

    logger.debug(" - Extract morphometrics features.")
    df_metrics = morph_stats.extract_dataframe(population, morphometrics_config)
    df_metrics.columns = df_metrics.columns.map("|".join).str.strip("|")
    df["property|name"] = df["filepath"].str.split("/").str[-1]
    df_metrics = df_metrics.merge(df, how="inner", on="property|name")

    logger.debug(f" - Write results to {output_features_dir}.")
    output_features_dir.mkdir(exist_ok=True, parents=True)
    output_file = output_features_dir / "features.csv"
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
        exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path
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
    type=click.Path(file_okay=False, path_type=pathlib.Path),
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
    import pandas as pd
    import yaml

    from morphoclass.morphometrics.cli import run_training
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
        conf_model=model_config,
        conf_splitter=splitter_config,
        features_dir=morphometrics_features_file,
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

    checkpoint_path = output_checkpoint_dir / "checkpoint.chk"
    logger.info(f"Saving checkpoint to {checkpoint_path}")
    checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
    training_log.save(checkpoint_path)

    logger.info("Done!")

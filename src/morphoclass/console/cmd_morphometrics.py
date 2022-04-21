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
) -> None:
    """Run the `morphometrics extract-features` subcommand."""
    logger.info("Start extracting morphometrics features.")
    logger.info(f"Input organised data path  : {organised_dataset_dir}")
    logger.info(f"Morphometrics config       : {morphometrics_config_file}")
    logger.info(f"Output features path       : {output_features_dir}")

    logger.debug("Read morphometrics config file.")
    with morphometrics_config_file.open() as fh:
        morphometrics_config = yaml.safe_load(fh)
    logger.debug(f"Morphometrics configurations: {morphometrics_config}")

    for layer_dir in sorted(f for f in organised_dataset_dir.iterdir() if f.is_dir()):
        logger.info(f"Process morphologies in layer {layer_dir}")

        dataset_csv = layer_dir / "dataset.csv"
        if not dataset_csv.is_file():
            raise click.ClickException(f"No dataset.csv file found in {layer_dir}")

        df = pd.read_csv(dataset_csv, names=["filepath", "m_type"])
        paths = [layer_dir / file_path for file_path in df["filepath"]]

        population = neurom.load_morphologies(paths)
        df_metrics = morph_stats.extract_dataframe(population, morphometrics_config)
        df_metrics.columns = df_metrics.columns.map("|".join).str.strip("|")

        df["property|name"] = df["filepath"].str.split("/").str[-1]
        df_metrics = df_metrics.merge(df, how="inner", on="property|name")

        output_features_dir.mkdir(exist_ok=True, parents=True)
        output_file = output_features_dir / f"features_{layer_dir.stem}.csv"
        df_metrics.to_csv(output_file, index=False)

    logger.info("Done!")


@cli.command(
    name="train",
    short_help="Train classification model based on morphometrics.",
)
@click.argument("checkpoint_path", type=click.Path(dir_okay=False, exists=True))
@click.argument("report_path", type=click.Path(dir_okay=False, exists=False))
def train():
    """Run the `morphometrics train` subcommand."""
    # logger.info("Setting up paths")
    # checkpoint_path = pathlib.Path(checkpoint_path)
    # report_path = pathlib.Path(report_path)
    # if not report_path.suffix.lower() == ".html":
    #     raise click.ClickException('The report path must have the extension ".html"')
    #
    # logger.info("Loading libraries")
    # from morphoclass import cleanlab
    #
    # if not cleanlab.check_installed():
    #     how_to_install = cleanlab.how_to_install_msg()
    #     raise click.ClickException(f"CleanLab not installed. {how_to_install}")
    # from morphoclass.console.evaluate import visualize_latent_features
    # from morphoclass.training.training_log import TrainingLog
    #
    # logger.info("Loading the checkpoint")
    # training_log = TrainingLog.load(checkpoint_path)
    #
    # logger.info("Running CleanLab analysis")
    # errors, self_confidence = cleanlab.analysis(
    #     training_log.targets, training_log.probas
    # )
    #
    # logger.info("Generating the report")
    # visualize_latent_features(training_log, errors, self_confidence, report_path)

    logger.info("Done.")

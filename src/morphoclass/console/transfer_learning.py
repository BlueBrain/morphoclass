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
"""Implementation of the `morphoclass train_and_eval` CLI command."""
from __future__ import annotations

import click

from morphoclass.console import helpers


@click.command(name="transfer-learning", help="Transfer learning.")
@click.option(
    "--results-file",
    required=True,
    type=click.Path(exists=False, dir_okay=False),
    help="Path to params.yaml config file.",
)
@click.option(
    "--checkpoint-paths-pretrained",
    type=click.STRING,
    required=False,
    help="Checkpoint paths of pretrained models wildcard.",
)
@click.option(
    "--input-csv",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Input csv path.",
)
@click.option(
    "--features-dir",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="The directory with the pre-extracted dataset features.",
)
@click.option(
    "--dataset-name",
    type=click.STRING,
    required=False,
    help="Dataset name.",
)
@click.option(
    "--feature-extractor-name",
    type=click.STRING,
    required=False,
    help="Name of the feature extractor.",
)
@click.option(
    "--model-class",
    type=click.STRING,
    required=False,
    help="Model class.",
)
@click.option(
    "--model-params",
    type=click.STRING,
    required=False,
    callback=helpers.validate_params,
    help="Model parameters, e.g., 'a=1 b=2'.",
)
@click.option(
    "--optimizer-class",
    type=click.STRING,
    required=False,
    help="Optimizer class.",
)
@click.option(
    "--optimizer-params",
    type=click.STRING,
    required=False,
    callback=helpers.validate_params,
    help="Optimizer parameters, e.g., 'a=1 b=2'.",
)
@click.option(
    "--n-epochs",
    type=click.INT,
    required=False,
    help="Number of epochs.",
)
@click.option(
    "--batch-size",
    type=click.INT,
    required=False,
    help="Batch size.",
)
@click.option(
    "--seed",
    type=click.INT,
    required=False,
    help="Seed.",
)
def cli(
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
    """Training and evaluation of the model.

    Parameters
    ----------
    results_file : str or pathlib.Path
        The HTML report output path.
    checkpoint_paths_pretrained : str
        A glob pattern that resolves all checkpoint files.
    input_csv : str, pathlib.Path
        Input csv path.
    model_class : str
        Model class.
    model_params : str
        Model parameters, e.g., 'a=1 b=2'.
    dataset_name : str
        Dataset name.
    feature_extractor_name :
        Name of the feature extractor.
    optimizer_class : str
        Optimizer class.
    optimizer_params : str
        Optimizer parameters, e.g., 'a=1 b=2'.
    n_epochs : int
        Number of epochs.
    batch_size : int
        Batch size.
    seed : int
        Seed.
    """
    from morphoclass.training import transfer_learning_report

    transfer_learning_report(
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
    )

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
"""Implementation of the `morphoclass predict` CLI command."""
from __future__ import annotations

import logging
import textwrap

import click

logger = logging.getLogger(__name__)


@click.command(
    name="predict",
    help="Run inference.",
)
@click.help_option("-h", "--help")
@click.option(
    "-i",
    "--input_csv",
    required=True,
    type=click.Path(exists=True, dir_okay=True),
    help=textwrap.dedent(
        """
        The CSV path with the path to all the morphologies to classify.
        """
    ).strip(),
)
@click.option(
    "-c",
    "--checkpoint",
    "checkpoint_file",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help=textwrap.dedent(
        """
        The path to the pre-trained model checkpoint.
        """
    ).strip(),
)
@click.option(
    "-o",
    "--output-dir",
    required=True,
    type=click.Path(exists=False, file_okay=False, writable=True),
    help="Output directory for the results.",
)
@click.option(
    "-n",
    "--results-name",
    required=False,
    type=click.STRING,
    help="The filename of the results file",
)
def cli(input_csv, checkpoint_file, output_dir, results_name):
    """Run the `morphoclass predict` CLI command.

    Parameters
    ----------
    input_csv
        The CSV with all the morphologies path.
    checkpoint_file
        The path to the checkpoint file.
    output_dir
        The path to the output directory.
    results_name
        File prefix for results output files.
    """
    import pathlib
    from datetime import datetime

    input_csv = pathlib.Path(input_csv).resolve()
    output_dir = pathlib.Path(output_dir).resolve()
    checkpoint_file = pathlib.Path(checkpoint_file).resolve()
    if results_name is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_name = f"results_{timestamp}"
    results_path = output_dir / (results_name + ".json")
    click.secho(f"Input CSV   : {input_csv}", fg="yellow")
    click.secho(f"Output file : {results_path}", fg="yellow")
    click.secho(f"Checkpoint  : {checkpoint_file}", fg="yellow")
    if results_path.exists():
        msg = f'Results file "{results_path}" exists, overwrite? (y/[n]) '
        click.secho(msg, fg="red", bold=True, nl=False)
        response = input()
        if response.strip().lower() != "y":
            click.secho("Stopping.", fg="red")
            return
        else:
            click.secho("You chose to overwrite, proceeding...", fg="red")

    click.secho("✔ Loading checkpoint...", fg="green", bold=True)
    import torch

    from morphoclass.console.cmd_extract_features import extract_features
    from morphoclass.console.cmd_predict import predict

    checkpoint = torch.load(checkpoint_file, map_location=torch.device("cpu"))
    neurites = ["apical", "axon", "basal", "all"]
    neurite_type = [
        neurite for neurite in neurites if neurite in str(checkpoint["features_dir"])
    ]
    features_type = [
        "graph-rd",
        "graph-proj",
        "diagram-tmd-rd",
        "diagram-tmd-proj",
        "diagram-deepwalk",
        "image-tmd-rd",
        "image-tmd-proj",
        "image-deepwalk",
    ]
    feature = [
        feature
        for feature in features_type
        if feature in str(checkpoint["features_dir"])
    ]

    extract_features(
        input_csv,
        neurite_type,
        feature,
        output_dir / "features",
        False,
        False,
        False,
        False,
    )

    predict(
        features_dir=output_dir / "features",
        checkpoint_file=checkpoint_file,
        output_dir=output_dir,
        results_name=results_name,
    )

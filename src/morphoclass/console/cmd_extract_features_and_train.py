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
"""Implementation of the `morphoclass train` CLI command."""
from __future__ import annotations

import logging
import pathlib
from typing import Literal

import click

from morphoclass.types import StrPath

logger = logging.getLogger(__name__)


@click.command(name="train", help="Train a morphology classification model.")
@click.argument("csv_path", type=click.Path(dir_okay=False))
@click.argument("neurite_type", type=click.Choice(["apical", "axon", "basal", "all"]))
@click.argument(
    "feature",
    type=click.Choice(
        [
            "graph-rd",
            "graph-proj",
            "diagram-tmd-rd",
            "diagram-tmd-proj",
            "diagram-deepwalk",
            "image-tmd-rd",
            "image-tmd-proj",
            "image-deepwalk",
        ]
    ),
)
@click.option(
    "--orient",
    is_flag=True,
    help="Orient the neurons so that the apicals are aligned with the positive y-axis.",
)
@click.option(
    "--no-simplify-graph",
    is_flag=True,
    help="""
    By default the neurite graph is reduced to branching nodes only. With this
    flag the full neurite graph will be preserved.
    """,
)
@click.option(
    "--keep-diagram",
    is_flag=True,
    help="After converting the diagram to persistence image don't discard the diagram.",
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
    "--output-dir",
    type=click.Path(file_okay=False),
    required=True,
    help="The output directory.",
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
    csv_path: StrPath,
    neurite_type: Literal["apical", "axon", "basal", "all"],
    feature: Literal[
        "graph-rd",
        "graph-proj",
        "diagram-tmd-rd",
        "diagram-tmd-proj",
        "diagram-deepwalk",
        "image-tmd-rd",
        "image-tmd-proj",
        "image-deepwalk",
    ],
    orient: bool,
    no_simplify_graph: bool,
    keep_diagram: bool,
    model_config: StrPath,
    splitter_config: StrPath,
    output_dir: StrPath,
    force: bool,
) -> None:
    """Extract features and train the model."""
    from morphoclass.console.cmd_extract_features import extract_features
    from morphoclass.console.cmd_train import train

    input_csv = pathlib.Path(csv_path).resolve()
    output_dir = pathlib.Path(output_dir).resolve()

    extract_features(
        input_csv,
        neurite_type[0],
        feature[0],
        output_dir / "features",
        orient,
        no_simplify_graph,
        keep_diagram,
        force,
    )

    train(
        output_dir / "features",
        model_config,
        splitter_config,
        output_dir / "checkpoints",
    )

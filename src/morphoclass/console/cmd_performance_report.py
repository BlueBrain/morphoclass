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
"""Script to collect all models' performances."""
from __future__ import annotations

import logging

import click

from morphoclass.types import StrPath

logger = logging.getLogger(__name__)


@click.command(
    name="performance-report",
    help="""
    Generate a summary report about the performance of all trained models.

    This command will load all model checkpoints from the provided
    checkpoint directory and compile their performance metrics into
    an HTML report.
    """,
)
@click.help_option("-h", "--help")
@click.option(
    "--checkpoint-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="The directory with all checkpoint files.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(exists=False, file_okay=False),
    help="The HTML report output directly.",
)
def cli(checkpoint_dir: StrPath, output_dir: StrPath) -> None:
    """Compile the table with models' results.

    Parameters
    ----------
    checkpoint_dir
        The directory with all checkpoint files.
    output_dir
        The HTML report output directly.
    """
    from morphoclass.console.performance_report import make_performance_report

    make_performance_report(checkpoint_dir, output_dir)

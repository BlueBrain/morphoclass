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
"""Default data preparation script shared across different datasets."""
from __future__ import annotations

import logging
import pathlib

import click

logger = logging.getLogger(__name__)


@click.command(
    name="plot-dataset-stats",
    help="""
    Ingest a morphology dataset through a CSV file and create a detailed
    report.

    The report contains sample count histograms, statistics on
    the neurite graph sizes and the sizes of the corresponding persistence
    diagrams. The report also contains renders of randomly selected
    morphologies for each m-type as well as the corresponding persistence
    diagrams and images.
    """,
)
@click.option(
    "--input-csv-path",
    type=click.Path(dir_okay=False),
    required=True,
    help="The dataset CSV file.",
)
@click.option(
    "--output-report-path",
    type=click.Path(dir_okay=False),
    required=True,
    help="The path for the output HTML file.",
)
def cli(input_csv_path: str, output_report_path: str) -> None:
    """Run the plot-dataset-stats subcommand."""
    logger.info("Imports")
    from morphoclass.console.stats_plotter import DatasetStatsPlotter

    csv_path = pathlib.Path(input_csv_path)
    report_path = pathlib.Path(output_report_path)

    logger.info("Starting plotting")
    plotter = DatasetStatsPlotter(csv_path, report_path)
    plotter.run()
    plotter.save_report(f"Dataset Plots - {report_path.stem}")

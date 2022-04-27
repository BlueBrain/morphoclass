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
"""The preprocess-dataset subcommand."""
from __future__ import annotations

import logging

import click

from morphoclass.constants import DatasetType

logger = logging.getLogger(__name__)


@click.command(
    name="preprocess-dataset",
    help="""
    Preprocess a raw morphology dataset.

    This command will:
    * Read a given database file in MorphDB format
    * Read the morphologies from the given directory. The morphology paths
      have to match those in the database file.
    * Remove rows with equal paths in the database. This can happen if the
      same morphology is assigned to two different cortical layers. We don't
      need the information on the cortical layer and therefore can discard
      the duplicates.
    * Remove all m-type classes with only one morphology.
    * Find and report duplicate morphologies using morph_tool. It can happen
      that different morphology files with different file names contain the
      same morphology.
    * For interneurons only all morphologies where the m-type contains "PC"
      or is equal to "L4_SCC" will be dropped.
    * A report with all the actions taken will be saved to disk.
    """,
)
@click.option(
    "--dataset-type",
    type=click.Choice([member.value for member in DatasetType]),
    required=True,
    help="They type of the morphology dataset",
)
@click.option(
    "--morphologies-dir",
    type=click.Path(file_okay=False),
    required=True,
    help="""
    The directory with all morphology files. It is assumed to be
    flat, i.e. no sub-directories and the files therein will be
    considered.
    """,
)
@click.option(
    "--db-file",
    type=click.Path(dir_okay=False),
    required=True,
    help="""
    The path to the neurondb file compatible with the ``morph_tool``
    specifications. Typical file extensions for this file are DAT or XML.
    See ``morph_tool.morphdb.MorphDB`` for more details.
    """,
)
@click.option(
    "--output-csv-path",
    type=click.Path(dir_okay=False),
    required=True,
    help="The path for the output CSV file.",
)
@click.option(
    "--output-report-path",
    type=click.Path(dir_okay=False),
    required=True,
    help="The path for the HTML report file.",
)
def cli(
    dataset_type: str,
    morphologies_dir: str,
    db_file: str,
    output_csv_path: str,
    output_report_path: str,
) -> None:
    """Run the preprocess-dataset command.

    Parameters
    ----------
    dataset_type
        The dataset type. Must match one of the values of
        `morphoclass.constants.DatasetType`. The dataset type determines
        some of the custom dataset processing.
    morphologies_dir
        The directory with all the dataset morphology files.
    db_file
        The path to the neurondb file compatible with the `morph_tool`
        specifications. Typical file extensions for this file are DAT or XML.
        See `morph_tool.morphdb.MorphDB` for more details.
    output_csv_path
        The path for the output CSV file.
    output_report_path
        The path for the HTML report file.
    """
    logger.info("Imports")
    from morphoclass.console.dataset_preprocessor import Preprocessor

    logger.info("Initialising the pre-processor")
    preprocessor = Preprocessor(
        db_file, morphologies_dir, output_report_path, output_csv_path
    )
    preprocessor.run(DatasetType(dataset_type))
    preprocessor.save_dataset_csv()
    preprocessor.save_report(f"Dataset Report: {dataset_type}")

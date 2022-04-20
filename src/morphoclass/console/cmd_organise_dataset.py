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
import shutil

import click

logger = logging.getLogger(__name__)


@click.command(
    "organise-dataset",
    help="""
    Read a morphology dataset from a CSV file and organise the samples
    contained in it into subdirectories according to their m-type.

    For every sample in the dataset the m-type is expected to be of the
    form "layer_type", e.g. the m-type "L5_TPC_B" has layer "L5" and type
    "TPC_B". The corresponding morphology file is the copied to the
    directory "<output_dir>/<layer>/<type>".

    Further more new dataset CSV files are created. Each of them has tow
    columns without headers. The first column is the morphology file path,
    the second is the m-type. The file "<output_dir>/dataset.csv" contains
    the complete dataset with all layers included, while the files
    "<output_dir>/<layer>/dataset.csv" contain only the subset of the
    morphologies for the respective layer.
    """,
)
@click.option(
    "--input-csv-path",
    type=click.Path(dir_okay=False, writable=True),
    required=True,
    help='The dataset CSV file. Should have columns "morph_path" and "mtype"',
)
@click.option(
    "--output-dataset-directory",
    "output_dir",
    type=click.Path(file_okay=False, writable=True),
    required=True,
    help="Output directory path.",
)
def cli(input_csv_path: str, output_dir: str | pathlib.Path) -> None:
    """Organizing ML data.

    Parameters
    ----------
    input_csv_path
        The CSV file with the morphology dataset. Should have columns
        "morph_path" and "mtype".
    output_dir
        The directory for the output dataset.
    """
    logger.info("Imports")
    import pandas as pd

    logger.info("Organising morphology files by their layer and m-type")
    output_dir = pathlib.Path(output_dir)
    df_data = pd.read_csv(input_csv_path)[["morph_path", "mtype"]]
    df_data = df_data.rename({"mtype": "full_mtype"}, axis=1)
    df_data[["layer", "mtype"]] = df_data["full_mtype"].str.split("_", n=1, expand=True)

    def make_output_path(row):
        morph_path = pathlib.Path(row["morph_path"])
        output_path = output_dir / row["layer"] / row["mtype"] / morph_path.name
        return str(output_path)

    df_data["new_morph_path"] = df_data.apply(make_output_path, axis=1)
    for row in df_data.itertuples():
        new_morph_path = pathlib.Path(row.new_morph_path)
        new_morph_path.parent.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(row.morph_path, new_morph_path)

    def change_base_dir(path: str, new_base: pathlib.Path) -> str:
        new_path = pathlib.Path(path).resolve().relative_to(new_base.resolve())
        return str(new_path)

    logger.info("Saving the CSV file for the complete dataset")
    df_out = df_data[["new_morph_path", "full_mtype"]].sort_values(by="new_morph_path")
    csv_path = output_dir / "dataset.csv"
    df_out["new_morph_path"] = df_out["new_morph_path"].map(
        lambda path: change_base_dir(path, csv_path.parent)
    )
    df_out.to_csv(csv_path, index=False, header=None)

    logger.info("Saving dataset CSV files for individual layers")
    for layer in sorted(df_data["layer"].unique()):
        df_out = df_data[df_data["layer"] == layer]
        df_out = df_out[["new_morph_path", "full_mtype"]]
        df_out = df_out.sort_values(by="new_morph_path")
        csv_path = output_dir / layer / "dataset.csv"
        df_out["new_morph_path"] = df_out["new_morph_path"].map(
            lambda path: change_base_dir(path, csv_path.parent)
        )
        df_out.to_csv(csv_path, index=False, header=None)

    logger.info("Finished")

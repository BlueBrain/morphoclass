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
"""Utilities for the morphoclass performance-table command."""
from __future__ import annotations

import logging
import pathlib
from collections.abc import Sequence

import click
import pandas as pd
import torch
from pandas.io.formats.style import Styler

import morphoclass as mc
import morphoclass.report.plumbing
from morphoclass.types import StrPath

logger = logging.getLogger(__name__)


def make_performance_table(
    checkpoint_paths: Sequence[StrPath], output_dir: StrPath
) -> None:
    """Create a performance report for trained models.

    Parameters
    ----------
    checkpoint_paths
        All checkpoints to be included in the summary  table.
    output_dir
        The report output directory.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Resolve paths from parameters
    files = [pathlib.Path(path).resolve() for path in checkpoint_paths]
    output_dir = pathlib.Path(output_dir).resolve()
    output_dir.mkdir(exist_ok=True, parents=True)

    # Additional output paths
    results_file = output_dir / "results_table.html"
    best_models_path = output_dir / "best_models.txt"

    # read checkpoint files that will be summarized in this report
    if len(files) == 0:
        raise click.ClickException("At least one checkpoint must be given.")
    logger.info(f"Found {len(files)} checkpoint files")

    table_rows = []
    for i, metrics_file in enumerate(files):
        logger.info(
            f"Processing checkpoint {i + 1} of {len(files)}: {str(metrics_file)!r}"
        )
        data = torch.load(metrics_file, map_location=device)

        data["metrics_file"] = metrics_file

        table_rows.append(make_report_row(data))

    df = pd.DataFrame(table_rows)
    # sort dataframe by f1 score and dataset
    df = df.sort_values(by=["dataset", "f1_weighted"], ascending=[True, False])
    df = df.reset_index(drop=True).reset_index()
    # map the links to corresponding model HTML

    # collect paths to best performing models
    mask = df.duplicated(subset="dataset", keep="first")
    df_best_models = df[~mask]["metrics_file"]
    df_best_models.to_csv(best_models_path, header=False, index=False)

    # checkpoint name
    df["checkpoint_name"] = df["metrics_file"]
    # work on table with performances
    # the same order will be in the report
    table_columns = [
        "dataset",
        "feature_extractor",
        "model_class",
        "oversampled",
        "splitter_class",
        "accuracy",
        "f1_micro",
        "f1_macro",
        "f1_weighted",
        "checkpoint_name",
    ]
    df = df[table_columns]

    # Apply styling to the HTML table
    # Remove the random UUID from the HTML table "id" property and also
    # don't include the "id" properties in the cells. This makes the table
    # generation reproducible and removes unnecessary HTML garbage.
    # Ref:
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html#1.-Remove-UUID-and-cell_ids
    df_style = Styler(df, uuid_len=0, cell_ids=False)
    cell_style = "padding-right: 1em; white-space: nowrap;"
    df_style.set_table_styles([{"selector": "th, td", "props": cell_style}])

    # generate a HTML report
    template = mc.report.plumbing.load_template("results-table")
    mc.report.plumbing.render(template, {"df_results": df_style.render()}, results_file)
    logger.info(f"Report stored in: {results_file.resolve().as_uri()}")

    logger.info("Done.")


def make_report_row(data: dict) -> dict:
    """Create a row for the summary report table.

    Parameters
    ----------
    data
        The checkpoint data.

    Returns
    -------
    dict
        The dictionary representing a data frame row.
    """
    return {
        "dataset": data["dataset_name"],
        "feature_extractor": data["features_dir"],
        "model_class": data["model_class"],
        "model_params": data["model_params"],
        "oversampled": bool(data["oversampling"]),
        "splitter_class": data["splitter_class"].rpartition(".")[2],
        "splitter_params": data["splitter_params"],
        # "accuracy_mean": data["accuracy_mean"],
        "accuracy": f"{data['accuracy_mean']:.3f}±{data['accuracy_std']:.2f}",
        # "f1_weighted_mean": data["f1_weighted_mean"],
        "f1_weighted": f"{data['f1_weighted_mean']:.3f}±{data['f1_weighted_std']:.2f}",
        "f1_micro": f"{data['f1_micro_mean']:.3f}±{data['f1_micro_std']:.2f}",
        "f1_macro": f"{data['f1_macro_mean']:.3f}±{data['f1_macro_std']:.2f}",
        "metrics_file": data["metrics_file"],
        # "model_report_directory": data["model_report_directory"],
    }

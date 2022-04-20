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
"""Implementation of the data preprocessor."""
from __future__ import annotations

import logging
import pathlib
from itertools import combinations

import pandas as pd
from morph_tool import diff
from morph_tool.morphdb import MorphDB
from morphio import Morphology

from morphoclass import report
from morphoclass.constants import DatasetType

logger = logging.getLogger(__name__)


class Preprocessor:
    """The data preprocessor.

    Parameters
    ----------
    neurondb_path
        The path to the neurondb file compatible with the `morph_tool`
        specifications. Typical file extensions for this file are DAT or XML.
        See `morph_tool.morphdb.MorphDB` for more details.
    morphology_dir
        The directory with all the dataset morphology files.
    report_path
        The path for the HTML report file.
    csv_path
        The path for the output CSV file.
    """

    def __init__(self, neurondb_path, morphology_dir, report_path, csv_path):
        self.morphology_dir = pathlib.Path(morphology_dir)
        self.csv_path = pathlib.Path(csv_path)
        self.report_path = pathlib.Path(report_path)
        self.template_vars = {
            "neurondb_path": neurondb_path,
            "morphology_dir": morphology_dir,
            "csv_path": csv_path,
            "report_path": report_path,
        }

        logger.info("Loading the MorphDB file")
        db = MorphDB.from_neurondb(neurondb_path, morphology_folder=morphology_dir)
        self.df_morph = db.df[["name", "mtype", "layer", "path"]]

        self.template_vars["morphdb_size"] = len(self.df_morph)
        self.template_vars["n_labels"] = len(self.df_morph.mtype.unique())

    def _preprocess_interneurons(self) -> None:
        """Preprocess interneurons."""
        logger.info("Custom processing for interneurons")

        drop_mask = self.df_morph.mtype.str.contains("PC") | (
            self.df_morph.mtype == "L4_SCC"
        )
        df_morph = self.df_morph[~drop_mask]
        df_drop = self.df_morph[drop_mask]
        self.df_morph = df_morph.copy()
        dropped_labels = sorted(df_drop.mtype.unique())
        self.template_vars["dropped_by_label"] = dropped_labels
        self.template_vars["n_dropped_by_label"] = len(dropped_labels)

    def _preprocess_pyramidal_cells(self) -> None:
        logger.info("There is no custom processing for pyramidal cells")

    def custom_preprocess(self, dataset_type: DatasetType) -> None:
        """Run custom processing depending on the dataset type.

        In this method, user is able to modify the dataset depending on
        specific requirements. Some things that can be modified are:

        * ``ml_data.df_morph`` is a dataframe with columns: ``name``,
          ``mtype``, ``path``, and user can filter out the rows containing
          morphologies not needed in this dataset (e.g. interneurons dataset
          contained some PC cells).
        * ``ml_data.template_vars`` is a ``jinja2`` dictionary that will be
          used to generate the report with raw data statistics
          (``report_rawdata.pdf``). If you add new keys, don't forget to
          modify the corresponding ``report_template.html``. Be careful,
          to add new key, use:
          ``ml_data.template_vars.update({"new_key":"new_value"})`` which
          won't overwrite already existing values in the dictionary.
        * For other options, check out the ``data.OrganizeMLData`` class.
        """
        if dataset_type == DatasetType.pyramidal:
            self._preprocess_pyramidal_cells()
        elif dataset_type == DatasetType.interneurons:
            self._preprocess_interneurons()
        else:
            raise ValueError(
                f"Data preparation for dataset {dataset_type} is not implemented."
            )

    def run(self, dataset_type: DatasetType) -> None:
        """Preprocess the data and save dataset.csv."""
        logger.info("Running the pre-processing")

        logger.info("Dropping duplicated files")
        self.drop_duplicate_files()
        logger.info("Dropping 1-neuron classes")
        self.drop_mtypes_with_one_neuron()
        logger.info("Collecting duplicate morphologies")
        self.collect_duplicated_morphologies()
        logger.info("Running custom preprocessing")
        self.custom_preprocess(dataset_type)

        logger.info("Computing the m-type counts")
        self.template_vars["df_mtype_counts"] = (
            self.df_morph.mtype.value_counts()
            .to_frame()
            .sort_index()
            .reset_index()
            .rename({"index": "m-type", "mtype": "count"}, axis=1)
            .style.hide_index()
            .render()
        )

    def drop_duplicate_files(self) -> None:
        """Filter out the duplicates, but store dropped for the report."""
        mask_duplicates = self.df_morph["path"].duplicated(keep=False)
        mask_drop = self.df_morph["path"].duplicated(keep="first")
        df_duplicates = self.df_morph[mask_duplicates]
        self.df_morph = self.df_morph[~mask_drop].copy()

        def color_dropped(row: pd.Series) -> list[str]:
            color = "#FF000044" if mask_drop[row.name] else "white"
            return [f"background-color: {color}"] * len(row)

        self.template_vars["df_duplicate_files"] = df_duplicates.style.apply(
            color_dropped, axis=1
        ).render()
        self.template_vars["n_duplicate_files"] = sum(mask_drop)

    def drop_mtypes_with_one_neuron(self) -> None:
        """Work only with the classes that have >=2 neurons."""
        # get only the classes that have >=2 neurons
        counts = self.df_morph.mtype.value_counts()
        good_labels = set(counts[counts > 1].index)
        bad_labels = set(counts[counts <= 1].index)
        self.df_morph = self.df_morph[self.df_morph.mtype.isin(good_labels)]

        self.template_vars["labels_one_neuron"] = sorted(bad_labels)
        self.template_vars["n_labels_one_neuron"] = len(bad_labels)

    def collect_duplicated_morphologies(self) -> None:
        """Collect duplicated morphologies based on file content.

        Collect for the report the neurons that have the same morphology,
        but happens they have different file names (bad if it happens...)
        """
        paths = sorted(self.morphology_dir.glob("*.h5"))
        logger.info(f"Loading all {len(paths)} morphologies")
        morphologies = {path: Morphology(path) for path in paths}

        logger.info("Checking for duplicate morphologies")
        duplicates = []
        for (path1, morph1), (path2, morph2) in combinations(morphologies.items(), 2):
            logger.debug(f"Comparing {path1.name} and {path2.name}")
            are_different = diff(morph1, morph2)
            if not are_different:
                logger.info(f"Same morphology detected: {path1.name} = {path2.name}")
                duplicates.append(
                    {
                        "morphology 1": path1.name,
                        "morphology 2": path2.name,
                        "info": are_different.info or "",
                    }
                )
        df_duplicated = pd.DataFrame(duplicates)
        self.template_vars["df_duplicated_morphologies"] = df_duplicated.style.render()

    def save_dataset_csv(self) -> None:
        """Save the dataset CSV file to disk."""
        logger.info(f"Saving the CSV file to {self.csv_path.resolve().as_uri()}")
        self.csv_path.parent.mkdir(exist_ok=True, parents=True)

        # The subsequent M-CAR curation workflow stage requires that the columns
        # be names "morph_name" and "morph_path".
        df_out = self.df_morph[["name", "path", "mtype"]]
        df_out = df_out.rename({"name": "morph_name", "path": "morph_path"}, axis=1)
        df_out.to_csv(self.csv_path, index=False)

    def save_report(self, report_title: str = "Dataset Report") -> None:
        """Write the report to disk."""
        logger.info(f"Saving the report file to {self.report_path.resolve().as_uri()}")
        self.template_vars["report_title"] = report_title
        self.report_path.parent.mkdir(exist_ok=True, parents=True)
        template = report.load_template("preprocess-report")
        report.render(template, self.template_vars, self.report_path)

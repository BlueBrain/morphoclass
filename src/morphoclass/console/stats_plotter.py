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
from typing import Any

import matplotlib.style
import pandas as pd
import tmd
from matplotlib.figure import Figure
from morphio import Morphology
from tmd.Topology.methods import get_persistence_diagram

import morphoclass as mc
import morphoclass.report.plumbing
from morphoclass.vis import plot_counts_per_layer
from morphoclass.vis import plot_counts_per_subclass
from morphoclass.vis import plot_histogram
from morphoclass.vis import plot_histogram_panel
from morphoclass.vis import plot_morphology_images

logger = logging.getLogger(__name__)


class DatasetStatsPlotter:
    """Plotter for morphology dataset statistics.

    Parameters
    ----------
    dataset_csv_path
        The path for the morphology dataset CSV file.
    report_path
        The output path for the HTML report.
    """

    def __init__(
        self,
        dataset_csv_path: pathlib.Path,
        report_path: pathlib.Path,
        morph_name_column: str = "morph_name",
        morph_path_column: str = "morph_path",
        mtype_column: str = "mtype",
    ) -> None:
        self.report_path = report_path
        self.report_dir = self.report_path.parent
        self.images_dir = self.report_dir / (self.report_path.stem + "-images")

        self.df_dataset = pd.read_csv(
            dataset_csv_path,
            usecols=[morph_name_column, morph_path_column, mtype_column],
        ).rename(
            {
                morph_name_column: "name",
                morph_path_column: "path",
                mtype_column: "mtype",
            },
            axis=1,
        )
        self.template_vars: dict[str, Any] = {"dataset_csv_path": dataset_csv_path}

        self._neurons: dict | None = None
        self._df_stats: pd.DataFrame | None = None

    @property
    def neurons(self) -> dict[str, tmd.Neuron.Neuron]:
        """Get a map from a neuron morphology file path to the loaded neuron."""
        if self._neurons is None:
            logger.info("Loading neurons")
            self._neurons = {}
            for row in self.df_dataset.itertuples():
                logger.debug(f"Loading {row.path}")
                self._neurons[row.path] = self.load_neuron(row.path)
            logger.info("Done loading neurons")

        return self._neurons

    @staticmethod
    def load_neuron(path: str) -> tmd.Neuron.Neuron:
        """Load a TMD neuron given a file path."""
        morphology = Morphology(path)
        neuron = tmd.Neuron.Neuron(name=pathlib.Path(path).stem)
        for tree in tmd.io.convert_morphio_trees(morphology):
            neuron.append_tree(tree, tmd.utils.TREE_TYPE_DICT)

        return neuron

    def savefig(self, fig: Figure, path: pathlib.Path) -> pathlib.Path:
        """Save a figure and return the input path."""
        fig.savefig(path)
        return path.relative_to(self.report_dir)

    def run(self) -> None:
        """Run all plotting routines and save all plots to disk."""
        self.images_dir.mkdir(exist_ok=True, parents=True)
        with matplotlib.style.context("ggplot"):
            self.make_data_count_plots()
            for neurite in ["axon", "apical", "basal", "neurites"]:
                self.make_node_count_plots(neurite)
            self.make_neuron_preview_plots()

    def make_data_count_plots(self) -> None:
        """Make data count plots and save them to disk."""
        logger.info("Plotting sample counts")
        fig = plot_counts_per_subclass(self.df_dataset["mtype"])
        path = self.images_dir / "counts_per_subclass.png"
        self.template_vars["counts_per_mtype"] = self.savefig(fig, path)

        fig = plot_counts_per_layer(self.df_dataset["mtype"])
        path = self.images_dir / "counts_per_layer.png"
        self.template_vars["counts_per_layer"] = self.savefig(fig, path)

    @property
    def df_stats(self) -> pd.DataFrame:
        """Get a data frame with dataset statistics.

        The data frame contains the following columns:

        * path: the neuron morphology file path
        * layer: the m-type layer; e.g. the m-type L5_UPC corresponds
          to layer L5
        * {axon,apical,basal,neurites}-size: the node count in the
          respective neurite graph
        * {axon,apical,basal,neurites}-pd-size: the point count in the
          respective persistence diagram.

        Returns
        -------
        pd.DataFrame: The data frame with dataset statistics
        """

        def mean_graph_size(neuron, tree_type=None):
            trees = getattr(neuron, tree_type)
            if len(trees) == 0:
                return 0
            return sum(len(tree.x) for tree in trees) / len(trees)

        def mean_pd_size(neuron, tree_type=None):
            trees = getattr(neuron, tree_type)
            n = len(trees)
            if n == 0:
                return 0
            return sum(len(get_persistence_diagram(tree)) for tree in trees) / n

        def mtype_to_layer(mtype):
            return mtype.partition("_")[0]

        if self._df_stats is None:
            logger.info("Computing dataset stats")
            self._df_stats = self.df_dataset[["path"]].copy()
            self._df_stats["layer"] = self.df_dataset["mtype"].map(mtype_to_layer)
            for neurite in ["axon", "apical", "basal", "neurites"]:
                self._df_stats[f"{neurite}-size"] = self._df_stats["path"].map(
                    lambda path: mean_graph_size(self.neurons[path], neurite)
                )
                self._df_stats[f"{neurite}-pd-size"] = self._df_stats["path"].map(
                    lambda path: mean_pd_size(self.neurons[path], neurite)
                )
            logger.info("Done computing dataset stats")

        return self._df_stats

    def make_node_count_plots(self, neurite: str) -> None:
        """Make neurite graph and PD count plots and save them to disk."""
        logger.info(f"Plotting node counts for {neurite}")

        # Plots - node count in neurites
        title = f"Number of nodes in the {neurite} graph"
        fig = plot_histogram(title, self.df_stats[f"{neurite}-size"])
        path = self.images_dir / f"n_nodes_{neurite}.png"
        self.template_vars[f"n_nodes_{neurite}"] = self.savefig(fig, path)

        # Plots - node count in neurites, per layer
        title = f"Number of nodes in the {neurite} graph per layer"
        data = {}
        for layer in set(self.df_stats["layer"]):
            rows = self.df_stats["layer"] == layer
            data[layer] = self.df_stats[rows][f"{neurite}-size"]
        fig = plot_histogram_panel(title, data)
        path = self.images_dir / f"n_nodes_per_layer_{neurite}.png"
        self.template_vars[f"n_nodes_per_layer_{neurite}"] = self.savefig(fig, path)

        # Plots - point count in PDs
        title = f"Number of points in the {neurite} persistence diagram"
        fig = plot_histogram(title, self.df_stats[f"{neurite}-pd-size"])
        path = self.images_dir / f"n_nodes_pd_{neurite}.png"
        self.template_vars[f"n_nodes_pd_{neurite}"] = self.savefig(fig, path)

        # Plots - point count in PDs, per layer
        title = f"Number of points in the {neurite} persistence diagram per layer"
        data = {}
        for layer in set(self.df_stats["layer"]):
            rows = self.df_stats["layer"] == layer
            data[layer] = self.df_stats[rows][f"{neurite}-pd-size"]
        fig = plot_histogram_panel(title, data)
        path = self.images_dir / f"n_nodes_pd_per_layer_{neurite}.png"
        self.template_vars[f"n_nodes_pd_per_layer_{neurite}"] = self.savefig(fig, path)

    def make_neuron_preview_plots(self) -> None:
        """Make neuron preview plots and save them to disk.

        For each m-type one sample is selected randomly and plotted. The
        plot contains two renders of the 3D morphology, one in the front
        view, and one in the side view, as well as a combined plot of a
        persistence diagram overlaid on top of a persistence image.
        """
        logger.info("Plotting example morphologies")
        # Get one sample per m-type
        df_subset = self.df_dataset.drop_duplicates("mtype").sort_values(by="mtype")
        image_paths = []
        for row in df_subset.itertuples():
            neuron = self.neurons[row.path]
            logger.info(f"Plotting morphology example for {neuron.name}")
            fig = plot_morphology_images(neuron, row.mtype, row.name)
            path = self.images_dir / f"morphology_example_{row.mtype}_{row.name}.png"
            image_paths.append(self.savefig(fig, path))
        self.template_vars["image_paths"] = image_paths

    def save_report(self, report_title: str = "Dataset Plots") -> None:
        """Save the HTML report to disk."""
        self.template_vars["report_title"] = report_title
        template = mc.report.plumbing.load_template("plot-dataset-stats")
        self.report_dir.mkdir(exist_ok=True, parents=True)
        mc.report.plumbing.render(template, self.template_vars, self.report_path)

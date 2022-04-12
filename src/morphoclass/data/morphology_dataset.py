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
"""Implementation of the MorphologyDataset class."""
from __future__ import annotations

import logging
import os
import pathlib
import shutil
import textwrap
from typing import Iterable

import neurom as nm
import numpy as np
import pandas as pd
from torch_geometric.data import Dataset

from morphoclass.data.morphology_data import MorphologyData
from morphoclass.data.morphology_embedding_dataset import MorphologyEmbeddingDataset

logger = logging.getLogger(__name__)


class MorphologyDataset(Dataset):
    """Dataset class for neuron morphologies.

    This class is derived from `torch_geometric.data.Dataset` and loads
    morphologies using the MorphIO library. The morphology data is integrated
    into the samples by setting the `MorphologyData.morphology` attribute.

    To extract features from the morphology data use the appropriate
    transformers in `morphoclass.transforms`.

    Parameters
    ----------
    data : iterable
        A sequence of instances of `Data`.
    transform
        The transformation to apply to every sample whenever
        this sample is retrieved from the dataset. Useful
        for example for data augmentation.
    pre_transform
        The transformation to apply to the data before it
        is stored in the dataset. This will happen only once
        per sample.
    pre_filter
        The filter to apply upon loading data into the
        dataset. It should be a function that takes objects
        of type `torch_geometric.data` and returns a boolean value.
    """

    def __init__(
        self,
        data,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        super().__init__(
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )
        self.data = list(data)
        if pre_filter is not None:
            self.data = [data for data in self.data if pre_filter(data)]
        if pre_transform is not None:
            self.data = [pre_transform(data) for data in self.data]

        labels = {data.label for data in self.data}
        if any(labels):
            self.label_to_y = {label: y for y, label in enumerate(sorted(labels))}
            self.y_to_label = {y: label for label, y in self.label_to_y.items()}
            for d in self.data:
                d.y = self.label_to_y[d.label]
        else:
            self.label_to_y = {}
            self.y_to_label = {}

    def get_sample_by_morph_name(self, morph_name):
        """Get morphology sample by name.

        Parameters
        ----------
        morph_name : str
            Morphology name.

        Returns
        -------
        sample : torch.torch_geometric.Data
            Sample instance.
        """
        for sample in self:
            if pathlib.Path(sample.path).stem == morph_name:
                return sample
        return None

    @property
    def ys(self):
        """Get the y-values of all samples.

        Returns
        -------
        list
            List of the y-values of all samples.
        """
        return [data.y for data in self]

    @property
    def labels(self):
        """Get label strings.

        Returns
        -------
        list
            List of labels' str.
        """
        return [data.label for data in self]

    def to_labels(self, ys):
        """Convert custom label ids to class.

        Parameters
        ----------
        ys : list
            List of label IDs.

        Returns
        -------
        list
            List of labels (as str name).
        """
        return [self.y_to_label[y] for y in ys]

    @classmethod
    def from_paths(
        cls,
        morph_paths,
        labels=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        """Load data from given file paths.

        This just calls the constructor serves as an alternative
        convenience method to it.

        Parameters
        ----------
        morph_paths : list_like
            A list of paths to morphology files.
        labels : list_like, optional
            A list of labels for the morphology files in `paths`.
        transform : callable, optional
            The transformation to apply to every sample whenever
            this sample is retrieved from the dataset. Useful
            for example for data augmentation.
        pre_transform : callable, optional
            The transformation to apply to the data before it
            is stored in the dataset. This will happen only once
            per sample.
        pre_filter : callable, optional
            The filter to apply upon loading data into the
            dataset. It should be a function that takes objects
            of type `torch_geometric.data` and returns a
            boolean.

        Returns
        -------
        morphoclass.data.MorphologyDataset
            An instance of `MorphologyDataset` with loaded data.
        """
        # Either same number of labels as morphologies, or no labels at all
        if labels and len(morph_paths) != len(labels):
            raise ValueError(
                "The number of labels does not match the number of "
                f"morphologies: {len(labels)} != {len(morph_paths)}"
            )

        # Load the data
        data = []
        for i, path in enumerate(morph_paths):
            morphology = nm.load_morphology(str(path))
            if not morphology:
                raise RuntimeError(f"Failed to load the morphology file {str(path)!r}")

            sample = MorphologyData(morphology=morphology, path=path)
            if labels:
                sample.y_str = labels[i]  # TODO: deprecate y_str, use "label" instead
                sample.label = labels[i]
            else:
                sample.label = None
            data.append(sample)

        return cls(
            data,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )

    @classmethod
    def from_features(cls, features: Iterable[dict]) -> MorphologyDataset:
        """Create a dataset from pre-extracted features.

        Parameters
        ----------
        features
            The pre-extracted features of the data. Each dictionary in
            the iterable corresponds to the features of one morphology.
            Given an existing `MorphologyDataset` the samples in which
            are instances of the class `torch_geometric.data.Data`, then
            the features can be obtained via `sample.to_dict()`.

        Returns
        -------
        MorphologyDataset
            A dataset instance constructed from pre-extracted data features.
        """
        data = []
        for feature_dict in features:
            sample = MorphologyData.from_dict(feature_dict)
            data.append(sample)

        return cls(data)

    @classmethod
    def from_structured_dir(
        cls,
        data_path,
        layer="",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        ignore_unknown_filetypes=True,
    ):
        """Load data from a structured directory.

        The directory `data_path` should have sub-directories, which,
        in turn, contain the morphologies. The names of these sub-
        directories will be interpreted as the morphology types and
        will be used as the class labels.

        Parameters
        ----------
        data_path : str or pathlib.Path
            Path pointing to the location of the data. The different
            morphology types should be organised in different
            sub-directories in `data_path`.
        layer : str
            The cortical layer for which the morphology data
            should be loaded. The corresponding m-type folders
            should start with the string specified in `layer`,
            e.g. for layer 5 the valid m-type folders are
            L5_TPC_A, L5_UPC, etc. If no layer is provided
            then all data is loaded.
        transform : callable, optional
            The transformation to apply to every sample whenever
            this sample is retrieved from the dataset. Useful
            for example for data augmentation.
        pre_transform : callable, optional
            The transformation to apply to the data before it
            is stored in the dataset. This will happen only once
            per sample.
        pre_filter : callable, optional
            The filter to apply upon loading data into the
            dataset. It should be a function that takes objects
            of type `torch_geometric.data` and returns a
            boolean.

        Returns
        -------
        morphoclass.data.MorphologyDataset
            An instance of `MorphologyDataset` with loaded data.
        """
        data_path = pathlib.Path(data_path)
        if not data_path.is_dir():
            raise ValueError(f"{data_path} is not a valid directory.")

        morph_paths = []
        labels = []
        known_extensions = {".swc.", ".asc", ".h5"}
        for directory_layer in sorted(data_path.iterdir()):
            if directory_layer.name != layer or not directory_layer.is_dir():
                continue
            for directory_layer_subclass in sorted(directory_layer.iterdir()):
                if not directory_layer_subclass.is_dir():
                    continue
                label = f"{directory_layer.name}/{directory_layer_subclass.name}"
                new_morph_paths = []
                for path in directory_layer_subclass.iterdir():
                    if (
                        ignore_unknown_filetypes
                        and path.suffix.lower() not in known_extensions
                    ):
                        logger.warning(f"Ignored a non-morphology file: {str(path)!r}")
                        continue
                    new_morph_paths.append(path)
                if not new_morph_paths:
                    continue
                new_morph_paths = sorted(new_morph_paths)
                morph_paths += new_morph_paths
                labels += [label] * len(new_morph_paths)

        return cls.from_paths(
            morph_paths,
            labels,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )

    @classmethod
    def from_csv(cls, csv_file, transform=None, pre_transform=None, pre_filter=None):
        """Load data listed in a CSV file.

        The CSV file should have no header. The first column should
        list paths to morphology files. The second optional column
        should contain labels.

        The paths can be either absolute or relative. The relative paths
        should be relative to the directory with the CSV file.

        Parameters
        ----------
        csv_file : str or pathlib.Path
            The CSV file with data paths and labels
        transform : callable or None
            The transformation to apply to every sample whenever
            this sample is retrieved from the dataset. Useful
            for example for data augmentation.
        pre_transform : callable or None
            The transformation to apply to the data before it
            is stored in the dataset. This will happen only once
            per sample.
        pre_filter : callable or None
            The filter to apply upon loading data into the
            dataset. It should be a function that takes objects
            of type `torch_geometric.data` and returns a
            boolean.

        Returns
        -------
        morphoclass.data.MorphologyDataset
            An instance of `MorphologyDataset` with loaded data.
        """
        csv_file = pathlib.Path(csv_file)
        if not csv_file.is_file():
            raise ValueError(f"{csv_file} is not a valid file.")

        df_data = pd.read_csv(csv_file, header=None)

        if len(df_data.columns) == 0:
            raise ValueError("The CSV file contains no data.")
        elif len(df_data.columns) == 1:
            morph_paths = df_data.iloc[:, 0].tolist()
            labels = None
        else:
            morph_paths = df_data.iloc[:, 0].tolist()
            labels = df_data.iloc[:, 1].tolist()

        # Resolve relative paths
        csv_dir = csv_file.parent
        for i in range(len(morph_paths)):
            if not os.path.isabs(morph_paths[i]):
                morph_paths[i] = str(csv_dir / morph_paths[i])

        return cls.from_paths(
            morph_paths,
            labels,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )

    def to_csv(self, path, write_labels=True):
        """Write data paths and labels to a CSV file.

        Parameters
        ----------
        path : str or pathlib.Path
            The target CSV file.
        write_labels : bool
            If True labels will be included in a separate column.
        """
        rows = []
        for sample in self:
            row = {"path": sample.path}
            if write_labels:
                row["label"] = self.class_dict.get(sample.y)
            rows.append(row)
        df_data = pd.DataFrame(rows)
        df_data.to_csv(path, index=False, header=False)

    def len(self):
        """Compute the length of the dataset.

        Returns
        -------
        int
            The number of samples in the dataset.
        """
        return len(self.data)

    def get(self, idx):
        """Get one data sample.

        Parameters
        ----------
        idx : int
            The index of the data sample.

        Returns
        -------
        torch_geometric.data.Data
            The data sample corresponding to `idx`.
        """
        return self.data[idx]

    def save_data(self, output_dir, indices=None):
        """Save the dataset or a subset of it to disk.

        Note that the saving is done by copying the original files
        from which the data was read to a new location in the
        `output_dir`, so the original files must be accessible.

        Note also that since it is just a copy of the original files
        no transforms or pre-transforms assigned to this dataset
        are applied to the data.

        It is important that any pre-transforms assigned to this dataset
        keep the `file` attribute of the samples intact since this is
        how the original morphology file is located.

        Parameters
        ----------
        output_dir : str or pathlib.Path
            The output directory.
        indices : list_like, optional
            The indices of the samples to be saved. If None
            then all samples will be saved.
        """
        output_dir = pathlib.Path(output_dir)
        if output_dir.exists() and any(output_dir.iterdir()):
            error_msg = """
            The output directory already exists and is non-empty. Please
            delete its contents or choose a different output directory,
            then try again."
            """
            raise ValueError(textwrap.dedent(error_msg).strip())
        if indices is None:
            indices = range(len(self))
        for idx in indices:
            original_file = self.get(idx).path
            relative_file = original_file.relative_to(original_file.parent.parent)
            output_file = output_dir / relative_file
            output_file.parent.mkdir(exist_ok=True, parents=True)
            shutil.copy(original_file, output_file)

    def guess_layer(self):
        """Guess the layer based on the labels.

        M-types of the form "L5_TPC_A" contain the layer
        information in the second character of the m-type string.
        If all labels contain the same number in that position
        then this is the guessed layer. Otherwise None is returned.

        Returns
        -------
        int or None
            The guessed layer number or None in case the guess
            was unsuccessful.
        """
        all_m_types = self.class_dict.values()
        all_layers = {m_type[1] for m_type in all_m_types}
        if len(all_layers) == 1:
            layer = all_layers.pop()
            if layer.isnumeric():
                return int(layer)
            else:
                return None
        else:
            return None

    def to_lowerdim_dataset(self, embedding_type, **feat_extr_kwargs):
        """Create a lower-dimensional dataset from MorphologyDataset.

        Parameters
        ----------
        embedding_type : str
            The type of the embedding: tmd, deepwalk.
        feat_extr_kwargs : dict
            Set of parameters for embedding.

        Returns
        -------
        dataset : morphoclass.data.MorphologyEmbeddingDataset
            Dataset with morphology embedding (lower-dimensional
            morphology representation).
        """
        # The embedding of multiple neurites is simply all points from all
        # neurite embeddings put together. This mirrors the way it's done in
        # tmd.Topology.methods.get_ph_neuron
        embeddings = []
        if embedding_type == "deepwalk":
            from morphoclass import deepwalk

            deepwalk.warn_if_not_installed()

            for data in self:
                points = []
                for tree in data.tmd_neurites:
                    points.append(deepwalk.get_embedding(tree, **feat_extr_kwargs))
                embeddings.append(np.concatenate(points))
        elif embedding_type == "tmd":
            from tmd.Topology.methods import get_persistence_diagram

            for data in self:
                diagram = []
                for tree in data.tmd_neurites:
                    diagram.extend(get_persistence_diagram(tree, **feat_extr_kwargs))
                embeddings.append(diagram)
        else:
            raise ValueError(f"Value {embedding_type} not supported as embedding.")

        # Check that the embeddings are big enough - at least 3 points.
        # Otherwise, we can't generate any persistence images and such data
        # doesn't make sense anyway.
        # TODO: is this the right place to do this? Also see a similar check
        #       in the feature_extractors.
        idx_keep = []
        for i, embedding in enumerate(embeddings):
            if len(embedding) < 3:
                logger.warning(
                    f"The embedding of {self[i].path} has fewer than 3 points. "
                    f"We'll remove this morphology from the dataset."
                )
            else:
                idx_keep.append(i)

        data = [self[i] for i in idx_keep]
        embeddings = [np.array(embeddings[i]) for i in idx_keep]

        dataset = MorphologyEmbeddingDataset.from_embedding(
            embeddings=[embeddings[i] for i in idx_keep],
            labels=[d.y_str for d in data],
            morph_paths=[d.path for d in data],
            morph_names=[pathlib.Path(d.path).stem for d in data],
            morphologies=data,
        )

        return dataset

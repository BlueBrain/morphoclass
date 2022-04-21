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
"""Implementation of the MorphologyEmbedding class."""
from __future__ import annotations

import pathlib
import pickle

import numpy as np
from tmd.Topology.analysis import get_persistence_image_data
from torch_geometric.data import Data
from torch_geometric.data import Dataset


class MorphologyEmbedding(Data):
    """Data with low-dimensional morphology representation.

    This class creates instances that store embeddings from TMD (2D) or
    DeepWalk (2D+) embeddings. If the embedding is 2D, it is converted to the
    image using Gaussian kernels.

    Parameters
    ----------
    embedding : np.ndarray
        The list of embedding points.
    morph_path : str, pathlib.Path, optional
        Path to original morphology.
    morph_name : str
        Morphology name.
    y : int or None
        Integer class label.
    y_str : str or None
        String class label.
    file : str, pathlib.Path or None
        Path to the file embedding.
    scale : flaot, int, default 1
        Scale the embedding.
    """

    def __init__(
        self,
        embedding,
        morphology=None,
        morph_path=None,
        morph_name=None,
        y=None,
        y_str=None,
        file=None,
        scale=1,
    ):
        self._file: pathlib.Path | None = None
        self._morph_path: pathlib.Path | None = None
        self._scale: float = 1

        self.path = file
        self.embedding = embedding
        self.morphology = morphology
        self.morph_path = morph_path
        self.morph_name = morph_name
        self.image = self.diagram2image()
        self.y = y
        self.y_str = y_str
        self.scale = scale

        super().__init__()

    @property
    def file(self):
        """Get file path to the embedding.

        Returns
        -------
        str or pathlib.Path
            File path to the embedding.
        """
        return self._file

    @file.setter
    def file(self, val):
        """Set file location of the embedding.

        Parameters
        ----------
        val : str or pathlib.Path
            File path to the embedding.
        """
        if val is not None:
            self._file = pathlib.Path(val)
        else:
            self._file = None

    @property
    def morph_path(self):
        """Get file path to the original morphology.

        Returns
        -------
        str or pathlib.Path
            File path to the original morphology.
        """
        return self._morph_path

    @morph_path.setter
    def morph_path(self, val):
        """Set file location of the original morphology.

        Parameters
        ----------
        val : str or pathlib.Path
            File path to the original morphology.
        """
        if val is not None:
            self._morph_path = pathlib.Path(val)
        else:
            self._morph_path = None

    @property
    def scale(self):
        """Get the embedding scale factor.

        Returns
        -------
        The embedding's scale factor.
        """
        return float(self._scale)

    @scale.setter
    def scale(self, val):
        """Set the embedding scale factor.

        Parameters
        ----------
        val : float, int
            The new scale factor.
        """
        self._scale = float(val)
        self.embedding = self.embedding / self._scale

    def diagram2image(self):
        """Convert 2D embedding to 2D image.

        Returns
        -------
        image : np.ndarray or None
            The 2D image created using Gaussian kernels on the 2D embedding.
            If embedding is not 2D, the image will be None.
        """
        if self.embedding.shape[1] != 2:
            return None

        xmax, ymax = self.embedding.max(axis=0)
        xmin, ymin = self.embedding.min(axis=0)

        # If all values are positive then we'll set xmin = ymin = 0, which
        # seems more natural. In particular this is true if the filtration
        # function is positive definite, e.g. radial distances. Note that
        # this step is not done in the original TMD package.
        xmin = min(xmin, 0)
        ymin = min(ymin, 0)

        image = get_persistence_image_data(
            self.embedding,
            xlims=(xmin, xmax),
            ylims=(ymin, ymax),
        )
        image = np.rot90(image)
        image = image[np.newaxis, np.newaxis, :, :]

        return image


class MorphologyEmbeddingDataset(Dataset):
    """Dataset with lower-dimensional morphology representation.

    This class is derived from `torch_geometric.data.Dataset` and loads
    morphologies' embeddings from `pickle` files.

    Parameters
    ----------
    paths : iterable
        A list of paths to morphology files.
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
        of type `torch_geometric.data` and returns a
        boolean.
    """

    def __init__(
        self,
        data,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.data = data

        labels = [sample.y_str for sample in self.data]
        self.labels = list(labels)
        unique_labels = set(self.labels)
        self.y_to_label = dict(enumerate(sorted(unique_labels)))

        # normalize embeddings & update data with id labels
        scale = max(sample.embedding.max() for sample in self.data)
        for sample in self.data:
            sample.scale = scale
            sample.y = self.label_to_y[sample.y_str]

        super().__init__(
            root="",
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )

    @property
    def y_to_label(self):
        """Get mapper from id to label str.

        Example of a dictionary item: 0 -> 'L5_TPC_A'.

        Returns
        -------
        dict
            Mapper from id to label str.
        """
        return self._id2label_mapping

    @y_to_label.setter
    def y_to_label(self, new_id2label_mapping):
        """Set new mapper from id to label str.

        Example of a dictionary item: 0 -> 'L5_TPC_A'.
        The dictionary `label2id_mapping` also needs to be modified accordingly.

        Parameters
        ----------
        new_id2label_mapping : dict
            New mapper to use.

        Raises
        ------
        RuntimeError
            If not all labels are mapped to their corresponding id value.
        """
        if (
            len(self.labels) > 0
            and self.labels[0] is not None
            and not all(label in new_id2label_mapping.values() for label in self.labels)
        ):
            raise RuntimeError("Make sure the mapper has a mapping for all the labels!")

        self._id2label_mapping = dict(new_id2label_mapping)
        self._label2id_mapping = {
            label: i for i, label in sorted(self._id2label_mapping.items())
        }

    @property
    def label_to_y(self):
        """Get mapper from label str to id.

        Example of a dictionary item: 'L5_TPC_A' -> 0.

        Returns
        -------
        dict
            Mapper from label str to id.
        """
        return self._label2id_mapping

    @label_to_y.setter
    def label_to_y(self, new_label2id_mapping):
        """Set new mapper from label str to id.

        Example of a dictionary item: 'L5_TPC_A' -> 0.
        The dictionary `id2label_mapping` also needs to be modified accordingly.

        Parameters
        ----------
        new_label2id_mapping : dict
            New mapper to use.

        Raises
        ------
        RuntimeError
            If not all labels are mapped to their corresponding id value.
        """
        if (
            len(self.labels) > 0
            and self.labels[0] is not None
            and not all(label in new_label2id_mapping for label in self.labels)
        ):
            raise RuntimeError("Make sure the mapper has a mapping for all the labels!")

        self._label2id_mapping = dict(new_label2id_mapping)
        self._id2label_mapping = {
            label: i for i, label in sorted(self._label2id_mapping.items())
        }

    @property
    def ys(self):
        """Get label ids.

        Returns
        -------
        list
            List of labels' ids.
        """
        return [sample.y for sample in self]

    @property
    def labels_str(self):
        """Get label strings.

        Returns
        -------
        list
            List of labels' str.
        """
        return [sample.y_str for sample in self]

    @property
    def morph_paths(self):
        """Get paths to morphology files.

        Returns
        -------
        list
            List of paths to the morphologies.
        """
        self._morph_paths = [sample.morph_path for sample in self.data]
        return self._morph_paths

    @property
    def morph_names(self):
        """Get paths to morphology files.

        Returns
        -------
        list
            List of paths to the morphologies.
        """
        self._morph_names = [sample.morph_name for sample in self.data]
        return self._morph_names

    def id2label(self, label_ids):
        """Convert custom label ids to class.

        Parameters
        ----------
        label_ids : list
            List of label IDs.

        Returns
        -------
        list
            List of labels (as str name).
        """
        return [self.y_to_label[lid] for lid in label_ids]

    def len(self):
        """Compute the length of the dataset.

        Returns
        -------
        int
            The number of samples in the dataset.
        """
        return len(self.data)

    def _download(self):
        # Data is already local, don't need to download
        pass

    def _process(self):
        # We won't be saving data to disc, do just do the processing part
        self.process()

    @property
    def raw_file_names(self):
        """Not used.

        Returns
        -------
        list
            Empty list.
        """
        return []

    @property
    def processed_file_names(self):
        """Not used.

        Returns
        -------
        list
            Empty list.
        """
        return []

    def download(self):
        """Not used."""
        pass

    def get(self, idx):
        """Get one data sample.

        Parameters
        ----------
        idx : int
            The index of the data sample.

        Returns
        -------
        data.morphology_dataset.MorphologyEmbedding
            The data sample corresponding to `idx`.
        """
        return self.data[idx]

    @classmethod
    def from_paths(
        cls,
        paths,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        """Load data from given file paths.

        This just calls the constructor serves as an alternative
        convenience method to it.

        Parameters
        ----------
        paths : list_like
            A list of paths to morphology embedding files.
        file_name_suffix :
            A file name suffix that is appended to the morphology embedding file.
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
        morphoclass.data.MorphologyEmbeddingDataset
            An instance of `MorphologyEmbeddingDataset` with loaded data.
        """
        data = []

        for path in paths:
            with pathlib.Path(path).open("rb") as fp:
                datum = pickle.load(fp)
            data.append(datum)

        return cls(
            data=data,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )

    @classmethod
    def from_embedding(
        cls,
        embeddings,
        morph_paths=None,
        morph_names=None,
        morphologies=None,
        paths=None,
        labels=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        """Create dataset from embeddings and paths.

        Parameters
        ----------
        embeddings : list
            List of embeddings.
        morph_paths : list
            List of paths to original morphologies.
        morph_names : list
            List of morphology names.
        paths : list, optional
            List of paths to store the embedding.
        labels : list, optional
            List of labels corresponding to embeddings (label str).
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
        """
        data = []

        if labels is None:
            labels = [None] * len(embeddings)
        if paths is None:
            paths = [None] * len(embeddings)
        if morph_paths is None:
            morph_paths = [None] * len(embeddings)
        if morphologies is None:
            morphologies = [None] * len(embeddings)

        for embedding, label, path, morph_path, morph_name, morphology in zip(
            embeddings, labels, paths, morph_paths, morph_names, morphologies
        ):
            sample = MorphologyEmbedding(
                embedding=embedding,
                morphology=morphology,
                morph_path=morph_path,
                morph_name=morph_name,
                file=path,
                y_str=label,
            )
            data.append(sample)

        return cls(
            data=data,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )

    def to_paths(self, paths, overwrite=False):
        """Store embedded samples into disk.

        Parameters
        ----------
        paths : list
            List of paths to store the embedded points.
        overwrite : bool, False
            To overwrite or not the existing file.

        Raises
        ------
        RuntimeError
            If number of paths and samples doesn't match.
        FileExistsError
            If file with dataset exists.
        """
        if len(paths) != len(self.data):
            raise RuntimeError("The number of paths and samples should be the same!")

        for path, sample in zip(paths, self.data):
            if not path.exists() or overwrite:
                sample.path = path
                with open(sample.path, "wb") as fp:
                    pickle.dump(sample, fp)
            else:
                raise FileExistsError(
                    f"Path {path} exists."
                    "Change overwrite to True if you want to overwrite."
                )

    def process(self):
        """Load data from disc, apply filters and transforms."""
        if self.pre_filter is not None:
            self.data = [data for data in self.data if self.pre_filter(data)]

        if self.pre_transform is not None:
            self.data = [self.pre_transform(data) for data in self.data]

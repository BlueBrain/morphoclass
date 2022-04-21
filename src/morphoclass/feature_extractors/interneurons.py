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
"""Morphology preprocessing and feature extractor."""
from __future__ import annotations

import logging
from typing import Any

from torch.utils.data import DataLoader

from morphoclass.data import MorphologyDataset
from morphoclass.data.morphology_data_loader import MorphologyDataLoader
from morphoclass.data.morphology_embedding_data_loader import (
    MorphologyEmbeddingDataLoader,
)

# from morphoclass.data.morphology_embedding_dataset import MorphologyEmbeddingDataset
from morphoclass.training import make_transform
from morphoclass.transforms import BranchingOnlyNeurites
from morphoclass.transforms import Compose
from morphoclass.transforms import ExtractEdgeIndex
from morphoclass.transforms import ExtractRadialDistances
from morphoclass.transforms import ExtractTMDNeurites

logger = logging.getLogger(__name__)


def feature_extractor(
    file_name_suffix,
    input_csv,
    embedding_type,
    neurite_type=None,
    overwrite=False,
):
    """Prepare morphology dataset and extract features.

    Parameters
    ----------
    file_name_suffix : str
        The suffix for the extracted features file that will be saved to the
        same location as the original file.
    input_csv : str or pathlib.Path
        The path to the file with mappings: h5 cleaned morphology file to class label.
    embedding_type : str
        Three possible options: morphology, tmd, deepwalk.
    neurite_type
        The neurite type to extract. One of "axon", "apical", "basal",
        "neurites". The latter will extract all neurites.
    overwrite
        Whether to overwrite the pickle files with TMD/deepwalk features if
        they already exist.

    Returns
    -------
    dataset : morphoclass.data.MorphologyDataset
        The instance of dataset that will be used in the model.
    loader_cls : DataLoader
        The loader class for the dataset that will be used in the trainer.
    results : dict
        Dictionary with some main stats from the feature extractor.

    Raises
    ------
    ValueError
        If the embedding type doesn't exist.
    """
    if embedding_type not in {"morphology", "tmd", "deepwalk"}:
        raise ValueError(f"Embedding type {embedding_type} doesn't exist!")
    if neurite_type is None:
        neurite_type = "axon"
        logger.info(f"Using the default neurite type: {neurite_type}")
    else:
        logger.info(f"Using a custom neurite type: {neurite_type}")

    feature_extractor_ = ExtractRadialDistances()
    pre_transform = Compose(
        [
            ExtractTMDNeurites(neurite_type=neurite_type),
            BranchingOnlyNeurites(),
            ExtractEdgeIndex(),
        ]
    )

    logger.info("✔ Loading data...")
    dataset = MorphologyDataset.from_csv(input_csv, pre_transform=pre_transform)

    logger.info(f"✔ Dataset size... {len(dataset)}")
    logger.info("> Dataset classes:")
    for cls_name in sorted(dataset.y_to_label.values()):
        logger.info(f" - {cls_name}")

    transform, fitted_scaler = make_transform(
        dataset=dataset,
        feature_extractor=feature_extractor_,
        n_features=1,
    )
    dataset.transform = transform

    # Remove the samples with fewer than 3 nodes in the neurite graph.
    # Otherwise, there will be problems with the generation of persistence
    # images, and such small graphs don't make sense anyway.
    # TODO: should this check be done elsewhere? Problem is that the call of
    #       `to_lowerdim_dataset` further below fails if the embeddings are
    #       too small, and we're doing another check within that method.
    idx_keep = []
    for i in range(len(dataset)):
        if len(dataset[i].x) < 3:
            logger.warning(
                f"Morphology {dataset[i].path} gave fewer than 3 features. "
                f"We'll remove this morphology from the dataset."
            )
        else:
            idx_keep.append(i)
    dataset = dataset.index_select(idx_keep)

    feat_extractor_kwargs: dict[str, Any] = {}
    loader_cls: type[DataLoader]
    if embedding_type == "morphology":
        loader_cls = MorphologyDataLoader
    else:
        loader_cls = MorphologyEmbeddingDataLoader

        if embedding_type == "tmd":
            feat_extractor_kwargs = {"feature": "radial_distances"}
        elif embedding_type == "deepwalk":
            feat_extractor_kwargs = {
                "representation_size": 2,
                "max_memory_data_size": 1000000000,
                "walk_length": 20,
                "number_walks": 10,
                "seed": 0,
                "undirected": True,
                "window_size": 5,
                "workers": 1,
            }
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type!r}")

        # paths = [
        #     path.with_name(f"{path.stem}_{file_name_suffix}.pickle")
        #     for path in dataset.morph_paths
        # ]
        #
        # if not all(path.exists() for path in paths) or overwrite:
        #     dataset = dataset.to_lowerdim_dataset(
        #         embedding_type,
        #         **feat_extractor_kwargs,
        #     )
        #     logger.info("✔ Store dataset...")
        #     dataset.to_paths(paths, overwrite=True)
        #
        # logger.info("✔ Read dataset...")
        # dataset = MorphologyEmbeddingDataset.from_paths(paths)
        # TODO: this replaces the commented lines above. They were meant to
        #       cache the extracted features (persistence images etc.), but
        #       it this doesn't work with different dataset CSVs that share
        #       the same data. Reason: code above pickles the `Data` objects,
        #       which also contains the paths and labels from the dataset
        #       being processed. These values are then used with a different
        #       dataset and the paths and labels from the CSV are ignored.
        dataset = dataset.to_lowerdim_dataset(embedding_type, **feat_extractor_kwargs)

    results = {
        "fitted_scaler": fitted_scaler.get_config() if fitted_scaler else None,
        "class_dict": dataset.y_to_label,
        **feat_extractor_kwargs,
    }

    return dataset, loader_cls, results

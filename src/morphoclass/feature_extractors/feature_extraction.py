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
"""Implementation of the `morphoclass feature-extraction` CLI command."""
from __future__ import annotations

import logging
import pathlib
from typing import Callable

from torch.utils.data import DataLoader

from morphoclass.constants import DatasetType
from morphoclass.data import MorphologyDataset

logger = logging.getLogger(__name__)


def feature_extraction_method(
    input_csv,
    feature_extractor_name,
    dataset_name,
    overwrite=False,
):
    """Extract the features from dataset.

    Parameters
    ----------
    input_csv : str or pathlib.Path
        Input csv path.
    feature_extractor_name : str
        Name of the feature extractor.
    dataset_name : str
        Dataset name.
    overwrite
        Whether or not to overwrite the pickle files with TMD/deepwalk
        features if they already exist.

    Returns
    -------
    dataset : morphoclass.data.MorphologyEmbeddingDataset,
              morphoclass.data.Morphology
        The instance of dataset that will be used in the model.
    results : dict
        Dictionary with some main stats from the feature extractor.
    loader_cls : morphoclass.data.MorphologyEmbeddingDataLoader,
                 morphoclass.data.MorphologyDataLoader
        The loader for the dataset that will be used in the trainer.

    Raises
    ------
    ValueError
        When feature extractor is not implemented for a specified dataset.
    """
    input_csv = pathlib.Path(input_csv).resolve()

    logger.info(f"Dataset           : {dataset_name}")
    logger.info(f"Input CSV         : {input_csv}")
    logger.info(f"Feature extractor : {feature_extractor_name}")

    logger.info("✔ Feature extraction...")

    # import feature_extractor
    feature_extractor: Callable[
        [str, str, str, bool], tuple[MorphologyDataset, DataLoader, dict]
    ]
    if DatasetType(dataset_name) == DatasetType.pyramidal:
        from morphoclass.feature_extractors import pyramidal_cells

        feature_extractor = pyramidal_cells.feature_extractor
    elif DatasetType(dataset_name) == DatasetType.interneurons:
        from morphoclass.feature_extractors import interneurons

        feature_extractor = interneurons.feature_extractor
    else:
        raise ValueError(
            f"Feature extractor for dataset {dataset_name} is not implemented."
        )

    dataset, loader_cls, results = feature_extractor(
        file_name_suffix=feature_extractor_name,
        input_csv=input_csv,
        embedding_type=feature_extractor_name,
        overwrite=overwrite,
    )

    logger.info("✔ Done.")

    return dataset, loader_cls, results

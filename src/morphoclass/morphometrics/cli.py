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
"""Utilities for the `morphoclass moprhometrics` command."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from morphoclass.training._helpers import reset_seeds
from morphoclass.training.cli import get_model
from morphoclass.training.training_config import TrainingConfig
from morphoclass.training.training_log import TrainingLog
from morphoclass.utils import make_torch_deterministic

logger = logging.getLogger(__name__)


# TODO: The functions of this package simply adapt the content of what is found in
#  morphoclass.training.cli. We had to re-implement those functionalities because
#  here we are working with morphometrics,
#  so we cannot have a morphoclass.data.MorphologyDataset.
#  It would however be nice to merge these functionalities together, by creating
#  more generic functions that are independent from the type of Dataset being used.


def run_training(
    df_morphometrics: pd.DataFrame, training_config: TrainingConfig
) -> TrainingLog:
    """Training and evaluation of the model."""
    make_torch_deterministic()

    logger.info("Run training and evaluation.")

    x_features = df_morphometrics.drop(columns=["property|name", "filepath", "m_type"])
    logger.debug(f"- Input features : {x_features.columns.to_list()}")
    x_features = x_features.to_numpy()

    # Convert to numeric labels: "UPC", "TPC-A", "TPC-B", ... -> 2, 0, 1, ...
    label_enc = LabelEncoder()
    label_enc.fit(df_morphometrics["m_type"])
    y_labels = label_enc.transform(df_morphometrics["m_type"])  # 2, 0, 1, 1, 0, ...
    labels_unique_str = label_enc.classes_  # "TPC-A", "TPC-B", "UPC"
    logger.debug(f"- Classes        : {labels_unique_str}")

    n_samples = len(x_features)
    n_classes = len(np.unique(y_labels))
    logger.debug(f"- N. of samples  : {n_samples}")
    logger.debug(f"- N. of classes  : {n_classes}")

    if training_config.model_class.startswith("xgboost"):
        training_config.model_params["num_class"] = n_classes
        training_config.model_params["use_label_encoder"] = False  # suppress warning
    elif training_config.model_class.startswith("morphoclass"):
        training_config.model_params["n_classes"] = n_classes
    logger.debug(f"- Model class    : {training_config.model_class}")

    if training_config.seed is not None:
        reset_seeds(numpy_seed=training_config.seed, torch_seed=training_config.seed)

    splitter = training_config.splitter_cls(**training_config.splitter_params)
    split = splitter.split(X=x_features, y=y_labels)
    n_splits = splitter.get_n_splits(X=x_features, y=y_labels)
    logger.debug(f"- Splitter class  : {training_config.splitter_class}")
    logger.debug(f"- N of splits     : {n_splits}")

    probabilities = np.empty((n_samples, n_classes))
    predictions = np.empty(n_samples, dtype=int)

    training_log = TrainingLog(config=training_config, labels_str=labels_unique_str)
    # SPLIT MODEL
    for n, (train_idx, val_idx) in enumerate(split):
        logger.info(f"Split {n + 1}/{n_splits}, ratio: {len(train_idx)}:{len(val_idx)}")
        history = train_ml_model(
            training_config, train_idx, val_idx, x_features, y_labels
        )
        history["ground_truths"] = y_labels[val_idx]
        history["train_idx"] = list(train_idx)
        history["valid_idx"] = list(val_idx)

        y_labels[val_idx] = history["ground_truths"]
        predictions[val_idx] = history["predictions"]
        probabilities[val_idx] = np.array(history["probabilities"])

        # Save split history
        training_log.add_split(history)

    # collect results
    training_log.set_y(y_labels, predictions, probabilities)

    # MODEL ALL
    if training_config.train_all_samples:
        logger.info("Fit model on all samples")
        train_idx = np.arange(len(y_labels))

        history = train_ml_model(
            training_config, train_idx, val_idx, x_features, y_labels
        )
        training_log.set_all_history(history)

    return training_log


def train_ml_model(
    config: TrainingConfig,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    x_features: np.ndarray,
    y_labels: np.ndarray,
) -> dict[str, Any]:
    """Train a sklearn-like model."""
    if not config.model_class.startswith(
        "sklearn"
    ) and not config.model_class.startswith("xgboost"):
        raise ValueError(f"Training model of type {config.model_class} not supported.")

    n_classes = len(np.unique(y_labels))
    model = get_model(config, None, n_classes)

    x_train, y_train = x_features[train_idx], y_labels[train_idx]
    x_valid, y_valid = x_features[valid_idx], y_labels[valid_idx]

    model.fit(x_train, y_train)

    latent_features = x_features

    probabilities = model.predict_proba(x_valid)
    labels_val_pred = model.predict(x_valid)

    history = {
        "model": model,
        "predictions": labels_val_pred,
        "ground_truths": y_valid,
        "probabilities": probabilities,
        "latent_features": latent_features,
    }

    return history

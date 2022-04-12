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
"""Implementation of the training log."""
from __future__ import annotations

import dataclasses
import pathlib
from typing import Sequence

import numpy as np
import torch

from morphoclass.training.training_config import TrainingConfig
from morphoclass.utils import add_metadata


class TrainingLog:
    """A training log.

    At the moment this mirrors the dictionary log in the training entry point.
    The structure is suboptimal and should be refactored at a later time.
    """

    def __init__(self, config: TrainingConfig, labels_str: Sequence[str]):
        self.config: TrainingConfig = config
        self.labels_str: Sequence[str] = labels_str
        self.pretraining_log_: TrainingLog | None = None
        self.split_history: list[dict] = []
        self.all_history: dict | None = None
        self.targets: np.ndarray | None = None
        self.preds: np.ndarray | None = None
        self.probas: np.ndarray | None = None
        self.metrics: dict = {}
        self.split_metrics: dict = {}
        self.cleanlab_errors: np.ndarray | None = None
        self.cleanlab_self_confidence: np.ndarray | None = None
        self.interactive: bool = False
        self.metadata: dict = self.get_env_metadata()

    @staticmethod
    def get_env_metadata() -> dict:
        """Get the morphoclass environment metadata."""
        meta: dict[str, dict] = {}
        add_metadata(meta)
        return meta["metadata"]

    @property
    def pretraining_log(self) -> TrainingLog | None:
        """Get the training log of the pretrained model."""
        if self.config.checkpoint_path_pretrained is None:
            return None
        if self.pretraining_log_ is None:
            self.pretraining_log_ = TrainingLog.load(
                self.config.checkpoint_path_pretrained
            )

        return self.pretraining_log_

    def add_split(self, history: dict) -> None:
        """Add the history of a split training to the training log."""
        self.split_history.append(history)

    def set_all_history(self, history):
        """Add the history of the training on all data to the training log."""
        self.all_history = history

    def set_y(self, targets, predictions, probabilities):
        """Set the targets, predictions, and the probabilities."""
        self.targets = targets
        self.preds = predictions
        self.probas = probabilities

    def set_metrics(self, metrics, split_metrics):
        """Set the metric."""
        self.metrics = metrics
        self.split_metrics = split_metrics

    def set_cleanlab_stats(self, errors, self_confidence):
        """Set the cleanlab evaluation data."""
        self.cleanlab_errors = errors
        self.cleanlab_self_confidence = self_confidence

    def to_dict(self) -> dict:
        """Convert the training log to a dictionary."""
        if self.metrics is None:
            raise ValueError('The "metrics" attribute is missing, set it first.')
        if self.split_metrics is None:
            raise ValueError('The "split_metrics" attribute is missing, set it first.')
        data = {
            "splits": self.split_history,
            "all": self.all_history,
            "ground_truths": self.targets,
            "predictions": self.preds,
            "probabilities": self.probas,
            **self.metrics,
            **self.split_metrics,
            "cleanlab_ordered_label_errors": self.cleanlab_errors,
            "cleanlab_self_confidence": self.cleanlab_self_confidence,
            **dataclasses.asdict(self.config),
            "labels_unique_str": self.labels_str,
            "model_params": self.config.model_params,
            "oversampling": self.config.oversampling,
            "skip_user_input": not self.interactive,
            "metadata": self.metadata,
        }
        return data

    @classmethod
    def from_dict(cls, data: dict) -> TrainingLog:
        """Reconstruct the training log from a dictionary."""
        config = TrainingConfig(
            input_csv=data.get("input_csv"),
            model_class=data["model_class"],
            model_params=data["model_params"],
            splitter_class=data["splitter_class"],
            splitter_params=data["splitter_params"],
            dataset_name=data["dataset_name"],
            feature_extractor_name=data["feature_extractor_name"],
            optimizer_class=data["optimizer_class"],
            optimizer_params=data["optimizer_params"],
            n_epochs=data["n_epochs"],
            batch_size=data["batch_size"],
            seed=data["seed"],
            oversampling=data["oversampling"],
            train_all_samples=data["train_all_samples"],
            checkpoint_path_pretrained=data["checkpoint_path_pretrained"],
            frozen_backbone=data["frozen_backbone"],
        )
        training_log = cls(config=config, labels_str=data["labels_unique_str"])
        for split_history in data["splits"]:
            training_log.add_split(split_history)
        training_log.all_history = data["all"]
        training_log.set_y(
            data["ground_truths"],
            data["predictions"],
            data["probabilities"],
        )
        training_log.set_cleanlab_stats(
            data["cleanlab_ordered_label_errors"],
            data["cleanlab_self_confidence"],
        )
        training_log.interactive = not data["skip_user_input"]
        metrics_dict = {
            "classification_report": data["classification_report"],
            "confusion_matrix": data["confusion_matrix"],
            "accuracy": data["accuracy"],
            "f1_micro": data["f1_micro"],
            "f1_macro": data["f1_macro"],
            "f1_weighted": data["f1_weighted"],
        }
        split_metrics_dict = {
            "accuracy_mean": data["accuracy_mean"],
            "accuracy_std": data["accuracy_std"],
            "f1_micro_mean": data["f1_micro_mean"],
            "f1_micro_std": data["f1_micro_std"],
            "f1_macro_mean": data["f1_macro_mean"],
            "f1_macro_std": data["f1_macro_std"],
            "f1_weighted_mean": data["f1_weighted_mean"],
            "f1_weighted_std": data["f1_weighted_std"],
        }
        training_log.set_metrics(metrics_dict, split_metrics_dict)
        training_log.metadata = data["metadata"]

        return training_log

    def save(self, path: pathlib.Path) -> None:
        """Save the training log to disk."""
        torch.save(self.to_dict(), path)

    @classmethod
    def load(cls, path: pathlib.Path) -> TrainingLog:
        """Load the training log from disk."""
        data = torch.load(path)
        return cls.from_dict(data)

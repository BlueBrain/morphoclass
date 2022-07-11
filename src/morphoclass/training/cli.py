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
"""Utilities for the morphoclass train command."""
from __future__ import annotations

import collections
import dataclasses
import logging
import pathlib
from typing import Any
from typing import Sequence

import numpy as np
import sklearn
import torch
import torch_geometric
from imblearn.over_sampling import RandomOverSampler
from sklearn import metrics
from tqdm import tqdm

from morphoclass.data.morphology_data_loader import MorphologyDataLoader
from morphoclass.data.morphology_dataset import MorphologyDataset
from morphoclass.training import reset_seeds
from morphoclass.training.trainers import Trainer
from morphoclass.training.training_config import TrainingConfig
from morphoclass.training.training_log import TrainingLog
from morphoclass.utils import make_torch_deterministic
from morphoclass.utils import warn_if_nondeterministic
from morphoclass.vis import plot_confusion_matrix

logger = logging.getLogger(__name__)


# TODO: refactor the functions in this module to appropriate other modules.
#  There should be no module in `morphoclass.training` that is called `cli`,
#  All CLI-related functionality should be in the `morphoclass.console`
#  module. On the other hand, `run_training`, and other functions in this
#  module should not be in the `console` module, since they're used by other
#  non-CLI modules. In the worst case scenario this leads to circular imports,
#  e.g. `console` imports `something`, then `something` imports
#  `console.run_training`.


def run_training(dataset: MorphologyDataset, config: TrainingConfig) -> TrainingLog:
    """Training and evaluation of the model."""
    # Get a copy of the config - we might mutate it later
    config = dataclasses.replace(config)

    # Load pre-trained model
    if config.checkpoint_path_pretrained:
        logger.info(f"Pretrained on   : {config.checkpoint_path_pretrained}")
        checkpoint_pre = torch.load(config.checkpoint_path_pretrained)
        pretrained_model = checkpoint_pre["all"]["model"]
        labels_unique_str_pretrained = checkpoint_pre["labels_unique_str"]
        if config.dataset_name == checkpoint_pre["dataset_name"]:
            raise ValueError("Cannot pretrain on the same dataset.")
    else:
        pretrained_model = None
        labels_unique_str_pretrained = None

    make_torch_deterministic()

    # Check that all classes have at least 2 members, remove those classes
    # that have only one member
    dataset = prune_one_member_classes(dataset)

    all_labels = np.array([s.y for s in dataset])

    # labels sorted by their ID
    label_to_y: dict[str, int] = dataset.label_to_y
    labels_unique_str = sorted(label_to_y, key=lambda label: label_to_y[label])

    # Configure model_params
    # TODO: this modifies the config, which we shouldn't do. All model
    #       parameters should be saved in the training log and not in the config.
    if config.model_class.startswith("xgboost"):
        config.model_params["num_class"] = len(labels_unique_str)
        config.model_params["use_label_encoder"] = False  # suppress warning
    elif config.model_class.startswith("morphoclass"):
        if pretrained_model is not None:
            config.model_params["n_classes"] = len(labels_unique_str_pretrained)
        else:
            config.model_params["n_classes"] = len(labels_unique_str)

    # Set seed
    if config.seed is not None:
        reset_seeds(numpy_seed=config.seed, torch_seed=config.seed)

    # import splitter and splits
    splitter = config.splitter_cls(**config.splitter_params)
    split = splitter.split(X=all_labels, y=all_labels)  # X doesn't matter
    n_splits = splitter.get_n_splits(X=all_labels, y=all_labels)

    probabilities = np.empty((len(dataset), len(dataset.y_to_label)))
    predictions = np.empty(len(dataset), dtype=int)

    training_log = TrainingLog(config=config, labels_str=labels_unique_str)
    # SPLIT MODEL
    for n, (train_idx, val_idx) in enumerate(split):
        if config.oversampling:
            train_idx = oversample(train_idx, all_labels[train_idx], config.seed)

        logger.info(f"Split {n + 1}/{n_splits}, ratio: {len(train_idx)}:{len(val_idx)}")
        history = train_model(
            config,
            pretrained_model,
            train_idx,
            val_idx,
            dataset,
        )
        history["ground_truths"] = all_labels[val_idx]
        history["train_idx"] = list(train_idx)
        history["val_idx"] = list(val_idx)

        all_labels[val_idx] = history["ground_truths"]
        predictions[val_idx] = history["predictions"]
        probabilities[val_idx] = np.array(history["probabilities"])

        # Save split history
        training_log.add_split(history)

    # collect results
    training_log.set_y(all_labels, predictions, probabilities)

    # MODEL ALL
    if config.train_all_samples:
        logger.info("Fit model on all samples")
        train_idx = np.arange(len(dataset))
        if config.oversampling:
            train_idx = oversample(train_idx, all_labels[train_idx], config.seed)
        history = train_model(
            config,
            pretrained_model,
            train_idx,
            val_idx,
            dataset,
        )
        training_log.set_all_history(history)

    return training_log


def ask(msg: str) -> bool:
    """Ask an interactive question in the terminal."""
    return input(f"{msg} (y/[n]) ").strip().lower() == "y"


def prune_one_member_classes(dataset):
    """Prune classes with only one member."""
    # Find bad classes
    bad_classes: set[str] = set()
    label: str
    for label, count in collections.Counter(dataset.labels).items():
        if count == 1:
            bad_classes.add(label)

    # Nothing to do if no bad classes
    if not bad_classes:
        return dataset

    # Remove bad classes
    logger.warning(
        f"The following classes have only 1 member: {bad_classes}. "
        f"We'll remove them."
    )
    idx_keep = [
        idx for idx, item in enumerate(dataset) if item.y_str not in bad_classes
    ]

    return dataset.index_select(idx_keep)


def plot_confusion_matrices(
    img_dir: pathlib.Path | None, training_log: TrainingLog
) -> None:
    """Plot confusion matrices."""
    if img_dir is None:
        training_log.metrics["confusion_matrix"] = None
        for split_history in training_log.split_history:
            split_history["confusion_matrix"] = None
        return

    # all samples
    cm_file_name = img_dir / "cm_all.png"
    plot_confusion_matrix(
        training_log.metrics["confusion_matrix"],
        cm_file_name,
        labels=training_log.labels_str,
    )
    training_log.metrics["confusion_matrix"] = cm_file_name

    # splits
    for n, split_history in enumerate(training_log.split_history):
        cm_file_name = img_dir / f"cm_split{n}.png"
        plot_confusion_matrix(
            split_history["confusion_matrix"],
            cm_file_name,
            labels=training_log.labels_str,
        )
        split_history["confusion_matrix"] = cm_file_name


def split_metrics(splits: Sequence[dict]) -> dict[str, float]:
    """Compute average metrics across splits."""

    def mean_std(key):
        mean = np.mean([split[key] for split in splits])
        std = np.std([split[key] for split in splits])
        logger.info(f"{key}: {mean:.2f} (± {std:.2f})")

        return mean, std

    metrics_dict = {}
    metrics_dict["accuracy_mean"], metrics_dict["accuracy_std"] = mean_std(
        "val_acc_final"
    )
    metrics_dict["f1_micro_mean"], metrics_dict["f1_micro_std"] = mean_std("f1_micro")
    metrics_dict["f1_macro_mean"], metrics_dict["f1_macro_std"] = mean_std("f1_macro")
    metrics_dict["f1_weighted_mean"], metrics_dict["f1_weighted_std"] = mean_std(
        "f1_weighted"
    )

    return metrics_dict


def get_model(config, pretrained_state_dict, n_classes):
    """Reconstruct the model from the config file."""
    model = config.model_cls(**config.model_params)
    if pretrained_state_dict is None:
        return model

    # Now will load the pre-trained state dict
    if not config.model_class.startswith("morphoclass"):
        raise NotImplementedError("Pretraining only supported for morphoclass models.")

    # Load weights
    model.load_state_dict(pretrained_state_dict)
    if config.frozen_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace head
    n_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(n_features, n_classes)

    return model


def train_model(
    config,
    pretrained_model,
    train_idx,
    val_idx,
    dataset,
):
    """Train a model."""
    n_classes = max(dataset.y_to_label) + 1
    model = get_model(config, pretrained_model, n_classes)
    if config.model_class.startswith("sklearn") or config.model_class.startswith(
        "xgboost"
    ):
        history = train_ml_model(
            model=model,
            train_idx=train_idx,
            val_idx=val_idx,
            dataset=dataset,
        )
    elif config.model_class.startswith("morphoclass"):
        optimizer = config.optimizer_cls(model.parameters(), **config.optimizer_params)
        history = train_dm_model(
            model=model,
            train_idx=train_idx,
            val_idx=val_idx,
            dataset=dataset,
            optimizer=optimizer,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
        )
    else:
        raise ValueError(f"Model {config.model_class} is not supported!")

    return history


def train_ml_model(
    model: sklearn.base.BaseEstimator,
    dataset: MorphologyDataset,
    train_idx: torch_geometric.data.dataset.IndexType,
    val_idx: torch_geometric.data.dataset.IndexType,
) -> dict[str, Any]:
    """Train a sklearn-like model.

    Parameters
    ----------
    model
        A sklearn-like model with methods ``fit``, ``predict`` and
        ``predict_proba``.
    dataset
        A morphology dataset.
    train_idx
        The indices of the training set.
    val_idx
        The indices of the validation set.

    Returns
    -------
    dict
        The training history with the keys "model", "predictions",
        "ground_truths", "probabilities", "latent_features".
    """
    ds_train = dataset.index_select(train_idx)
    ds_val = dataset.index_select(val_idx)

    labels_train = np.array([sample.y for sample in ds_train])
    labels_val = np.array([sample.y for sample in ds_val])

    images_train = np.array([sample.image.numpy() for sample in ds_train])
    images_val = np.array([sample.image.numpy() for sample in ds_val])
    images_train = images_train.reshape(-1, 10_000)
    images_val = images_val.reshape(-1, 10_000)

    model.fit(images_train, labels_train)

    latent_features = np.empty((len(dataset), *images_train.shape[1:]))
    latent_features[train_idx] = images_train
    latent_features[val_idx] = images_val

    probabilities = model.predict_proba(images_val)
    labels_val_pred = model.predict(images_val)

    history = {
        "model": model,
        "predictions": labels_val_pred,
        "ground_truths": labels_val,
        "probabilities": probabilities,
        "latent_features": latent_features,
    }

    return history


def train_dm_model(
    model: torch.nn.Module,
    dataset: MorphologyDataset,
    train_idx: torch_geometric.data.dataset.IndexType,
    val_idx: torch_geometric.data.dataset.IndexType | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    batch_size: int | None = None,
    n_epochs: int | None = None,
    interactive: bool = False,
) -> dict[str, Any]:
    """Train morphoclass models."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    warn_if_nondeterministic(device)
    model = model.to(device)
    trainer = Trainer(model, dataset, optimizer, MorphologyDataLoader)
    history = trainer.train(
        n_epochs=n_epochs,
        batch_size=batch_size,
        train_idx=train_idx,
        val_idx=val_idx,
        progress_bar=tqdm if interactive else iter,
    )
    history["latent_features"] = trainer.get_latent_features(batch_size=batch_size)
    history["model"] = model.to("cpu").state_dict()

    return history


def collect_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, target_names: Sequence[str]
) -> dict:
    """Collect different evaluation metrics.

    Parameters
    ----------
    y_true : 1d array-like
        Ground truth labels
    y_pred : 1d array-like
        Predicted labels.
    target_names : array-like of shape (n_labels,)
        Names of the classes.

    Returns
    -------
    metrics_dict : dict
        The dictionary with all computed metrics. The keys are:

        * `classification_report`: The sklearn classification report.
        * `confusion_matrix`: The sklearn confusion matrix.
        * `accuracy`: The accuracy score.
        * `f1_micro`: The micro-averaged F1-score.
        * `f1_macro`: The macro-averaged F1-score.
        * `f1_weighted`: The weighted average of the F1-score.
    """
    cr = metrics.classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(target_names)),
        target_names=target_names,
        zero_division=1,
    )
    cm = metrics.confusion_matrix(
        [target_names[idx] for idx in y_true],
        [target_names[idx] for idx in y_pred],
        labels=target_names,
    )
    accuracy = metrics.accuracy_score(y_true, y_pred)
    f1_micro = metrics.f1_score(y_true, y_pred, average="micro")
    f1_macro = metrics.f1_score(y_true, y_pred, average="macro")
    f1_weighted = metrics.f1_score(y_true, y_pred, average="weighted")

    metrics_dict = {
        "classification_report": cr,
        "confusion_matrix": cm,
        "accuracy": accuracy,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }

    return metrics_dict


def _save_embedding(
    module: torch.nn.Module,
    input_: torch.Tensor,
    output: torch.Tensor,
    latent_features: list | None = None,
) -> None:
    """Save latent features produced by a PyTorch model or layer.

    This function is used as a PyTorch forward hook for the GNN, CNN, and PersLay models
    to save the morphology latent features produced by the models.

    Parameters
    ----------
    module
        The layer that this hook was registered to.
    input_
        The input tensors of the layer given as tuple: (Batch(...),).
    output
        The output tensors of the layer.
    latent_features
        If None then this hook is a no-op. This parameter has to be set
        via `partial` before the hook can be registered. It will collect the
        latent features from PyTorch model into the list.
    """
    if latent_features is not None:
        latent_features.append(output.cpu().detach().numpy())


def oversample(ids, labels, random_state=None):
    """Oversample to balance the label count."""
    oversampler = RandomOverSampler(
        sampling_strategy="minority", random_state=random_state
    )
    ids_over, _ = oversampler.fit_resample(np.array(ids).reshape(-1, 1), labels)
    logger.info(f"number of elements before over-sampling: {len(ids)}")
    logger.info(f"number of elements after over-sampling: {len(ids_over)}")

    return ids_over.ravel()

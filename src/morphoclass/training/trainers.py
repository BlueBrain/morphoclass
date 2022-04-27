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
"""A collection of model trainers."""
from __future__ import annotations

import logging
from collections.abc import Iterable
from collections.abc import Sequence
from copy import deepcopy
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as nnf
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch_geometric.data import Dataset

from morphoclass.data.morphology_data_loader import MorphologyDataLoader
from morphoclass.training import prepare_smart_split

logger = logging.getLogger(__name__)


class ConcateCNNetTrainer:
    """Trainer class for the ConcateCNNet network.

    Parameters
    ----------
    concatecnnet : morphoclass.models.ConcateCNNet
        An instance of the concate-cnnet model.
    dataset : morphoclass.data.MorphologyDataset
        The dataset with all morphologies (train and validation), the exact
        splits are specified in the `train_split` method.
    images : iterable
        The dataset with all persistence images (train and validation), the
        exact splits are specified in the `train_split` method.
    labels
        All data labels.
    optimizer : torch.optim.Optimizer
        an instance of a torch optimizer.
    """

    def __init__(self, concatecnnet, dataset, images, labels, optimizer):
        self.concatecnnet = concatecnnet
        self.dataset = dataset
        self.images = images
        self.labels = labels
        self.optimizer = optimizer

    def train_split(
        self, train_idx, val_idx, batch_size=32, n_epochs=500, verbose=False
    ):
        """Train the concate-cnnet on a given split.

        Parameters
        ----------
        train_idx : iterable of int
            The train set indices.
        val_idx : iterable of int
            The validation set indices.
        batch_size : int, default 32
            The batch size.
        n_epochs : int, default 500
            The number of epochs.
        verbose : bool, default False
            If true print the training progress statistics on the stdout.

        Returns
        -------
        probabilities : np.ndarray
            The history of predictions per epoch. Will have shape
            `(n_epochs, n_val_samples, n_classes)`.
        """
        if batch_size is None:
            batch_size = len(train_idx)

        ds_cnn_train = TensorDataset(self.images[train_idx], self.labels[train_idx])
        ds_cnn_val = TensorDataset(self.images[val_idx], self.labels[val_idx])
        train_loader_cnn = DataLoader(ds_cnn_train, batch_size=batch_size)
        val_loader_cnn = DataLoader(ds_cnn_val, batch_size=batch_size)

        ds_gnn_train, ds_gnn_val = prepare_smart_split(self.dataset, train_idx, val_idx)
        train_loader_gnn = MorphologyDataLoader(ds_gnn_train, batch_size=batch_size)
        val_loader_gnn = MorphologyDataLoader(ds_gnn_val, batch_size=batch_size)

        for x_data, (_x_img, x_y) in zip(ds_gnn_train, ds_cnn_train):
            assert x_data.y == x_y.item(), "Labels differ."
        for gnn_batch, (_img_batch, y_batch) in zip(train_loader_gnn, train_loader_cnn):
            assert all(gnn_batch.y == y_batch), "Batch labels differ"

        # Training
        n_classes = max(self.dataset.class_dict) + 1
        device = next(self.concatecnnet.parameters()).device

        probabilities = np.empty(
            shape=(n_epochs, len(val_idx), n_classes), dtype=np.float
        )
        for epoch in range(n_epochs):
            # Train
            self.concatecnnet.train()
            for gnn_batch, (image_batch, _label_batch) in zip(
                train_loader_gnn, train_loader_cnn
            ):
                gnn_batch = gnn_batch.to(device)
                image_batch = image_batch.to(device)

                self.optimizer.zero_grad()
                out = self.concatecnnet(gnn_batch, image_batch)
                loss = nnf.nll_loss(out, gnn_batch.y)
                loss.backward()
                self.optimizer.step()

            # Eval
            self.concatecnnet.eval()
            preds = []
            with torch.no_grad():
                for gnn_batch, (image_batch, _label_batch) in zip(
                    val_loader_gnn, val_loader_cnn
                ):
                    gnn_batch = gnn_batch.to(device)
                    image_batch = image_batch.to(device)

                    out = self.concatecnnet(gnn_batch, image_batch)
                    pred = torch.exp(out).cpu().numpy()
                    preds.append(pred)
            preds = np.concatenate(preds)
            probabilities[epoch] = preds

            # Print progress
            if (epoch + 1) % 50 == 0 and verbose:
                print(f"[epoch {epoch + 1:3d}]")

        return probabilities


class ConcateNetTrainer:
    """Trainer class for the ConcateNet network.

    Parameters
    ----------
    concatenet : morphoclass.models.ConcateNet
        An instance of the concate-net model.
    dataset : morphoclass.data.MorphologyDataset
        The dataset with all morphologies (train and validation), the exact
        splits are specified in the `train_split` method.
    diagrams
        All persistence diagrams (train and validation), the exact splits are
        specified in the `train_split` method.
    labels
        All data labels.
    optimizer : torch.optim.Optimizer
        an instance of a torch optimizer.

    """

    def __init__(self, concatenet, dataset, diagrams, labels, optimizer):
        self.concatenet = concatenet
        self.dataset = dataset
        self.diagrams = diagrams
        self.labels = labels
        self.optimizer = optimizer

    @staticmethod
    def perslay_collate_fn(samples):
        """Batch together persistence diagrams and their labels.

        Parameters
        ----------
        samples : iterable of tuple
            The sample for batching. Each element in the iterable is a tuple
            with a persistence diagram tensor and a label.

        Returns
        -------
        diagram_batch : torch.Tensor
            A batch of persistence diagrams.
        y_batch : torch.Tensor
            A batch of labels.
        point_index : torch.Tensor
            Segmentation map for samples in the batch.
        """
        diagrams = []
        labels = []
        point_index = []
        for i, (x, y) in enumerate(samples):
            diagrams.append(x)
            labels.append(y)
            point_index += [i] * len(x)

        return (
            torch.cat(diagrams, dim=0),
            torch.tensor(labels),
            torch.tensor(point_index),
        )

    def train_split(
        self, train_idx, val_idx, batch_size=32, n_epochs=500, verbose=False
    ):
        """Train the concate-net on a given split.

        Parameters
        ----------
        train_idx : iterable of int
            The train set indices.
        val_idx : iterable of int
            The validation set indices.
        batch_size : int, default 32
            The batch size.
        n_epochs : int, default 500
            The number of epochs.
        verbose : bool, default False
            If true print the training progress statistics on the stdout.

        Returns
        -------
        probabilities : np.ndarray
            The history of predictions per epoch. Will have shape
            `(n_epochs, n_val_samples, n_classes)`.
        """
        if batch_size is None:
            batch_size = len(train_idx)

        diagrams = torch.stack(self.diagrams)
        labels = torch.tensor(self.labels)
        scale = diagrams[train_idx].max()
        ds_train = TensorDataset(diagrams[train_idx] / scale, labels[train_idx])
        ds_val = TensorDataset(diagrams[val_idx] / scale, labels[val_idx])
        # ds_pl_train = [(self.diagrams[idx], self.labels[idx]) for idx in train_idx]
        # ds_pl_val = [(self.diagrams[idx], self.labels[idx]) for idx in val_idx]
        # scale = max(d.max() for d, l in ds_pl_train)
        # ds_pl_train = [
        #     (torch.tensor(d / scale, dtype=torch.float), l) for d, l in ds_pl_train
        # ]
        # ds_pl_val = [
        #     (torch.tensor(d / scale, dtype=torch.float), l) for d, l in ds_pl_val
        # ]
        # Prepare data loaders
        train_loader_pl: DataLoader = DataLoader(
            ds_train,
            batch_size=batch_size,
            collate_fn=self.perslay_collate_fn,
        )
        val_loader_pl: DataLoader = DataLoader(
            ds_val,
            batch_size=batch_size,
            collate_fn=self.perslay_collate_fn,
        )

        ds_gnn_train, ds_gnn_val = prepare_smart_split(self.dataset, train_idx, val_idx)
        train_loader_gnn = MorphologyDataLoader(ds_gnn_train, batch_size=batch_size)
        val_loader_gnn = MorphologyDataLoader(ds_gnn_val, batch_size=batch_size)

        for x_data, (_x_diagram, x_y) in zip(ds_gnn_train, ds_train):
            assert x_data.y == x_y.item(), "Labels differ."
        for gnn_batch, (_diagram_batch, y_batch, _point_index) in zip(
            train_loader_gnn, train_loader_pl
        ):
            assert all(gnn_batch.y == y_batch), "Batch labels differ"

        # Training
        n_classes = max(self.dataset.class_dict) + 1
        device = next(self.concatenet.parameters()).device

        probabilities = np.empty(
            shape=(n_epochs, len(val_idx), n_classes), dtype=np.float
        )
        for epoch in range(n_epochs):
            # Train
            self.concatenet.train()
            for gnn_batch, (diagram_batch, _y_batch, point_index) in zip(
                train_loader_gnn, train_loader_pl
            ):
                gnn_batch = gnn_batch.to(device)
                diagram_batch = diagram_batch.to(device)
                point_index = point_index.to(device)

                self.optimizer.zero_grad()
                out = self.concatenet(gnn_batch, diagram_batch, point_index)
                loss = nnf.nll_loss(out, gnn_batch.y)
                loss.backward()
                self.optimizer.step()

            # Eval
            self.concatenet.eval()
            preds = []
            with torch.no_grad():
                for gnn_batch, (diagram_batch, _y_batch, point_index) in zip(
                    val_loader_gnn, val_loader_pl
                ):
                    gnn_batch = gnn_batch.to(device)
                    diagram_batch = diagram_batch.to(device)
                    point_index = point_index.to(device)

                    out = self.concatenet(gnn_batch, diagram_batch, point_index)
                    pred = torch.exp(out).cpu().numpy()
                    preds.append(pred)
            preds = np.concatenate(preds)
            probabilities[epoch] = preds

            # Print progress
            if (epoch + 1) % 50 == 0 and verbose:
                print(f"[epoch {epoch + 1:3d}]")

        return probabilities


class Trainer:
    """A trainer for morphology classifiers.

    Parameters
    ----------
    net
        A morphoclass classifier instance.
    dataset
        A morphology dataset.
    optimizer
        An optimizer instance.
    loader_class
        The morphology dataset loader class.
    """

    def __init__(
        self,
        net: nn.Module,
        dataset: Dataset,
        optimizer: Optimizer,
        loader_class: type[DataLoader],
    ) -> None:
        self.net = net
        self.dataset = dataset
        self.optimizer = optimizer
        self.loader_class = loader_class
        self.device = next(self.net.parameters()).device

    def data_loader(
        self,
        idx: Sequence[int] | None = None,
        batch_size: int = 1,
        shuffle: bool = False,
    ) -> DataLoader:
        """Construct a data loader for a given data subset."""
        if idx is None:
            idx = torch.arange(0, len(self.dataset))
        if len(idx) == 0:
            raise ValueError("Can't construct a data loader for an empty index list")
        loader = self.loader_class(
            self.dataset.index_select(idx),
            batch_size=batch_size,
            shuffle=shuffle,
        )
        return loader

    def train(
        self,
        n_epochs: int,
        batch_size: int,
        train_idx: Sequence[int],
        val_idx: Sequence[int] | None = None,
        load_best: bool = False,
        progress_bar: Callable[[Iterable], Iterable] = iter,
    ) -> dict:
        """Train and evaluate on dataset subsets specified by indices.

        Parameters
        ----------
        train_idx : array of int
            The indices of the training samples.
        val_idx : array of int
            The indices of the evaluation samples.
        batch_size : int
            The batch size.
        n_epochs : int
            The number of epochs.
        load_best : bool
            If true the model with the best validation accuracy will be
            restored at the end of training. Only possible if `val_idx` is not
            `None`.
        save_latent_features : bool
            If true the latent features of the feature extractor part of the
            model will be saved under `history["latent_features"]`.
        progress_bar : callable
            A callable that wraps an iterable over the epoch numbers and creates
            a progress bar.

        Returns
        -------
        history : dict
            Dictionary with predictions, probabilities, training and
            validation losses and accuracies.
        """
        # Parameter value checks
        if val_idx is None and load_best:
            raise ValueError("load_best=True only possible if val_idx is not None")

        # Construct data loaders
        train_loader = self.data_loader(train_idx, batch_size, shuffle=True)

        # Training
        best_val_acc = 0.0
        best_state_dict = None
        train_loss_history = []
        train_acc_history = []
        val_loss_history = []
        val_acc_history = []

        for epoch in progress_bar(range(n_epochs)):
            self.net.train()
            for batch in train_loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                out = self.net(batch)
                loss = nnf.nll_loss(out, batch.y)
                loss.backward()
                self.optimizer.step()

            # Eval
            train_losses, train_logits, train_labels = self.predict(
                train_idx, batch_size
            )
            train_loss = train_losses.mean().item()
            train_acc = self.acc(train_labels, train_logits)
            train_loss_history.append(train_loss)
            train_acc_history.append(train_acc)

            if val_idx is not None:
                val_losses, val_logits, val_labels = self.predict(val_idx, batch_size)
                val_loss = val_losses.mean().item()
                val_acc = self.acc(val_labels, val_logits)
                val_loss_history.append(val_loss)
                val_acc_history.append(val_acc)

            # Print info
            if epoch % 50 == 0:
                msg = f"[{epoch:5d}] acc: {train_acc:.1%} loss: {train_loss:.2f}"
                if val_idx is not None:
                    msg += f" (val acc: {val_acc:.1%} val loss: {val_loss:.2f})"
                logger.info(msg)

            # Checkpoint best model
            if load_best and val_acc > best_val_acc:
                logger.info(
                    f"Saving best model at epoch {epoch} with val_acc={val_acc:.1%}"
                )
                best_val_acc = val_acc
                best_state_dict = deepcopy(self.net.state_dict())

        # Load the checkpointed model
        if load_best and best_state_dict is not None:
            logger.info("Loading the best model")
            self.net.load_state_dict(best_state_dict)

        # Evaluate on final model and create the history dict
        train_losses, train_logits, train_labels = self.predict(train_idx, batch_size)
        history = {
            "train_loss_final": train_losses.mean().item(),
            "train_acc_final": self.acc(train_labels, train_logits),
            "train_loss": train_loss_history,
            "train_acc": train_acc_history,
        }
        if val_idx is not None:
            val_losses, val_logits, val_labels = self.predict(val_idx, batch_size)
            history.update(
                {
                    "probabilities": val_logits.exp().cpu().numpy(),
                    "predictions": val_logits.argmax(dim=1).cpu().numpy(),
                    "val_loss_final": val_losses.mean().cpu().numpy(),
                    "val_acc_final": self.acc(val_labels, val_logits),
                    "val_loss": val_loss_history,
                    "val_acc": val_acc_history,
                }
            )
            logger.info(f'Final validation accuracy: {history["val_acc_final"]:.1%}')

        logger.info("Done.")
        return history

    def predict(
        self,
        idx: Sequence[int] | None = None,
        batch_size: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run inference on a data subset, get losses, probabilities, and labels.

        Parameters
        ----------
        idx
            An index subset of the dataset. If not given then the entire
            dataset is used.
        batch_size
            The batch size at which to process data. The results don't depend
            on the batch size. However, bigger batch size mean faster inference,
            but a too big batch size might exhaust the memory.

        Returns
        -------
        losses : torch.Tensor
            The non-reduced loss per sample.
        logits : torch.Tensor
            The predicted logits.
        labels : torch.Tensor
            All sample labels.
        """
        self.net.eval()
        logits = []
        losses = []
        labels = []
        with torch.no_grad():
            for batch in self.data_loader(idx, batch_size):
                batch = batch.to(self.device)
                out = self.net(batch)
                loss = nnf.nll_loss(out, batch.y, reduction="none")
                logits.append(out)
                losses.append(loss)
                labels.append(batch.y)

        return torch.cat(losses), torch.cat(logits), torch.cat(labels)

    @staticmethod
    def acc(labels: torch.Tensor, probas: torch.Tensor) -> float:
        """Compute the accuracy score given labels and probabilities."""
        correct: int = torch.eq(labels, probas.argmax(dim=1)).sum().item()
        return correct / len(labels)

    def get_latent_features(
        self,
        idx: Sequence[int] | None = None,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """Compute the forward pass and collect the latent features.

        Parameters
        ----------
        idx
            A sequence of indices specifying a data subset. If none then the
            whole dataset will be used.
        batch_size
            The batch size to use for the forward pass.

        Returns
        -------
        torch.Tensor
            A tensor of shape ``(n_samples, *n_features)`` that contain the
            activations of the feature extractor part of the model.
        """
        buffer: list[torch.Tensor] = []

        def hook(_module, _inputs, output):
            buffer.append(output)

        hook_handle = self.net.feature_extractor.register_forward_hook(hook)
        with torch.no_grad():
            for batch in self.data_loader(idx, batch_size):
                self.net(batch.to(self.device))
        hook_handle.remove()

        # buffer is a list of length n_calls and each element in it is a
        # tensor of shape (batch_size, n_features). After torch.cat we get the
        # shape (n_samples, n_features).
        return torch.cat(buffer).detach()

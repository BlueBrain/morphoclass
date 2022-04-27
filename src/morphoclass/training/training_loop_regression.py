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
"""Functions for running training of a regression model."""
from __future__ import annotations

from collections import defaultdict

import torch
from torch import nn

from morphoclass import data
from morphoclass import utils


def get_mse_loss(model, loader, device):
    """Compute the mean squared error loss.

    Parameters
    ----------
    model
        Model for the forward pass
    loader
        A data loader for all data for the loss.
    device
        A torch device.

    Returns
    -------
    float
        The mean squared error loss on all data in the `loader`.
    """
    model.eval()
    losses = []
    n = 0
    loss_fn = nn.MSELoss(reduction="sum")

    with torch.no_grad():
        for batch in iter(loader):
            batch = batch.to(device)
            y_pred = model(batch).flatten()
            loss = loss_fn(y_pred, batch.y)
            losses.append(loss.item())
            n += len(batch.y)

    return sum(losses) / n


def train_regression_model(
    model_fn,
    dataset_train,
    dataset_val,
    n_epochs=500,
    batch_size=None,
    lr=5e-4,
    wd=5e-3,
    optimizer_cls=torch.optim.Adam,
    device=None,
    pin_memory=False,
    n_workers_train=0,
    n_workers_val=0,
    silent=False,
):
    """Train a model with leave-one-out and save the training history.

    Parameters
    ----------
    model_fn : callable
        Factory function returning a model instance.
    dataset_train : pytorch_geometric.data.Data
        The training set.
    dataset_val : pytorch_geometric.data.Data
        The validation set.
    n_epochs : int
        The number of epochs.
    batch_size : int
        The batch size.
    lr : float
        The value of the learning rate.
    wd : float
        The value of the weight decay.
    optimizer_cls : callable
        Class for instantiating a PyTorch optimizer.
    device : str or torch.device
        The device for training.
    pin_memory : bool
        Passed through to the `MorphologyDataLoader` class for the train and
        validation data loader.
    n_workers_train : int
        The number of workers for training.
    n_workers_val : int
        The number of workers for the validation.
    silent : bool, default False
        If true print training and validation losses to stdout every 50 epochs.

    Returns
    -------
    model
        The trained model.
    history : dict
        History of training losses.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    utils.warn_if_nondeterministic(device)

    if batch_size is None:
        batch_size = len(dataset_train)

    train_loader = data.MorphologyDataLoader(
        dataset_train,
        batch_size=batch_size or len(dataset_train),
        pin_memory=pin_memory,
        num_workers=n_workers_train,
    )

    val_loader = data.MorphologyDataLoader(
        dataset_val,
        batch_size=batch_size or len(dataset_val),
        pin_memory=pin_memory,
        num_workers=n_workers_val,
    )

    n_features = dataset_train[0].x.shape[-1]
    model = model_fn(n_features=n_features, n_classes=1).to(device)
    optimizer = optimizer_cls(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()

    history = defaultdict(list)

    if not silent:
        print("Starting training loop")
    for epoch in range(n_epochs + 1):
        # Train
        model.train()
        for batch in iter(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch).flatten()

            loss = loss_fn(out, batch.y)
            loss.backward()
            optimizer.step()

        # Eval
        train_loss = get_mse_loss(model, train_loader, device)
        val_loss = get_mse_loss(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Print info
        if epoch % 50 == 0 and not silent:
            print(
                f"[Epoch {epoch:5d}] "
                f"(loss: {train_loss:.2f}, val_loss: {val_loss:.2f})"
            )

    return model, history

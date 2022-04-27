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
"""Functions for simple model training without cross-validation."""
from __future__ import annotations

import collections
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as nnf

from morphoclass import data
from morphoclass import utils


def train_model(
    model_fn,
    dataset_train,
    dataset_val,
    n_epochs=500,
    batch_size=None,
    acc_threshold=0.0,
    lr=5e-4,
    wd=5e-3,
    optimizer_cls=torch.optim.Adam,
    device=None,
    pin_memory=False,
    n_workers_train=0,
    n_workers_val=0,
    silent=False,
):
    """Train a model with cross-validation and save loss and accuracy.

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
    acc_threshold : float
        Start checkpointing the best performing model once the accuracy
        surpasses this value. This is useful to avoid too many checkpoints at
        the beginning when the accuracy improves at almost every step.
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
    silent : bool
        If true the training progress will be logged to stdout.

    Returns
    -------
    history : dict
        History of training and validation losses and accuracies.
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
    model = model_fn(n_features=n_features).to(device)
    optimizer = optimizer_cls(model.parameters(), lr=lr, weight_decay=wd)

    history = collections.defaultdict(list)
    best_flat_val_acc = acc_threshold
    best_flat_state_dict = None

    if not silent:
        print("Starting training loop")
    for epoch in range(n_epochs + 1):
        # Train
        model.train()
        for batch in iter(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)

            loss = nnf.nll_loss(out, batch.y)
            loss.backward()
            optimizer.step()

        # Eval
        model.eval()
        train_eval = [model.loss_acc(batch.to(device)) for batch in iter(train_loader)]

        val_eval = [model.loss_acc(batch.to(device)) for batch in iter(val_loader)]

        train_loss, train_acc = np.array(train_eval).mean(axis=0)
        val_loss, val_acc = np.array(val_eval).mean(axis=0)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Print info
        if epoch % 50 == 0 and not silent:
            print(
                f"[Epoch {epoch:5d}] "
                f"acc: {train_acc:.2f}, val_acc: {val_acc:.2f} "
                f"(loss: {train_loss:.2f}, val_loss: {val_loss:.2f})"
            )

        # Checkpoint best model
        if val_acc > best_flat_val_acc:
            if not silent:
                print(
                    "  > Checkpoint model at epoch "
                    f"{epoch} with val_acc={val_acc:.2f}"
                )
            best_flat_val_acc = val_acc
            best_flat_state_dict = deepcopy(model.state_dict())

    # Load saved model
    if best_flat_state_dict is not None:
        if not silent:
            print("Loading the best-performing model...")
        model.load_state_dict(best_flat_state_dict)

    # Evaluate on loaded model
    val_loss, val_acc = model.loss_acc(next(iter(val_loader)).to(device))
    if not silent:
        print(f"Final validation accuracy: {val_acc:.2f}")
        print("Done.")

    return model, history

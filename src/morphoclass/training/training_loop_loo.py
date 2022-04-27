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
"""Functions for model training with the leave-one-out strategy."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as nnf
from sklearn import model_selection

from morphoclass import data
from morphoclass import utils


def train_model_loo(
    model_fn,
    dataset,
    dataset_prep_fn,
    n_classes,
    n_epochs=500,
    batch_size=None,
    lr=5e-4,
    wd=5e-3,
    optimizer_cls=torch.optim.Adam,
    device=None,
    pin_memory=False,
    n_workers_train=0,
    n_workers_val=0,
    # show_widgets=True,
):
    """Train a model with leave-one-out and save the training history.

    Parameters
    ----------
    model_fn : callable
        Factory function returning a model instance.
    dataset : pytorch_geometric.data.Data
        The full dataset.
    dataset_prep_fn : callable
        A function for splitting and pre-processing the dataset. Should have the
        signature (dataset, train_idx, val_idx) and return a tuple of the form
        (dataset_train, dataset_val).
    n_classes : int
        The number of classes to predict. Needed ot allocate the prediction
        array for the history cache.
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

    Returns
    -------
    history : dict
        History of training and validation losses, accuracies, prediction
        probabilities, and predictions.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    utils.warn_if_nondeterministic(device)

    # Initialise lists for losses and accuracies
    hist_train_acc = []
    hist_val_acc = []
    hist_train_loss = []
    hist_val_loss = []
    hist_probabilities = np.zeros((len(dataset), n_epochs, n_classes))

    # Set up the progress bars
    # progressbar_splits = None
    # progressbar_epochs = None
    # if show_widgets:
    #     progressbar_splits = IntProgress(
    #         min=0, max=len(dataset), description="CV Split", bar_style="success"
    #     )
    #     progressbar_epochs = IntProgress(
    #         min=0, max=n_epochs, description="Epoch", bar_style="info"
    #     )
    #     display(progressbar_splits)
    #     display(progressbar_epochs)

    # CV Loop
    split = model_selection.LeaveOneOut().split(dataset)
    for n, (train_idx, val_idx) in enumerate(split):
        print(f"Split {n + 1} / {len(dataset)} ...")

        dataset_train, dataset_val = dataset_prep_fn(dataset, train_idx, val_idx)

        train_loader = data.MorphologyDataLoader(
            dataset_train,
            batch_size=batch_size or len(dataset_train),
            pin_memory=pin_memory,
            num_workers=n_workers_train,
        )

        val_loader = data.MorphologyDataLoader(
            dataset_val, pin_memory=pin_memory, num_workers=n_workers_val
        )

        n_features = dataset_train[0].x.shape[-1]
        model = model_fn(n_features=n_features).to(device)
        optimizer = optimizer_cls(model.parameters(), lr=lr, weight_decay=wd)

        # Initialise lists for losses and accuracies in the current split
        split_train_acc = []
        split_val_acc = []

        split_train_loss = []
        split_val_loss = []

        # Run the training loop on the current split
        # if show_widgets:
        #     progressbar_epochs.value = 0
        for epoch in range(n_epochs):
            model.train()
            # Optimisation step
            for batch in train_loader:
                optimizer.zero_grad()
                out = model(batch.to(device))
                loss = nnf.nll_loss(out, batch.y)
                loss.backward()
                optimizer.step()

            # Evaluation
            with torch.no_grad():
                model.eval()
                train_eval = [
                    model.loss_acc(batch.to(device)) for batch in iter(train_loader)
                ]

                val_eval = [
                    model.loss_acc(batch.to(device)) for batch in iter(val_loader)
                ]

                train_loss, train_acc = np.array(train_eval).mean(axis=0)
                val_loss, val_acc = np.array(val_eval).mean(axis=0)

                probas = model(next(iter(val_loader)).to(device))
                probas = torch.exp(probas).detach().cpu().numpy()

                split_train_acc.append(train_acc)
                split_val_acc.append(val_acc)

                split_train_loss.append(train_loss)
                split_val_loss.append(val_loss)

                hist_probabilities[val_idx, epoch] = probas

            # Update the epochs progress bar
            # if show_widgets:
            #     progressbar_epochs.value += 1

        # Save the losses and accuracies of the epoch
        hist_train_acc.append(split_train_acc)
        hist_val_acc.append(split_val_acc)

        hist_train_loss.append(split_train_loss)
        hist_val_loss.append(split_val_loss)

        # Update the splits progress bar
        # if show_widgets:
        #     progressbar_splits.value += 1

    hist_predictions = hist_probabilities.argmax(axis=2)
    history = {
        "train_acc": np.array(hist_train_acc),
        "val_acc": np.array(hist_val_acc),
        "train_loss": np.array(hist_train_loss),
        "val_loss": np.array(hist_val_loss),
        "predictions": hist_predictions,
        "probabilities": hist_probabilities,
    }

    print("Done.")

    return history

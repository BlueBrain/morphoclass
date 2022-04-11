"""Functions for model training with cross-validation."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as nnf
from sklearn import model_selection

from morphoclass import data
from morphoclass import utils


def train_model_cv(
    model_fn,
    dataset,
    dataset_prep_fn,
    n_epochs=500,
    batch_size=None,
    n_splits=5,
    lr=5e-4,
    wd=5e-3,
    optimizer_cls=torch.optim.Adam,
    device=None,
    pin_memory=False,
    n_workers_train=0,
    n_workers_val=0,
    cv_random_state=None,
    # show_widgets=True,
):
    """Train a model with cross-validation and save loss and accuracy.

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
    n_epochs : int
        The number of epochs.
    batch_size : int
        The batch size.
    n_splits : int
        The number of cross-validation splits.
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
    cv_random_state : int
        A random seed for the stratified K-fold split.

    Returns
    -------
    history : dict
        History of training and validation losses and accuracies.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    utils.warn_if_nondeterministic(device)

    # CV Strategy
    k_fold = model_selection.StratifiedKFold(
        n_splits, shuffle=True, random_state=cv_random_state
    )
    targets = [s.y for s in dataset]

    # Initialise lists for losses and accuracies
    hist_train_acc = []
    hist_val_acc = []

    hist_train_loss = []
    hist_val_loss = []

    # Set up the progress bars
    # progressbar_splits = None
    # progressbar_epochs = None
    # if show_widgets:
    #     progressbar_splits = IntProgress(min=0, max=n_splits,
    #                                      description='CV Split',
    #                                      bar_style='success')
    #     progressbar_epochs = IntProgress(min=0, max=n_epochs,
    #                                      description='Epoch',
    #                                      bar_style='info')
    #     display(progressbar_splits)
    #     display(progressbar_epochs)

    # CV Loop
    split = k_fold.split(X=targets, y=targets)  # X doesn't matter
    for n, (train_idx, val_idx) in enumerate(split):
        print(f"Split {n + 1} / {n_splits} ...")

        dataset_train, dataset_val = dataset_prep_fn(dataset, train_idx, val_idx)

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

        # Initialise lists for losses and accuracies in the current split
        split_train_acc = []
        split_val_acc = []

        split_train_loss = []
        split_val_loss = []

        # Run the training loop on the current split
        # if show_widgets:
        #     progressbar_epochs.value = 0
        for _epoch in range(n_epochs):
            model.train()
            # Optimisation step
            for batch in train_loader:
                optimizer.zero_grad()
                out = model(batch.to(device))
                loss = nnf.nll_loss(out, batch.y)
                loss.backward()
                optimizer.step()

            # Evaluation
            model.eval()
            train_eval = [
                model.loss_acc(batch.to(device)) for batch in iter(train_loader)
            ]

            val_eval = [model.loss_acc(batch.to(device)) for batch in iter(val_loader)]

            train_loss, train_acc = np.array(train_eval).mean(axis=0)
            val_loss, val_acc = np.array(val_eval).mean(axis=0)

            split_train_acc.append(train_acc)
            split_val_acc.append(val_acc)

            split_train_loss.append(train_loss)
            split_val_loss.append(val_loss)

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

    history = {
        "train_acc": np.array(hist_train_acc),
        "val_acc": np.array(hist_val_acc),
        "train_loss": np.array(hist_train_loss),
        "val_loss": np.array(hist_val_loss),
    }

    print("Done.")

    return history

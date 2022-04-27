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
from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from morphoclass.utils import np_temp_seed


def reset_seeds(numpy_seed=0, torch_seed=0):
    """Reset random seeds for numpy and torch.

    Parameters
    ----------
    numpy_seed : int or None
        The random seed for numpy. If None then the seed
        won't be set.
    torch_seed : int or None
        The random seed for torch. If None then the seed
        won't be set.
    """
    if numpy_seed is not None:
        np.random.seed(numpy_seed)
    if torch_seed is not None:
        torch.manual_seed(torch_seed)


def create_k_folds(n_splits, labels, seed):
    """Generate indices for a stratified k-fold split.

    For a given seed the generated results will always
    be the same.

    While the validation indices might be sorted, the
    training indices are randomly permuted. Therefore
    it is not necessary to shuffle the samples at
    training time any more.

    Parameters
    ----------
    n_splits : int
        The number of splits in the k-fold
    labels : list_like
        The labels upon which to generate splits.
    seed : int
        A random seed to ensure determinism.

    Returns
    -------
    folds : list
        A list of length `n_splits` containing tuples of the
        form `(train_idx, val_idx)`.
    """
    k_fold = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    folds = []
    for train_idx, val_idx in k_fold.split(labels, labels):
        with np_temp_seed(seed):
            train_idx = np.random.permutation(train_idx)
        folds.append((train_idx, val_idx))

    return folds

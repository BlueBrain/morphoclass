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
"""Morphology data loader."""
from __future__ import annotations

import functools
from typing import Any

import numpy as np
import torch
import torch_geometric
from torch.utils.data.dataloader import DataLoader
from torch_geometric.data import Batch

from morphoclass.data.morphology_dataset import MorphologyDataset


class MorphologyDataLoader(DataLoader):
    """A data loader for the morphology data set.

    This class is derived from `torch.utils.data.DataLoader` and unlike
    `torch_geometric.data.DataLoader` is able to handle `Data` objects with
    non-numeric fields. These fields are simply ignored upon constructing
    batches.

    Parameters
    ----------
    dataset
        The data set to apply the data loader to.
    kwargs
        Further parameter to pass on to the PyTorch DataLoader base class.
    """

    def __init__(self, dataset: MorphologyDataset, **kwargs: Any) -> None:
        # follow_batch lists fields in data, usually tensors of variable
        # length, for which batch segmentation tensors shall be created. E.g.
        # a data object may have the "diagram" field of variable length.
        # Because we list it in "follow_batch", the batch object will contain
        # an additional field "diagram_batch", which contains the batch
        # segmentation for "diagram". The segmentation is constructed in the
        # usual way - it's a vector of integers, where "0" marks the points
        # from the first sample, "1" from the second etc.
        collate_fn = functools.partial(_collate_fn, follow_batch=["diagram"])
        super().__init__(dataset, collate_fn=collate_fn, **kwargs)


def _collate_fn(data_list, follow_batch):
    """Collate a list of morphology sample to a batch.

    This method is derived from torch_geometric.data.Batch, and
    modified to ignore keys in samples that cannot be collated.

    Parameters
    ----------
    data_list
        List of `Data` objects to be collated to a batch.
    follow_batch
        List of field names for which to create segmentation masks for
        batches.

    Returns
    -------
    batch
        The batch of collated data objects form `data_list`.
    """
    # Step 1: collect all data fields that will be batched
    keys: set[str] = set()
    for data in data_list:
        keys = keys.union(data.keys)

    if "batch" in keys:
        raise ValueError("Trying to batch data that is already batched.")

    # Step 2: initialise the batch object
    batch = Batch()
    batch.__data_class__ = data_list[0].__class__
    batch.__slices__ = {key: [0] for key in keys}

    # Step 3: batching happens in 2 steps: first the data from objects
    # is collected in a list. Only after all data have been collected
    # the lists are concatenated to a data batch
    for key in keys:
        batch[key] = []

    # The field names that were provided via "follow_batch" will get
    # additional segmentation masks for batch elements.
    # Initialise all custom key segmentation masks with empty lists
    for key in follow_batch:
        if key in keys:
            batch[f"{key}_batch"] = []

    # Index shifts for index tensors (see below)
    idx_shifts = {key: 0 for key in keys}

    # Step 4: collect the data to batch into lists in preparation of the
    # concatenation
    batch.batch = []
    for i, data in enumerate(data_list):  # Enumerate samples
        for key in data.keys:  # Iterate keys in sample
            # Ignore keys which cannot be collated
            if not (
                torch.is_tensor(data[key])
                or isinstance(data[key], int)
                or isinstance(data[key], float)
            ):
                continue

            # Convert numpy arrays to torch tensors
            if isinstance(data[key], np.ndarray):
                data[key] = torch.from_numpy(data[key].copy()).float()

            # If the current tensor contains indices (e.g. edge_index)
            # then the indices in the batch need to be shifted. For
            # example, the concatenation of index tensors [0, 2, 1] and
            # [1, 0, 2] should be [0, 2, 1, 4, 3, 5] and not
            # [0, 2, 1, 1, 0, 2]. For non-index tensors idx_shifts[key]
            # will always be zero (see data.__inc__ below)
            item = data[key] + idx_shifts[key]

            # Update the batch slices
            if torch.is_tensor(data[key]):
                size = data[key].size(data.__cat_dim__(key, data[key]))
            else:
                size = 1
            batch.__slices__[key].append(size + batch.__slices__[key][-1])

            # Update the shift for index tensors
            idx_shifts[key] += data.__inc__(key, item)

            # Put the data to batch in a temporary list. The concatenation
            # will be done later.
            batch[key].append(item)

            # Update the batch segmentation masks for relevant keys
            if key in follow_batch:
                item = torch.full((size,), i, dtype=torch.long)
                batch[f"{key}_batch"].append(item)

        # Update batch segmentation mask with current sample
        num_nodes = data.num_nodes
        if num_nodes is not None:
            item = torch.full((num_nodes,), i, dtype=torch.long)
            batch.batch.append(item)

    # No batch segmentation mask was created, delete the field
    if len(batch.batch) == 0:
        batch.batch = None

    # Cat lists of tensors / ints / floats into tensors
    for key in batch.keys:
        # Remove keys for which no data was collected
        if key.startswith("__") or len(batch[key]) == 0:
            batch[key] = None
            continue

        # Cat lists to tensors
        item = batch[key][0]
        if torch.is_tensor(item):
            batch[key] = torch.cat(batch[key], dim=data_list[0].__cat_dim__(key, item))
        elif isinstance(item, int) or isinstance(item, float):
            batch[key] = torch.tensor(batch[key])
        else:
            raise ValueError("Unsupported attribute type")

    num_nodes = [data.num_nodes for data in data_list]
    if all(num is not None for num in num_nodes):
        batch.num_nodes = sum(num_nodes)

    if torch_geometric.is_debug_enabled():
        batch.debug()

    return batch.contiguous()
